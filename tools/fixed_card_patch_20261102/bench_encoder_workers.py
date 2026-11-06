"""Compare original/fixed-device two-worker Encoder dispatch on 32 records; no DDP."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path('/yinghepool/tech-dcu/zhaoyao')
_FIXED_DEVICE = None


class StopDiagnostic(BaseException):
    """Escape production batch retry, then propagate as a worker error."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def configure(root):
    os.chdir(root)
    for path in (root, root / 'model_design'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def trace_job(job, D, P, torch, np, trace_dir, expected):
    index, raw, checkpoint, device, _, _, _, _ = job
    trace_path = Path(trace_dir) / f'worker_{os.getpid()}.jsonl'

    def event(kind, **details):
        value = {'event': kind, 'pid': os.getpid(), 'job': index, 'device': device,
                 'time': time.time(), **details}
        with trace_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')

    old_identity = P._ENCODER_CHECKPOINT
    previous_device = old_identity[1] if old_identity is not None else None
    event('job_start', previous_device=previous_device, rows=len(raw))
    original_load, original_forward = D.load_checkpoint, D._encode_prepared_chunk

    def load(path, target='cpu'):
        event('load_start', target_device=str(target))
        tick = time.perf_counter()
        try:
            result = original_load(path, target)
            torch.cuda.synchronize(target)
            require(result[2]['sha256'] == expected['sha256'], 'Checkpoint digest changed')
            require(result[2]['config_sha256'] == expected['config_sha256'], 'Checkpoint config changed')
        except Exception as exc:
            event('load_error', seconds=time.perf_counter() - tick,
                  error=f'{type(exc).__name__}: {exc}')
            raise RuntimeError('Encoder load failed; diagnostic stops instead of OOM retry') from exc
        event('load_done', seconds=time.perf_counter() - tick)
        return result

    def forward(*args, **kwargs):
        try:
            return original_forward(*args, **kwargs)
        except Exception as exc:
            raise StopDiagnostic(f'Encoder forward failed; no batch retry: {exc}') from exc

    D.load_checkpoint, D._encode_prepared_chunk = load, forward
    tick = time.perf_counter()
    try:
        result = P._encoding_job(job)
        torch.cuda.synchronize(device)
        encoded = result[1]
        require(not encoded.failed, f'Encoding failures: {encoded.failed[:3]}')
        require(encoded.ids == [r['id'] for r in raw], 'Encoded ID order mismatch')
        require(encoded.embeddings.shape == (len(raw), expected['embed_dim']), 'Feature shape mismatch')
        require(np.isfinite(encoded.embeddings).all(), 'Non-finite features')
        require(result[2] == 1, 'Prepared cache miss: diagnostic setup changed')
        event('job_done', seconds=time.perf_counter() - tick, prepared_cache_hit=result[2])
        return result
    except BaseException as exc:
        event('job_error', seconds=time.perf_counter() - tick, error=f'{type(exc).__name__}: {exc}')
        raise RuntimeError(f'Worker {os.getpid()} job {index} failed: {exc}') from exc
    finally:
        D.load_checkpoint, D._encode_prepared_chunk = original_load, original_forward


def traced_encoding_job(payload):
    job, root, trace_dir, expected = payload
    job = worker_job(job)
    configure(Path(root))
    import numpy as np
    import torch
    from muse_encoder.pretrain import dashboard as D
    from muse_encoder.pretrain import parallel_dashboard as P
    return trace_job(job, D, P, torch, np, trace_dir, expected)


def bind_worker(device_queue):
    global _FIXED_DEVICE
    _FIXED_DEVICE = device_queue.get(timeout=10)


def worker_job(job):
    if _FIXED_DEVICE is None:
        return job
    return (*job[:3], _FIXED_DEVICE, *job[4:])


def fixed_parallel_map(jobs, function=traced_encoding_job):
    context = mp.get_context('spawn')
    devices = context.Queue()
    for device in ('cuda:0', 'cuda:1'):
        devices.put(device)
    results = []
    try:
        with context.Pool(2, initializer=bind_worker, initargs=(devices,)) as pool:
            for result in pool.imap_unordered(function, jobs, chunksize=1):
                results.append(result)
                print(f'[fixed-eval] completed={len(results)}/{len(jobs)}', flush=True)
    finally:
        devices.close()
        devices.join_thread()
    return results


def load_reference(np, directory, current, embed_dim):
    reference = read_json(directory / 'report.json')
    require(reference.get('status') == 'completed', 'Reference run did not complete')
    require(reference.get('mode', 'original') == 'original', 'Reference must use original dispatcher')
    for key in ('ids', 'checkpoint_sha256', 'input_sha256', 'source_files', 'torch_version', 'hip_version'):
        require(reference[key] == current[key], f'Reference differs: {key}')
    features = np.load(directory / 'features.npy', allow_pickle=False)
    require(features.shape == (len(current['ids']), embed_dim), 'Reference feature shape mismatch')
    require(np.isfinite(features).all(), 'Reference features contain NaN/Inf')
    return reference, features


def compare_features(np, original, fixed):
    require(original.shape == fixed.shape, 'Comparison feature shape mismatch')
    require(np.isfinite(original).all() and np.isfinite(fixed).all(), 'Non-finite comparison features')
    return {'features_allclose': bool(np.allclose(original, fixed, rtol=1e-4, atol=1e-5)),
            'features_exact_equal': bool(np.array_equal(original, fixed)),
            'max_abs_difference': float(np.max(np.abs(original - fixed))),
            'rtol': 1e-4, 'atol': 1e-5}


def summarize(trace_dir):
    workers = []
    for path in sorted(Path(trace_dir).glob('worker_*.jsonl')):
        events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        starts = [e for e in events if e['event'] == 'job_start']
        loads = [e for e in events if e['event'] == 'load_done']
        workers.append({
            'pid': events[0]['pid'],
            'devices_used': sorted({e['device'] for e in starts}),
            'job_device_sequence': [f"{e['job']}:{e['device']}" for e in starts],
            'device_changes': sum(e['previous_device'] is not None and
                                  e['previous_device'] != e['device'] for e in starts),
            'load_attempts': sum(e['event'] == 'load_start' for e in events),
            'successful_model_loads': len(loads),
            'model_load_seconds_sum': sum(e['seconds'] for e in loads),
            'completed_jobs': sum(e['event'] == 'job_done' for e in events),
            'errors': [e for e in events if e['event'] in ('load_error', 'job_error')],
        })
    return {'workers': workers,
            'successful_model_loads_total': sum(w['successful_model_loads'] for w in workers),
            'device_changes_total': sum(w['device_changes'] for w in workers),
            'completed_jobs_total': sum(w['completed_jobs'] for w in workers)}


def read_source(args, D, P, MuseConfig, samples):
    """Read and validate the same real source for small and cluster diagnostics."""
    state = read_json(args.run / 'run_state.json')
    checkpoint = Path(state['checkpoint']['path']).resolve()
    require(checkpoint.is_file(), f'Missing checkpoint: {checkpoint}')
    cfg = MuseConfig(**{f.name: state['model_config'][f.name] for f in fields(MuseConfig)
                        if f.name in state['model_config']})
    require(D._config_sha(cfg.to_dict()) == state['checkpoint']['config_sha256'], 'Source config mismatch')
    board = D.load_board()
    subset = board.subsets[('PEER', 'fluorescence')]
    meta = read_json(D._subset_cache(args.run / 'embeddings', subset) / 'train.encoding.json')
    raw = D._read_jsonl(D._split_file(D.DATA_ROOT, subset, 'train'), samples)
    require(meta['failed'] == [] and meta['checkpoint_sha256'] == state['checkpoint']['sha256'],
            'Source failure/identity mismatch')
    require(len(raw) == samples and [r['id'] for r in raw] == meta['ids'][:samples], 'Source IDs mismatch')
    require(len({r['id'] for r in raw}) == samples, 'Duplicate source IDs')
    require(not any(r.get('external_inputs') for r in raw), 'Unexpected external inputs')
    return state, checkpoint, cfg, subset, raw


def prepare_jobs(args, D, P, cfg, subset, raw, checkpoint, expected, trace_dir, devices=2):
    """CPU-only preparation; never query or write production feature caches."""
    external_base = D._external_base(D.DATA_ROOT, subset)
    version = D.sha256_file(D.DATA_ROOT / 'manifest.json')
    featurizer = D.RecordFeaturizer(cfg, D.load_tokenizers())
    jobs = []
    for index, start in enumerate(range(0, len(raw), 4)):
        batch = raw[start:start + 4]
        normalized, failed = P._normalize_batch(batch, external_base, args.output / 'external_cache')
        require(not failed and [r['id'] for r in normalized] == [r['id'] for r in batch],
                f'Input normalization failed: {failed[:3]}')
        items = []
        for row in normalized:
            prepared = D._prepare_featurized(featurizer, row)
            require(not P._can_window(prepared, cfg), 'Long-window sample outside current diagnostic')
            items.append({'record': prepared, 'error': None})
        key = P._prepared_key(normalized, P._normalization_key(normalized, external_base, version), cfg)
        path = P._cache_path(args.output / 'diagnostic_cache', 'prepared', 'fluorescence', key)
        P._write_cache(path, key, [r['id'] for r in normalized], items)
        job = (index, normalized, str(checkpoint), f'cuda:{index % devices}', path, key, 4, False)
        jobs.append((job, str(args.root), str(trace_dir), expected))
    return jobs


def run(args, report):
    configure(args.root)
    import numpy as np
    import torch
    from muse_encoder.config import MuseConfig
    from muse_encoder.pretrain import dashboard as D
    from muse_encoder.pretrain import parallel_dashboard as P
    require(torch.cuda.is_available() and torch.cuda.device_count() == 2,
            'Exactly two visible GPUs required; use HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1')
    state, checkpoint, cfg, subset, raw = read_source(args, D, P, MuseConfig, 32)
    trace_dir = args.output / 'trace'
    trace_dir.mkdir()
    report.update(scope='32 fluorescence records, batch=4, 2 spawn workers; no main GPU model', mode=args.mode,
                  ids=[r['id'] for r in raw], checkpoint=str(checkpoint),
                  checkpoint_sha256=state['checkpoint']['sha256'], input_sha256=P._digest(raw),
                  torch_version=torch.__version__, hip_version=torch.version.hip,
                  source_files={m.__file__: D.sha256_file(Path(m.__file__)) for m in (D, P)},
                  notes=['All prepared inputs created on CPU before timing the pool; no final feature cache.',
                         'Both GPUs remain visible; original mode assigns per job, fixed mode assigns once per worker.',
                         'Load time sums overlap across workers; do not sum them as wall-clock time.',
                         'Small scheduling observation, not full 90-task throughput or proof of OOM cause.',
                         'Diagnostic errors stop; production OOM wait/retry and batch reduction are disabled.'])
    reference = reference_features = None
    if args.mode == 'fixed':
        reference, reference_features = load_reference(np, args.reference, report, cfg.embed_dim)
        report['reference_directory'] = str(args.reference)
    write_json(args.output / 'report.json', report)
    print('Preparing 32 inputs on CPU in a separate diagnostic cache...', flush=True)
    tick = time.perf_counter()
    expected = {**state['checkpoint'], 'embed_dim': cfg.embed_dim}
    jobs = prepare_jobs(args, D, P, cfg, subset, raw, checkpoint, expected, trace_dir)
    report['input_preparation_seconds'] = time.perf_counter() - tick
    write_json(args.output / 'report.json', report)
    print(f'Starting {args.mode} dispatcher: 2 processes, GPUs 0/1, 8 jobs. Models load inside workers.', flush=True)
    tick = time.perf_counter()
    parts = (fixed_parallel_map(jobs) if args.mode == 'fixed'
             else P._parallel_map(traced_encoding_job, jobs, 2))
    report['pool_wall_seconds'] = time.perf_counter() - tick
    encoded, hits = P._merge_encoded(parts, cfg.embed_dim, [])
    require(encoded.ids == [r['id'] for r in raw] and not encoded.failed and hits == 8,
            'Incomplete encoding results')
    trace_summary = summarize(trace_dir)
    require(trace_summary['completed_jobs_total'] == 8, 'Incomplete worker trace')
    require(not any(w['errors'] for w in trace_summary['workers']), 'Worker errors in trace')
    report.update(trace_summary)
    np.save(args.output / 'features.npy', encoded.embeddings)
    if args.mode == 'fixed':
        require(len(trace_summary['workers']) == 2 and
                sorted(w['devices_used'] for w in trace_summary['workers']) == [['cuda:0'], ['cuda:1']],
                'Expected two active workers, each fixed to a different GPU')
        require(trace_summary['device_changes_total'] == 0 and
                all(w['successful_model_loads'] == 1 for w in trace_summary['workers']),
                'Fixed workers unexpectedly switched device or reloaded model')
        report['comparison'] = {
            'original_model_loads': reference['successful_model_loads_total'],
            'fixed_model_loads': trace_summary['successful_model_loads_total'],
            'original_pool_seconds': reference['pool_wall_seconds'],
            'fixed_pool_seconds': report['pool_wall_seconds'],
            **compare_features(np, reference_features, encoded.embeddings),
        }
        require(report['comparison']['features_allclose'], 'Fixed-mode features differ beyond tolerance')
        print(json.dumps(report['comparison'], indent=2), flush=True)
    report['status'] = 'completed'
    write_json(args.output / 'report.json', report)
    print(json.dumps({'pool_wall_seconds': report['pool_wall_seconds'], **trace_summary}, indent=2), flush=True)
    print(f'DONE: {args.output / "report.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('original', 'fixed'), default='original')
    parser.add_argument('--reference', type=Path, help='Completed original-mode output directory; required for fixed mode')
    args = parser.parse_args()
    if args.mode == 'fixed' and args.reference is None:
        parser.error('--mode fixed requires --reference')
    if args.mode == 'original' and args.reference is not None:
        parser.error('--reference is only used with --mode fixed')
    if args.reference is not None:
        args.reference = args.reference.resolve()
    args.root = args.root.resolve()
    args.run = (args.run or args.root / 'model_design/runs/s3-residual-pilot-02/official_a_peer_16').resolve()
    args.output = args.output.resolve()
    require(args.output != args.run and args.run not in args.output.parents, 'Output must be outside source run')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running'}
    write_json(args.output / 'report.json', report)

    def terminate(signum, frame):
        raise StopDiagnostic('SIGTERM: diagnostic did not complete')

    signal.signal(signal.SIGTERM, terminate)
    try:
        run(args, report)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if (args.output / 'trace').exists():
            report['partial_trace'] = summarize(args.output / 'trace')
        write_json(args.output / 'report.json', report)
        raise


if __name__ == '__main__':
    main()
