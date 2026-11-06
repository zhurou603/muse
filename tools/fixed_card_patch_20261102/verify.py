"""Compare production and candidate with forced device-switch requests."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import signal
import sys
import time

import bench_encoder_workers as B
import prepare

HERE = Path(__file__).resolve().parent


def candidate_module():
    location = str(HERE / 'candidate')
    if location not in sys.path:
        sys.path.insert(0, location)
    return importlib.import_module('fixed_parallel_dashboard')


def traced_job(payload):
    mode, (job, root, trace_dir, expected) = payload
    B.configure(Path(root))
    import numpy as np
    import torch
    from muse_encoder.pretrain import dashboard as D
    from muse_encoder.pretrain import parallel_dashboard as original

    module = original if mode == 'original' else candidate_module()

    previous = getattr(traced_job, '_requested_device', None)
    requested = job[3] if previous is None else (
        'cuda:1' if previous == 'cuda:0' else 'cuda:0'
    )
    traced_job._requested_device = requested
    job = (*job[:3], requested, *job[4:])

    if mode == 'candidate':
        B.require(
            module._ENCODING_DEVICE is not None,
            'Candidate worker was not bound'
        )
        job = (*job[:3], module._ENCODING_DEVICE, *job[4:])

    return B.trace_job(job, D, module, torch, np, trace_dir, expected)


def run(args, report):
    source = args.root / 'model_design/muse_encoder/pretrain/parallel_dashboard.py'
    data = source.read_bytes()
    B.require(
        hashlib.sha256(data).hexdigest() == prepare.EXPECTED,
        'Production source changed'
    )

    expected_candidate = prepare.patch_source(
        data.decode('utf-8').replace('\r\n', '\n')
    )
    candidate_path = HERE / 'candidate/fixed_parallel_dashboard.py'
    B.require(
        candidate_path.read_text(encoding='utf-8') == expected_candidate,
        'Candidate differs from patch'
    )

    B.configure(args.root)
    import numpy as np
    import torch
    from muse_encoder.config import MuseConfig
    from muse_encoder.pretrain import dashboard as D
    from muse_encoder.pretrain import parallel_dashboard as original

    candidate = candidate_module()
    B.require(
        D.sha256_file(Path(D.__file__)) ==
        '28c81c1631ad69ed48d6266f959bf395d8371791f0fc4a68b95e32664ab3838b',
        'dashboard source changed'
    )
    B.require(
        torch.cuda.is_available() and torch.cuda.device_count() == 2,
        'Exactly two visible GPUs required'
    )

    state, checkpoint, cfg, subset, raw = B.read_source(
        args, D, original, MuseConfig, 32
    )
    expected = {**state['checkpoint'], 'embed_dim': cfg.embed_dim}

    report.update(
        scope='Forced alternating device requests per worker; 32 records; not a throughput benchmark',
        checkpoint=str(checkpoint),
        checkpoint_sha256=state['checkpoint']['sha256'],
        ids=[r['id'] for r in raw],
        input_sha256=original._digest(raw),
        production_sha256=prepare.EXPECTED,
        candidate_sha256=D.sha256_file(candidate_path),
        torch_version=torch.__version__,
        hip_version=torch.version.hip,
        notes=[
            'Prepared inputs shared; final feature cache bypassed.',
            'Original then candidate; OS caching/order effects not controlled.',
            'Diagnostic stops on encoding failure; production OOM retries disabled.',
            'No full 90-task benchmark; production files unchanged.'
        ]
    )

    print(
        'Preparing 32 records on CPU; production files and caches unchanged.',
        flush=True
    )
    jobs = B.prepare_jobs(
        args, D, original, cfg, subset, raw,
        checkpoint, expected, args.output / 'unused'
    )

    results = {}
    for mode, module in [('original', original), ('candidate', candidate)]:
        trace = args.output / mode / 'trace'
        trace.mkdir(parents=True)
        payloads = [
            (mode, (job, root, str(trace), identity))
            for job, root, _, identity in jobs
        ]
        B.write_json(args.output / 'report.json', report)

        print(f'RUN {mode}: 32 records, 8 batches, 2 processes', flush=True)
        started = time.perf_counter()
        kwargs = (
            {'devices': ['cuda:0', 'cuda:1']}
            if mode == 'candidate' else {}
        )
        parts = module._parallel_map(traced_job, payloads, 2, **kwargs)
        elapsed = time.perf_counter() - started

        encoded, hits = module._merge_encoded(parts, cfg.embed_dim, [])
        B.require(
            encoded.ids == report['ids'] and not encoded.failed and hits == 8,
            'Incomplete results'
        )

        summary = B.summarize(trace)
        B.require(summary['completed_jobs_total'] == 8, 'Incomplete trace')
        B.require(
            not any(w['errors'] for w in summary['workers']),
            'Worker errors'
        )

        if mode == 'original':
            B.require(
                summary['device_changes_total'] > 0 and
                summary['successful_model_loads_total'] == 8,
                'Original did not reproduce forced device switching and reloading'
            )

        if mode == 'candidate':
            B.require(
                len(summary['workers']) == 2 and
                summary['device_changes_total'] == 0 and
                summary['successful_model_loads_total'] == 2 and
                sorted(w['devices_used'] for w in summary['workers']) ==
                [['cuda:0'], ['cuda:1']],
                'Candidate did not keep one model per device'
            )

        report[mode] = {'pool_wall_seconds': elapsed, **summary}
        results[mode] = encoded.embeddings
        np.save(args.output / mode / 'features.npy', encoded.embeddings)
        B.write_json(args.output / 'report.json', report)

    comparison = B.compare_features(
        np, results['original'], results['candidate']
    )
    report['comparison'] = comparison
    B.require(
        comparison['features_allclose'],
        'Candidate features differ beyond tolerance'
    )
    B.require(
        hashlib.sha256(source.read_bytes()).hexdigest() == prepare.EXPECTED,
        'Production source changed during test'
    )

    report['status'] = 'completed'
    B.write_json(args.output / 'report.json', report)
    print(
        json.dumps(
            {mode: report[mode] for mode in ('original', 'candidate', 'comparison')},
            indent=2
        ),
        flush=True
    )
    print(f'DONE: {args.output / "report.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=B.ROOT)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    args.root = args.root.resolve()
    args.run = (
        args.run or
        args.root / 'model_design/runs/s3-residual-pilot-02/official_a_peer_16'
    ).resolve()
    args.output = args.output.resolve()

    B.require(
        args.root / 'tools' in args.output.parents,
        'Diagnostic output must be inside root/tools'
    )
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running'}

    def stop(signum, frame):
        raise B.StopDiagnostic('Timed out/interrupted; verification incomplete')

    signal.signal(signal.SIGTERM, stop)
    B.write_json(args.output / 'report.json', report)

    try:
        run(args, report)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        report['partial_traces'] = {}

        for mode in ('original', 'candidate'):
            trace = args.output / mode / 'trace'
            if trace.is_dir():
                try:
                    report['partial_traces'][mode] = B.summarize(trace)
                except Exception as trace_error:
                    report['partial_traces'][mode] = {
                        'read_error': str(trace_error)
                    }

        B.write_json(args.output / 'report.json', report)
        raise


if __name__ == '__main__':
    main()