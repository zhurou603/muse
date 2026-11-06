"""Single-GPU real-feature cache verification; never train a prediction head."""
import argparse
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import signal
import sys
from unittest.mock import patch

import prepare_cache as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'fixed_card_patch_20261102'))
import bench_encoder_workers as B


class CacheMiss(RuntimeError):
    pass


def no_dispatch(*args, **kwargs):
    raise CacheMiss('Cache miss reached computation dispatch')


def run(args, report):
    B.configure(args.root)
    import numpy as np
    import torch
    from muse_encoder.config import MuseConfig
    from muse_encoder.pretrain import dashboard as D
    from muse_encoder.pretrain import parallel_dashboard as original

    source = Path(original.__file__)
    raw_source = source.read_bytes()
    B.require(hashlib.sha256(raw_source).hexdigest() == R.EXPECTED, 'Production fixed-card version changed')
    B.require(D.sha256_file(Path(D.__file__)) ==
              '28c81c1631ad69ed48d6266f959bf395d8371791f0fc4a68b95e32664ab3838b',
              'dashboard changed')
    candidate_path = HERE / 'candidate/cache_parallel_dashboard.py'
    B.require(candidate_path.read_text(encoding='utf-8') ==
              R.patch_source(raw_source.decode('utf-8').replace('\r\n', '\n')), 'Candidate changed')
    sys.path.insert(0, str(candidate_path.parent))
    candidate = importlib.import_module('cache_parallel_dashboard')
    B.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one visible GPU required')
    state, checkpoint, cfg, subset, raw = B.read_source(args, D, original, MuseConfig, 32)
    kwargs = dict(records=raw, checkpoint_path=checkpoint, checkpoint_sha=state['checkpoint']['sha256'],
                  cfg=cfg, cache_root=args.output / 'cache', external_base=D._external_base(D.DATA_ROOT, subset),
                  external_cache=args.output / 'external_cache', namespace='fluorescence/train',
                  data_version=D.sha256_file(D.DATA_ROOT / 'manifest.json'), devices=['cuda:0'],
                  cpu_workers=1, batch_size=4, token_level=False, probe_config=D.PROBE_CONFIG)
    report.update(rows=len(raw), checkpoint=str(checkpoint), checkpoint_sha256=kwargs['checkpoint_sha'],
                  production_sha256=R.EXPECTED, candidate_sha256=D.sha256_file(candidate_path),
                  scope='32 records; isolated cache; no head training or production change')
    counters = {'model_loads': 0, 'forward_calls': 0}
    load, forward = D.load_checkpoint, D._encode_prepared_chunk

    def counted_load(*a, **kw):
        counters['model_loads'] += 1
        result = load(*a, **kw)
        B.require(result[2]['sha256'] == state['checkpoint']['sha256'] and
                  result[2]['config_sha256'] == state['checkpoint']['config_sha256'], 'Checkpoint identity changed')
        return result

    def counted_forward(*a, **kw):
        counters['forward_calls'] += 1
        try:
            return forward(*a, **kw)
        except Exception as exc:
            raise B.StopDiagnostic(f'Forward failed; no silent batch retry: {exc}') from exc

    print('STEP 1: original computes 32 real features into a NEW diagnostic cache.', flush=True)
    with patch.object(D, 'load_checkpoint', counted_load), patch.object(D, '_encode_prepared_chunk', counted_forward):
        baseline, stats = original.encode_records_parallel(**kwargs)
        torch.cuda.synchronize()
        B.require(not stats['central_hit'] and not baseline.failed and baseline.ids == [r['id'] for r in raw],
                  'Original cache creation incomplete')
        B.require(counters['model_loads'] == 1 and counters['forward_calls'] == 8, 'Unexpected baseline computation')
        report['original'] = {**stats, **counters}
        B.write_json(args.output / 'report.json', report)
        old_files = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (args.output / 'cache/embeddings').rglob('*.pkl')}
        B.require(len(old_files) == 1, 'Expected one original feature cache')
        with patch.object(candidate, '_parallel_map', no_dispatch):
            print('STEP 2: candidate must reuse the matching old cache without computation.', flush=True)
            migrated, migration_stats = candidate.encode_records_parallel(**kwargs)
            B.require(migration_stats['central_hit'], 'Old cache was not reused')
            report['migration'] = {**migration_stats, **B.compare_features(np, baseline.embeddings, migrated.embeddings)}
            changed_probe = replace(D.PROBE_CONFIG, max_epochs=20)
            print('STEP 3: only a TEST head setting changes; no MLP is trained.', flush=True)
            changed, head_stats = candidate.encode_records_parallel(**{**kwargs, 'probe_config': changed_probe})
            B.require(head_stats['central_hit'], 'Head-only change invalidated feature cache')
            report['head_config_change'] = {**head_stats, **B.compare_features(np, baseline.embeddings, changed.embeddings)}
            changed_records = [{**raw[0], 'id': raw[0]['id'] + '__diagnostic_changed'}] + raw[1:]
            variants = {'checkpoint': {'checkpoint_sha': 'diagnostic-other-checkpoint'},
                        'records': {'records': changed_records}, 'data_version': {'data_version': 'diagnostic-other-data'},
                        'batch_size': {'batch_size': 2}, 'token_output': {'token_level': True}}
            report['must_invalidate'] = {}
            for name, changes in variants.items():
                try:
                    candidate.encode_records_parallel(**{**kwargs, **changes})
                except CacheMiss:
                    report['must_invalidate'][name] = True
                else:
                    raise RuntimeError(f'{name} incorrectly reused old features')
        B.require(counters == {'model_loads': 1, 'forward_calls': 8}, 'Cache checks unexpectedly computed features')
    for path, digest in old_files.items():
        B.require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'Original cache modified')
    B.require(len(list((args.output / 'cache/embeddings').rglob('*.pkl'))) == 2, 'New key was not saved')
    for name in ('migration', 'head_config_change'):
        B.require(report[name]['features_allclose'], 'Cached output differs')
    B.require(hashlib.sha256(source.read_bytes()).hexdigest() == R.EXPECTED, 'Production changed during test')
    report.update(status='completed', total_model_loads=counters['model_loads'],
                  total_forward_calls=counters['forward_calls'], production_unchanged=True)
    B.write_json(args.output / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)
    print(f'DONE: {args.output / "report.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=B.ROOT)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.run = (args.run or args.root / 'model_design/runs/s3-residual-pilot-02/official_a_peer_16').resolve()
    args.output = args.output.resolve()
    B.require(args.root / 'tools' in args.output.parents, 'Output must be inside root/tools')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running'}

    def stop(signum, frame):
        raise B.StopDiagnostic('Verification interrupted; incomplete')

    signal.signal(signal.SIGTERM, stop)
    try:
        run(args, report)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        B.write_json(args.output / 'report.json', report)
        raise


if __name__ == '__main__':
    main()
