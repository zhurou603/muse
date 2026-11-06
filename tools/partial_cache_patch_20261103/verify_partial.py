"""One-GPU feature reuse verification with a temporary normalization failure."""
import argparse
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import signal
import sys
from unittest.mock import patch

import prepare_partial as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'long_no_grad_patch_20261102'))
import verify_long as V
B = V.B


def run(args, report):
    B.configure(args.root)
    import numpy as np
    import torch
    from muse_encoder.config import MuseConfig
    from muse_encoder.pretrain import dashboard as D
    from muse_encoder.pretrain import parallel_dashboard as original
    source = Path(original.__file__)
    B.require(D.sha256_file(source) == R.EXPECTED, 'Production version changed')
    B.require(D.sha256_file(Path(D.__file__)) ==
              '28c81c1631ad69ed48d6266f959bf395d8371791f0fc4a68b95e32664ab3838b', 'dashboard changed')
    path = HERE / 'candidate/partial_parallel_dashboard.py'
    B.require(path.read_text(encoding='utf-8') == R.patch_source(source.read_text(encoding='utf-8')),
              'Candidate differs from expected patch')
    sys.path.insert(0, str(path.parent))
    candidate = importlib.import_module('partial_parallel_dashboard')
    B.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One visible GPU required')
    state, checkpoint, cfg, subset, all_rows = B.read_source(args, D, original, MuseConfig, 32)
    raw = all_rows[:4]
    bad_id = raw[1]['id']
    success_ids = [r['id'] for r in raw if r['id'] != bad_id]
    error = 'DIAGNOSTIC injected normalization failure'
    fault = {'enabled': True}
    counters = {'forward_calls': 0, 'forward_rows': 0, 'model_loads': 0}
    models = []
    normalize, forward, load = D.normalize_record, D._encode_prepared_chunk, D.load_checkpoint

    def normalize_with_failure(row, resolver):
        if fault['enabled'] and row['id'] == bad_id:
            raise ValueError(error)
        return normalize(row, resolver)

    def traced_forward(encoder, cfg, records, device):
        counters['forward_calls'] += 1
        counters['forward_rows'] += len(records)
        try:
            return forward(encoder, cfg, records, device)
        except Exception as exc:
            raise B.StopDiagnostic(f'Unexpected real forward failure: {exc}') from exc

    def traced_load(*a, **kw):
        result = load(*a, **kw)
        V.check_checkpoint_identity(D, result[2], state['checkpoint'], result[1], cfg)
        counters['model_loads'] += 1
        models.append((result[0], {n: (p._version, p.requires_grad) for n, p in result[0].named_parameters()}))
        return result

    common_args = dict(records=raw, checkpoint_path=checkpoint, checkpoint_sha=state['checkpoint']['sha256'],
                       cfg=cfg, external_base=D._external_base(D.DATA_ROOT, subset),
                       external_cache=args.output / 'external_cache', namespace='fluorescence/train',
                       data_version=D.sha256_file(D.DATA_ROOT / 'manifest.json'), devices=['cuda:0'],
                       cpu_workers=1, batch_size=1, probe_config=D.PROBE_CONFIG)
    report.update(production_sha256=R.EXPECTED, candidate_sha256=D.sha256_file(path),
                  checkpoint_sha256=state['checkpoint']['sha256'], rows=4, failed_id_injected=bad_id,
                  scope='4 real rows; isolated caches; single process/device; no training', cases=[])

    def compare_by_id(expected, actual):
        B.require(set(expected.ids) == set(actual.ids) and len(actual.ids) == len(set(actual.ids)), 'ID mismatch')
        order = [actual.ids.index(i) for i in expected.ids]
        result = {'embeddings': B.compare_features(np, expected.embeddings, actual.embeddings[order])}
        if expected.tokens is not None:
            B.require(actual.tokens is not None, 'Token features missing')
            pieces = [actual.tokens[actual.offsets[i]:actual.offsets[i + 1]] for i in order]
            B.require([len(p) for p in pieces] == np.diff(expected.offsets).tolist(), 'Token counts changed')
            result['tokens'] = B.compare_features(np, expected.tokens, np.concatenate(pieces))
        B.require(all(value['features_allclose'] for value in result.values()), 'Features changed')
        return result

    with patch.object(D, 'normalize_record', normalize_with_failure), \
            patch.object(D, '_encode_prepared_chunk', traced_forward), patch.object(D, 'load_checkpoint', traced_load):
        for token_level in (False, True):
            case = {'token_level': token_level, 'stages': {}}
            report['cases'].append(case)
            roots = {name: args.output / f'{name}_tokens_{token_level}' for name in ('original', 'candidate')}
            fault['enabled'] = True

            def measure(module, name, cache_name, expected_rows, expect_failed, **overrides):
                before = dict(counters)
                result, stats = module.encode_records_parallel(**{
                    **common_args, 'cache_root': roots[cache_name], 'token_level': token_level, **overrides})
                torch.cuda.synchronize()
                delta = {k: counters[k] - before[k] for k in counters}
                B.require(delta['forward_rows'] == expected_rows, f'{name}: unexpected computation {delta}')
                expected_failed = [{'id': bad_id, 'error': error}] if expect_failed else []
                B.require(result.failed == expected_failed, f'{name}: failure status differs: {result.failed}')
                candidate._validate_encoded_cache(result, [r['id'] for r in raw], cfg.embed_dim, token_level)
                if expect_failed:
                    B.require(set(result.ids) == set(success_ids) and not stats['central_hit'], 'Partial reported as full')
                case['stages'][name] = {**stats, **delta, 'successful_rows': len(result.ids), 'failed': result.failed}
                B.write_json(args.output / 'report.json', report)
                return result, stats

            print(f'tokens={token_level}: original repeats 3 successes; candidate should reuse them.', flush=True)
            old_first, _ = measure(original, 'original_first', 'original', 3, True)
            old_second, _ = measure(original, 'original_repeat', 'original', 3, True)
            new_first, _ = measure(candidate, 'candidate_first', 'candidate', 3, True)
            new_second, stats = measure(candidate, 'candidate_repeat', 'candidate', 0, True)
            B.require(stats['partial_hit'] and stats['reused_embedding_rows'] == 3 and stats['retried_records'] == 1,
                      'Partial reuse counters differ')
            case['partial_comparison'] = compare_by_id(old_first, new_second)
            case['reuse_comparison'] = compare_by_id(new_first, new_second)
            fault['enabled'] = False
            print('Remove injected failure: only the remaining row should compute.', flush=True)
            recovered, stats = measure(candidate, 'candidate_recovered', 'candidate', 1, False)
            B.require(stats['reused_embedding_rows'] == 3, 'Recovered cache did not reuse successes')
            B.require(recovered.ids == [r['id'] for r in raw], 'Recovered rows are out of input order')
            complete, stats = measure(candidate, 'candidate_complete_hit', 'candidate', 0, False,
                                      probe_config=replace(D.PROBE_CONFIG, max_epochs=20))
            B.require(stats['central_hit'], 'Complete result not cached')
            reference, _ = measure(original, 'original_clean_reference', 'original', 4, False)
            case['recovery_comparison'] = compare_by_id(reference, recovered)
            case['complete_comparison'] = compare_by_id(recovered, complete)
            migrated, stats = measure(candidate, 'old_complete_cache_hit', 'original', 0, False)
            B.require(stats['central_hit'], 'Existing complete cache incompatible')
            case['old_cache_comparison'] = compare_by_id(reference, migrated)
    for model, versions in models:
        B.require(versions == {n: (p._version, p.requires_grad) for n, p in model.named_parameters()} and
                  all(p.grad is None for p in model.parameters()), 'Model parameters/gradients changed')
    B.require(D.sha256_file(source) == R.EXPECTED, 'Production changed')
    report.update(status='completed', production_unchanged=True, parameter_versions_unchanged=True,
                  total_counters=counters, notes=['Failure was temporarily injected; input records unchanged.',
                                                 'Recovered results preserve input ID order and token alignment.',
                                                 'This measures avoided work, not full 90-task wall-time improvement.'])
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
        raise B.StopDiagnostic('Interrupted; verification incomplete')

    signal.signal(signal.SIGTERM, stop)
    try:
        run(args, report)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        B.write_json(args.output / 'report.json', report)
        raise


if __name__ == '__main__':
    main()
