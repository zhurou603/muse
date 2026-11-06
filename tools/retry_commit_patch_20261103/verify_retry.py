"""Verify staged results with real encoding and a deliberately injected compute error."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import signal
import sys
from unittest.mock import patch

import check_retry as C
import prepare_retry as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'long_no_grad_patch_20261102'))
import verify_long as V
B = V.B


def run(args, report):
    B.configure(args.root)
    import numpy as np
    import torch
    from muse_encoder.config import MuseConfig
    from muse_encoder.featurize import RecordFeaturizer
    from muse_encoder import inference as I
    from muse_encoder.pretrain import dashboard as D
    from muse_encoder.pretrain import parallel_dashboard as original

    source = Path(original.__file__)
    text = source.read_text(encoding='utf-8')
    B.require(D.sha256_file(source) == R.EXPECTED, 'Production version differs')
    B.require(D.sha256_file(Path(D.__file__)) ==
              '28c81c1631ad69ed48d6266f959bf395d8371791f0fc4a68b95e32664ab3838b', 'dashboard changed')
    path = HERE / 'candidate/retry_parallel_dashboard.py'
    candidate_text = path.read_text(encoding='utf-8')
    B.require(candidate_text == R.patch_source(text), 'Candidate differs from minimal patch')
    report['cpu_cases'] = C.run_checks(candidate_text)
    duplicate = C.encode(C.load_function(text), [{'long': True}, {'bad': True}])
    B.require(duplicate.ids == ['0', '0'], 'Original duplicate not reproduced in controlled case')
    sys.path.insert(0, str(path.parent))
    candidate = importlib.import_module('retry_parallel_dashboard')
    B.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One visible GPU required')
    state, checkpoint, cfg, subset, short_rows = B.read_source(args, D, original, MuseConfig, 32)
    samples_file = HERE.parent / 'long_no_grad_patch_20261102/long_samples.json'
    samples = B.read_json(samples_file)
    B.require(D.sha256_file(Path(I.__file__)) == samples['inference_sha256'], 'Inference source changed')
    raw = [samples['samples'][0]['record'], short_rows[0]]
    B.require(len({r['id'] for r in raw}) == 2 and not any(r.get('external_inputs') for r in raw),
              'Unexpected input identity/external assets')
    raw, failures = original._normalize_batch(raw, args.root, args.output / 'external_cache')
    B.require(not failures and len(raw) == 2, f'Normalization failed: {failures}')
    fz = RecordFeaturizer(cfg, D.load_tokenizers())
    records = [D._prepare_featurized(fz, row) for row in raw]
    B.require(original._can_window(records[0], cfg) and not original._can_window(records[1], cfg),
              'Expected one long and one short record')
    items = [{'record': r, 'error': None} for r in records]
    print('Loading checkpoint once; 2 real records, 1 GPU; no training.', flush=True)
    encoder, loaded_cfg, identity = D.load_checkpoint(checkpoint, 'cuda:0')
    report['checkpoint_checks'] = V.check_checkpoint_identity(D, identity, state['checkpoint'], loaded_cfg, cfg)
    versions = {n: (p._version, p.requires_grad) for n, p in encoder.named_parameters()}
    report.update(production_sha256=R.EXPECTED, candidate_sha256=D.sha256_file(path),
                  checkpoint_sha256=identity['sha256'], record_ids=[r['id'] for r in raw],
                  scope='2 real records; temporary injected compute failure; no data or production cache edits',
                  cases=[])

    def compare(expected, actual):
        B.require(actual.ids == expected.ids and actual.failed == expected.failed, 'IDs/failures differ')
        result = {'embeddings': B.compare_features(np, expected.embeddings, actual.embeddings)}
        if expected.tokens is not None:
            B.require(actual.tokens is not None and np.array_equal(expected.offsets, actual.offsets),
                      'Token offsets differ')
            result['tokens'] = B.compare_features(np, expected.tokens, actual.tokens)
        else:
            B.require(actual.tokens is None and actual.offsets is None, 'Unexpected tokens')
        B.require(all(r['features_allclose'] for r in result.values()), 'Feature values differ')
        return result

    for token_level in (True, False):
        print(f'Checking normal output and retry with token_level={token_level}.', flush=True)
        normal_old = original._encode_items(encoder, cfg, raw, items, 2, token_level, 'cuda:0')
        normal_new = candidate._encode_items(encoder, cfg, raw, items, 2, token_level, 'cuda:0')
        B.require(normal_old.ids == [r['id'] for r in raw] and not normal_old.failed, 'Normal baseline failed')
        normal_comparison = compare(normal_old, normal_new)
        reference = original._encode_items(encoder, cfg, raw[:1], items[:1], 2, token_level, 'cuda:0')
        V.check_result(reference, raw[0], records[0].n_real_tokens, cfg.embed_dim, token_level)
        error = 'DIAGNOSTIC injected short-record compute failure'
        with patch.object(D, '_encode_prepared_chunk', side_effect=RuntimeError(error)):
            broken = original._encode_items(encoder, cfg, raw, items, 2, token_level, 'cuda:0')
            fixed = candidate._encode_items(encoder, cfg, raw, items, 2, token_level, 'cuda:0')
        expected_failed = [{'id': raw[1]['id'], 'error': error}]
        B.require(broken.ids == [raw[0]['id']] * 2 and broken.failed == expected_failed,
                  'Original duplicate was not reproduced with real model')
        B.require(fixed.ids == [raw[0]['id']] and fixed.failed == expected_failed,
                  'Retry lost/duplicated success or hid failure')
        # Keep the deliberate failure in the reference for exact result comparison.
        reference.failed = expected_failed
        retry_comparison = compare(reference, fixed)
        recovered = candidate._encode_items(encoder, cfg, raw, items, 2, token_level, 'cuda:0')
        recovery_comparison = compare(normal_old, recovered)
        report['cases'].append({'token_level': token_level, 'original_success_ids': broken.ids,
                                'candidate_success_ids': fixed.ids, 'candidate_failed': fixed.failed,
                                'normal_comparison': normal_comparison, 'retry_comparison': retry_comparison,
                                'after_removing_injection_comparison': recovery_comparison})
        B.write_json(args.output / 'report.json', report)
    B.require(versions == {n: (p._version, p.requires_grad) for n, p in encoder.named_parameters()} and
              all(p.grad is None for p in encoder.parameters()), 'Model parameters/gradients changed')
    B.require(D.sha256_file(source) == R.EXPECTED, 'Production source changed')
    report.update(status='completed', production_unchanged=True, parameter_versions_unchanged=True,
                  notes=['Failure was deliberately injected, not a newly found bad sample.',
                         'Only parallel _encode_items is patched; standalone dashboard.encode_records is unchanged.',
                         'Existing cached duplicate outputs are not repaired by this test.'])
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
