"""Reproduce a historical gradient failure, then compare corrected long/short inference."""
import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import signal
import sys
from unittest.mock import patch

import prepare_long as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'fixed_card_patch_20261102'))
import bench_encoder_workers as B


def check_result(result, raw, count, dim, token_level=True):
    import numpy as np
    B.require(not result.failed and result.ids == [raw['id']], f'Encoding failed: {result.failed}')
    B.require(result.embeddings.shape == (1, dim) and np.isfinite(result.embeddings).all(), 'Invalid embedding')
    if token_level:
        B.require(result.tokens is not None and result.tokens.shape[0] == count and
                  result.offsets.tolist() == [0, count] and np.isfinite(result.tokens).all(),
                  'Tokens missing, duplicated or non-finite')


def check_checkpoint_identity(D, identity, expected, loaded_cfg, cfg):
    loaded_hash = D._config_sha(loaded_cfg.to_dict())
    requested_hash = D._config_sha(cfg.to_dict())
    checks = {
        'checkpoint_sha256': identity['sha256'] == expected['sha256'],
        'checkpoint_config_sha256': identity['config_sha256'] == expected['config_sha256'],
        'loaded_config_sha256': loaded_hash == expected['config_sha256'],
        'requested_config_sha256': requested_hash == expected['config_sha256'],
    }
    B.require(all(checks.values()), 'Checkpoint identity differs: ' + json.dumps({
        'checks': checks, 'actual': identity, 'expected': expected,
        'loaded_config_sha256': loaded_hash, 'requested_config_sha256': requested_hash,
    }, ensure_ascii=False, default=str))
    return checks


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
    source_bytes = source.read_bytes()
    B.require(hashlib.sha256(source_bytes).hexdigest() == R.EXPECTED, 'Production version changed')
    B.require(D.sha256_file(Path(D.__file__)) ==
              '28c81c1631ad69ed48d6266f959bf395d8371791f0fc4a68b95e32664ab3838b', 'dashboard changed')
    candidate_path = HERE / 'candidate/long_parallel_dashboard.py'
    B.require(candidate_path.read_text(encoding='utf-8') ==
              R.patch_source(source_bytes.decode('utf-8').replace('\r\n', '\n')), 'Candidate differs')
    sys.path.insert(0, str(candidate_path.parent))
    candidate = importlib.import_module('long_parallel_dashboard')
    samples_path = HERE / 'long_samples.json'
    samples = B.read_json(samples_path)
    B.require(D.sha256_file(Path(I.__file__)) == samples['inference_sha256'], 'Inference source changed')
    B.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One visible GPU required')
    state, checkpoint, cfg, subset, short_rows = B.read_source(args, D, original, MuseConfig, 32)
    report.update(scope='3 historical long failures + 1 short control; no training or production cache',
                  production_sha256=R.EXPECTED, candidate_sha256=D.sha256_file(candidate_path),
                  sample_file_sha256=D.sha256_file(samples_path), checkpoint=str(checkpoint),
                  checkpoint_sha256=state['checkpoint']['sha256'], samples=[])
    raw_rows = [s['record'] for s in samples['samples']] + [short_rows[0]]
    B.require(len(raw_rows) == 4 and len({r['id'] for r in raw_rows}) == 4, 'Sample set incomplete')
    normalized, failures = original._normalize_batch(raw_rows, args.root, args.output / 'external_cache')
    B.require(not failures and [r['id'] for r in normalized] == [r['id'] for r in raw_rows],
              f'Normalization failed: {failures}')
    fz = RecordFeaturizer(cfg, D.load_tokenizers())
    prepared = [D._prepare_featurized(fz, row) for row in normalized]
    B.require(all(original._can_window(p, cfg) for p in prepared[:3]) and
              not original._can_window(prepared[3], cfg), 'Unexpected long/short routing')
    for sample, record in zip(samples['samples'], prepared[:3]):
        B.require(record.n_real_tokens == sample['sequence_length'] and
                  'requires grad' in sample['historical_error'], 'Historical sample identity differs')
    print('Loading checkpoint once; one GPU, window batch_size=1.', flush=True)
    encoder, loaded_cfg, identity = D.load_checkpoint(checkpoint, 'cuda:0')
    report['checkpoint_checks'] = check_checkpoint_identity(D, identity, state['checkpoint'], loaded_cfg, cfg)
    B.require(not encoder.training and any(p.requires_grad for p in encoder.parameters()),
              'Expected eval model with trainable parameter flags for reproduction')
    versions = {name: (p._version, p.requires_grad) for name, p in encoder.named_parameters()}
    print('STEP 1: reproduce original requires-grad error on shortest historical sample.', flush=True)
    with torch.enable_grad():
        old = original._encode_items(encoder, cfg, normalized[:1],
                                     [{'record': prepared[0], 'error': None}], 1, True, 'cuda:0')
    B.require(old.ids == [] and len(old.failed) == 1 and 'requires grad' in old.failed[0]['error'],
              f'Original did not reproduce expected error: {old.failed}')
    report['original_error'] = old.failed
    B.write_json(args.output / 'report.json', report)
    del old
    gc.collect()
    torch.cuda.empty_cache()
    for index, (raw, record) in enumerate(zip(normalized, prepared)):
        print(f'STEP 2: sample {index + 1}/4, tokens={record.n_real_tokens}; candidate vs existing safe path.', flush=True)
        items = [{'record': record, 'error': None}]
        flags = []
        old_i, old_d = I._encode_compatible, D._encode_compatible

        def traced_i(*a, **kw):
            flags.append(torch.is_grad_enabled())
            return old_i(*a, **kw)

        def traced_d(*a, **kw):
            flags.append(torch.is_grad_enabled())
            return old_d(*a, **kw)

        with torch.enable_grad():
            with patch.object(I, '_encode_compatible', traced_i), patch.object(D, '_encode_compatible', traced_d):
                actual = candidate._encode_items(encoder, cfg, [raw], items, 1, True, 'cuda:0')
            B.require(torch.is_grad_enabled(), 'Caller gradient state was not restored')
            reference = D.encode_records(encoder, fz, cfg, [raw], 1, True, 'cuda:0')
        check_result(actual, raw, record.n_real_tokens, cfg.embed_dim)
        check_result(reference, raw, record.n_real_tokens, cfg.embed_dim)
        B.require(flags and not any(flags), 'Candidate forward recorded gradients')
        graph_compare = B.compare_features(np, reference.embeddings, actual.embeddings)
        token_compare = B.compare_features(np, reference.tokens, actual.tokens)
        B.require(graph_compare['features_allclose'] and token_compare['features_allclose'],
                  'Candidate differs from established safe inference')
        with torch.enable_grad():
            graph_only = candidate._encode_items(encoder, cfg, [raw], items, 1, False, 'cuda:0')
        check_result(graph_only, raw, record.n_real_tokens, cfg.embed_dim, False)
        graph_only_compare = B.compare_features(np, actual.embeddings, graph_only.embeddings)
        B.require(graph_only.tokens is None and graph_only_compare['features_allclose'], 'Graph-only output differs')
        if index == 3:
            with torch.enable_grad():
                short_original = original._encode_items(encoder, cfg, [raw], items, 1, True, 'cuda:0')
            check_result(short_original, raw, record.n_real_tokens, cfg.embed_dim)
            B.require(B.compare_features(np, short_original.embeddings, actual.embeddings)['features_allclose'] and
                      B.compare_features(np, short_original.tokens, actual.tokens)['features_allclose'],
                      'Ordinary short output changed')
        report['samples'].append({'id': raw['id'], 'tokens': record.n_real_tokens,
                                  'windowed': original._can_window(record, cfg), 'failed': [],
                                  'candidate_forward_calls': len(flags), 'grad_enabled_in_forward': False,
                                  'embedding_comparison': graph_compare, 'token_comparison': token_compare,
                                  'graph_only_comparison': graph_only_compare})
        B.write_json(args.output / 'report.json', report)
    B.require(versions == {name: (p._version, p.requires_grad) for name, p in encoder.named_parameters()},
              'Parameter versions or requires_grad flags changed')
    B.require(all(p.grad is None for p in encoder.parameters()), 'Unexpected parameter gradients')
    B.require(hashlib.sha256(source.read_bytes()).hexdigest() == R.EXPECTED, 'Production changed')
    report.update(status='completed', production_unchanged=True, parameter_versions_unchanged=True,
                  notes=['Small representative test, not recovery of every historical failure.',
                         'OOM, HIP kernel errors and partial-batch retry duplication are outside this patch.'])
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
