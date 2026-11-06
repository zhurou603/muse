"""Exercise the actual dispatch function with deterministic small tensor outputs."""
import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


@dataclass
class Result:
    ids: list
    embeddings: object
    tokens: object
    offsets: object
    failed: list


def load_function(text):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == '_encode_items')
    scope = {'torch': torch, 'np': np}

    def long(encoder, fz, cfg, record, batch, token_level, device):
        if record.bad:
            raise ValueError('injected compute failure')
        return np.full(2, record.number, np.float32), np.full((record.count, 2), record.number, np.float32)

    def regular(encoder, cfg, records, device):
        if any(r.bad for r in records) or (len(records) > 1 and any(r.batch_bad for r in records)):
            raise ValueError('injected compute failure')
        return {'embedding': torch.tensor([[r.number] * 2 for r in records], dtype=torch.float32),
                'tokens': torch.tensor([[[r.number] * 2] * 4 for r in records], dtype=torch.float32)}

    def token_output(record, values):
        return values[:record.count - (1 if record.token_bad else 0)]

    scope.update(_can_window=lambda r, cfg: r.long, _encode_long_record=long,
                 _expected_tokens=lambda raw, r: r.count, _token_output=token_output,
                 D=SimpleNamespace(EncodedSplit=Result, _encode_prepared_chunk=regular))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '_encode_items', 'exec'), scope)
    return scope['_encode_items']


def encode(function, specs, token_level=True):
    raw = [{'id': str(i)} for i in range(len(specs))]
    records = [SimpleNamespace(number=i + 1, count=2, long=s.get('long', False),
                               bad=s.get('bad', False), token_bad=s.get('token_bad', False),
                               batch_bad=s.get('batch_bad', False))
               for i, s in enumerate(specs)]
    items = [{'record': r, 'error': 'injected preparation failure' if s.get('prepare_bad') else None}
             for r, s in zip(records, specs)]
    return function(None, SimpleNamespace(embed_dim=2), raw, items, len(specs) or 1, token_level, 'cpu')


def run_checks(text):
    function = load_function(text)
    cases = {
        'long_success_then_compute_failure': [{'long': True}, {'bad': True}],
        'regular_success_then_token_failure': [{}, {'token_bad': True}],
        'nested_retry_preserves_siblings': [{'long': True}, {'long': True}, {'long': True, 'bad': True}, {}, {}],
        'all_success': [{'long': True}, {}, {}],
        'mixed_order_success': [{}, {'long': True}, {}],
        'batch_failure_recovers_without_failed_records': [{'long': True}, {'batch_bad': True}, {}],
        'all_failed': [{'bad': True}, {'long': True, 'bad': True}],
        'preparation_failure': [{'long': True}, {'prepare_bad': True}],
        'empty': [],
    }
    completed = []
    for name, specs in cases.items():
        for token_level in (True, False):
            result = encode(function, specs, token_level)
            good = [str(i) for i, s in enumerate(specs)
                    if not (s.get('bad') or s.get('prepare_bad') or (token_level and s.get('token_bad')))]
            bad = sorted(set(map(str, range(len(specs)))) - set(good))
            assert sorted(result.ids) == good, (name, token_level, result.ids, good)
            assert sorted(r['id'] for r in result.failed) == bad, (name, result.failed, bad)
            assert all('injected' in r['error'] or 'token' in r['error'] for r in result.failed)
            expected = np.asarray([[int(i) + 1] * 2 for i in result.ids], dtype=np.float32).reshape(-1, 2)
            np.testing.assert_array_equal(result.embeddings, expected)
            if token_level:
                np.testing.assert_array_equal(result.tokens, np.repeat(expected, 2, axis=0))
                np.testing.assert_array_equal(result.offsets, np.arange(len(good) + 1) * 2)
            else:
                assert result.tokens is None and result.offsets is None
            completed.append(f'{name}/tokens={token_level}')
    return completed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    print(run_checks(args.source.read_text(encoding='utf-8')))
