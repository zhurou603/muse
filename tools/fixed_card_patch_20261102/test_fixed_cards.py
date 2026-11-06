"""Exercise the patched source with real spawn workers and a fake CPU Encoder."""
import ast
from collections import defaultdict
import multiprocessing as mp
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import prepare
import verify

SOURCE = Path(__file__).resolve().parents[2] / '1028/model_design/muse_encoder/pretrain/parallel_dashboard.py'
# The distribution contains the exact original backup after prepare.py runs.
if not SOURCE.is_file():
    SOURCE = Path(__file__).resolve().parent / 'candidate/parallel_dashboard.original.py'
TEXT = prepare.patch_source(SOURCE.read_text(encoding='utf-8'))
NAMES = {'_init_encoding_worker', '_parallel_map', '_encoding_job', '_load_encoder'}
tree = ast.parse(TEXT)
nodes = [n for n in tree.body if getattr(n, 'name', None) in NAMES or
         (isinstance(n, ast.ImportFrom) and n.module == '__future__')]
_ENCODING_DEVICE = None
_ENCODER = None
_ENCODER_CHECKPOINT = None
_LOADS = 0


def fake_load(path, device):
    global _LOADS
    _LOADS += 1
    encoder = SimpleNamespace(cfg=SimpleNamespace(embed_dim=2), device=device)
    return encoder, encoder.cfg, {}


def fake_encode(encoder, cfg, raw, items, batch_size, token_level, device):
    time.sleep(0.03)
    return {'ids': [r['id'] for r in raw], 'value': [r['value'] * 2 for r in raw],
            'device': device, 'pid': os.getpid(), 'loads': _LOADS}


D = SimpleNamespace(load_checkpoint=fake_load)
torch = SimpleNamespace(OutOfMemoryError=MemoryError)
_read_cache = lambda *args: ['prepared']
_encode_items = fake_encode
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), globals())


def job(index):
    return (index, [{'id': str(index), 'value': index}], 'same.pt', f'cuda:{index % 2}',
            Path('unused'), 'key', 4, False)


def normal_job(value):
    return value * 2, _ENCODING_DEVICE


def broken_job(value):
    raise ValueError('injected failure')


def dead_job(value):
    os._exit(7)


class FixedCardTests(unittest.TestCase):
    def setUp(self):
        global _ENCODING_DEVICE, _ENCODER, _ENCODER_CHECKPOINT, _LOADS
        _ENCODING_DEVICE = _ENCODER = _ENCODER_CHECKPOINT = None
        _LOADS = 0

    def test_real_encoding_jobs_keep_device_and_load_once(self):
        tasks = [job(i) for i in range(32)]
        output = _parallel_map(_encoding_job, tasks, 2, devices=['cuda:0', 'cuda:1'])
        self.assertEqual(sorted(p[0] for p in output), list(range(32)))
        by_pid = defaultdict(list)
        for index, result, _, _ in output:
            self.assertEqual(result['ids'], [str(index)])
            self.assertEqual(result['value'], [index * 2])
            by_pid[result['pid']].append(result)
        self.assertEqual(len(by_pid), 2)
        used = []
        for results in by_pid.values():
            devices = {r['device'] for r in results}
            self.assertEqual(len(devices), 1)
            self.assertEqual({r['loads'] for r in results}, {1})
            used.extend(devices)
        self.assertEqual(sorted(used), ['cuda:0', 'cuda:1'])
        self.assertIsNone(_ENCODING_DEVICE)

    def test_single_process_and_normalization_remain_compatible(self):
        jobs = [(*job(i)[:3], 'cuda:0', *job(i)[4:]) for i in range(3)]
        output = _parallel_map(_encoding_job, jobs, 1, devices=['cuda:0'])
        self.assertEqual([p[1]['loads'] for p in output], [1, 1, 1])
        normalized = _parallel_map(normal_job, [1, 2, 3], 2)
        self.assertEqual(sorted(normalized), [(2, None), (4, None), (6, None)])

    def test_no_jobs_and_invalid_devices(self):
        self.assertEqual(_parallel_map(normal_job, [], 2, devices=['cuda:0', 'cuda:1']), [])
        with self.assertRaisesRegex(ValueError, 'distinct device'):
            _parallel_map(normal_job, [1], 2, devices=['cuda:0', 'cuda:0'])

    def test_worker_errors_propagate(self):
        with self.assertRaisesRegex(ValueError, 'injected failure'):
            _parallel_map(broken_job, [1, 2], 2, devices=['cuda:0', 'cuda:1'])
        self.assertFalse(mp.active_children())

    def test_worker_death_is_not_silently_replaced(self):
        with self.assertRaisesRegex(RuntimeError, 'worker exited'):
            _parallel_map(dead_job, [1, 2], 2, devices=['cuda:0', 'cuda:1'])
        self.assertFalse(mp.active_children())

    def test_production_caller_passes_devices_only_to_encoding_pool(self):
        function = next(n for n in tree.body if getattr(n, 'name', '') == 'encode_records_parallel')
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == '_parallel_map']
        self.assertEqual(len(calls), 2)
        for call in calls:
            keywords = {kw.arg for kw in call.keywords}
            self.assertEqual('devices' in keywords, call.args[0].id == '_encoding_job')

    def test_candidate_contains_only_scoped_function_changes(self):
        original = ast.parse(SOURCE.read_text(encoding='utf-8'))
        old = {n.name: ast.dump(n) for n in original.body if isinstance(n, ast.FunctionDef)}
        new = {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
        changed = {name for name in old if old[name] != new[name]}
        self.assertEqual(changed, {'_parallel_map', '_encoding_job', 'encode_records_parallel'})
        self.assertEqual(set(new) - set(old), {'_init_encoding_worker'})
        self.assertEqual((Path(__file__).parent / 'candidate/fixed_parallel_dashboard.py').read_text(
            encoding='utf-8'), TEXT)

    def test_server_trace_uses_candidate_effective_device(self):
        import sys
        fake_module = SimpleNamespace(_ENCODING_DEVICE='cuda:1')
        fake_pretrain = SimpleNamespace(dashboard=object(), parallel_dashboard=object())
        fake_np = SimpleNamespace()
        with patch.dict(sys.modules, {'muse_encoder.pretrain': fake_pretrain,
                                     'numpy': fake_np, 'torch': object()}), \
                patch.object(verify.B, 'configure'), \
                patch.object(verify, 'candidate_module', return_value=fake_module), \
                patch.object(verify.B, 'trace_job', side_effect=lambda *args: args) as trace:
            result = verify.traced_job(('candidate', (job(0), '.', '.', {})))
            self.assertEqual(result[0][3], 'cuda:1')
            self.assertIs(result[2], fake_module)
            self.assertEqual(trace.call_count, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
