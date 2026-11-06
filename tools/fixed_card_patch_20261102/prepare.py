"""Create a fixed-device candidate beside this tool; never overwrite production."""
import argparse
import difflib
import hashlib
from pathlib import Path

EXPECTED = '2a9afbd3e51900bb4306250a8e7a4ab9343d7096af8ce6f5b018b6cbc8a07a0a'
SOURCE = Path('/yinghepool/tech-dcu/zhaoyao/model_design/muse_encoder/pretrain/parallel_dashboard.py')


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Patch context not unique: {old[:90]}')
    return text.replace(old, new, 1)


def patch_source(text):
    text = replace_once(text, '_ENCODER_CHECKPOINT = None',
                        '_ENCODER_CHECKPOINT = None\n_ENCODING_DEVICE = None')
    text = replace_once(text, '    started = time.monotonic()\n    ids = [record["id"] for record in raw]',
                        '    if _ENCODING_DEVICE is not None:\n'
                        '        device = _ENCODING_DEVICE\n'
                        '    started = time.monotonic()\n    ids = [record["id"] for record in raw]')
    text = replace_once(text, 'def _parallel_map(function, jobs, processes: int):',
                        'def _init_encoding_worker(device_queue):\n'
                        '    global _ENCODING_DEVICE\n'
                        '    _ENCODING_DEVICE = device_queue.get(timeout=30)\n\n\n'
                        'def _parallel_map(function, jobs, processes: int, *, devices=None):')
    text = replace_once(text, '    if processes == 1:\n        results = []',
                        '    if devices is not None:\n'
                        '        devices = tuple(devices)\n'
                        '        if len(devices) != processes or len(set(devices)) != processes:\n'
                        '            raise ValueError("Encoding workers require one distinct device each")\n'
                        '    if not jobs:\n'
                        '        return []\n'
                        '    if processes == 1:\n        results = []')
    old = '''    with context.Pool(processes=processes) as pool:
        for completed, result in enumerate(pool.imap_unordered(function, jobs, chunksize=1), start=1):
            results.append(result)
            report(completed)
    return results'''
    new = '''    device_queue = None
    options = {}
    if devices is not None:
        device_queue = context.Queue()
        for device in devices:
            device_queue.put(device)
        options = {"initializer": _init_encoding_worker, "initargs": (device_queue,)}
    try:
        with context.Pool(processes=processes, **options) as pool:
            workers = list(pool._pool)
            iterator = pool.imap_unordered(function, jobs, chunksize=1)
            while len(results) < total:
                if devices is not None and any(not worker.is_alive() for worker in workers):
                    raise RuntimeError("Encoding worker exited; refusing silent worker replacement")
                try:
                    result = iterator.next(timeout=1)
                except mp.TimeoutError:
                    continue
                results.append(result)
                report(len(results))
    finally:
        if device_queue is not None:
            device_queue.close()
            device_queue.join_thread()
    return results'''
    text = replace_once(text, old, new)
    text = replace_once(text,
                        '        parts = _parallel_map(_encoding_job, encoding_jobs, min(len(devices), len(encoding_jobs)))',
                        '        worker_count = min(len(devices), len(encoding_jobs))\n'
                        '        parts = _parallel_map(_encoding_job, encoding_jobs, worker_count,\n'
                        '                              devices=devices[:worker_count])')
    compile(text, 'fixed_parallel_dashboard.py', 'exec')
    return text


def write_chunks(path, text):
    lines = text.splitlines(keepends=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for start in range(0, len(lines), 200):
            stream.writelines(lines[start:start + 200])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    args = parser.parse_args()
    source = args.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != EXPECTED:
        raise ValueError('Production source SHA256 changed; no candidate created')
    original = source.decode('utf-8').replace('\r\n', '\n')
    candidate = patch_source(original)
    target = Path(__file__).resolve().parent / 'candidate'
    target.mkdir(exist_ok=False)
    (target / 'parallel_dashboard.original.py').write_bytes(source)
    write_chunks(target / 'fixed_parallel_dashboard.py', candidate)
    write_chunks(target / 'parallel_dashboard.patch', ''.join(difflib.unified_diff(
        original.splitlines(True), candidate.splitlines(True),
        fromfile='parallel_dashboard.py (original)', tofile='parallel_dashboard.py (candidate)')))
    print(f'PREPARED: {target}; production unchanged; no GPU task started')


if __name__ == '__main__':
    main()
