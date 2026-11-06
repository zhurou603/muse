"""Prepare a cache-key-only candidate on top of the deployed fixed-card version."""
import argparse
import difflib
import hashlib
from pathlib import Path

EXPECTED = '2f3e3dcdd8c550a014f7ca696697809eb26f1f0b2a16c7270bdede66e6bfe1d6'
SOURCE = Path('/yinghepool/tech-dcu/zhaoyao/model_design/muse_encoder/pretrain/parallel_dashboard.py')


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Patch context not unique; stopped')
    return text.replace(old, new, 1)


def patch_source(text):
    text = once(text, '    central_key = _digest({', '    central_payload = {')
    text = once(text, '        "probe_config": probe_config.to_json(),\n    })\n    ids =',
                '    }\n    central_key = _digest(central_payload)\n    ids =')
    text = once(text,
                '    cached = _read_cache(central_path, central_key, ids)\n    if cached is not None:',
                '    cached = _read_cache(central_path, central_key, ids)\n'
                '    if cached is None:\n'
                '        legacy_key = _digest({**central_payload, "probe_config": probe_config.to_json()})\n'
                '        legacy_path = _cache_path(cache_root, "embeddings", namespace, legacy_key)\n'
                '        cached = _read_cache(legacy_path, legacy_key, ids)\n'
                '        if cached is not None:\n'
                '            _write_cache(central_path, central_key, ids, cached)\n'
                '    if cached is not None:')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'candidate')
    args = parser.parse_args()
    raw = args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('Production differs from verified fixed-card version; stopped')
    text = raw.decode('utf-8').replace('\r\n', '\n')
    candidate = patch_source(text)
    compile(candidate, 'cache_parallel_dashboard.py', 'exec')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'parallel_dashboard.before_cache.py').write_bytes(raw)
    path = args.output / 'cache_parallel_dashboard.py'
    lines = candidate.splitlines(keepends=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for start in range(0, len(lines), 200):
            stream.writelines(lines[start:start + 200])
    diff = difflib.unified_diff(text.splitlines(True), candidate.splitlines(True),
                                fromfile='parallel_dashboard.py (fixed cards)',
                                tofile='parallel_dashboard.py (cache candidate)')
    (args.output / 'cache.patch').write_text(''.join(diff), encoding='utf-8')
    print(f'PREPARED: {path}; production unchanged')
    print(f'SHA256: {hashlib.sha256(path.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
