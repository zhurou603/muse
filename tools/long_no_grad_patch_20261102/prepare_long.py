"""Add the same no_grad boundary already used by the ordinary inference path."""
import argparse
import difflib
import hashlib
from pathlib import Path

EXPECTED = '99d22c6c8f3f0f3f3954043f2f6b6f3dc4c30901a7d3a483095c7bb44a4f30b3'
SOURCE = Path('/yinghepool/tech-dcu/zhaoyao/model_design/muse_encoder/pretrain/parallel_dashboard.py')
MARKER = '\ndef _encode_items(encoder, cfg, raw, items, batch_size: int, token_level: bool, device: str):'


def patch_source(text):
    if text.count(MARKER) != 1 or '\n@torch.no_grad()' + MARKER in text:
        raise ValueError('Expected undecorated _encode_items not found; stopped')
    return text.replace(MARKER, '\n@torch.no_grad()' + MARKER, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'candidate')
    args = parser.parse_args()
    raw = args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('Production differs from verified cache+fixed-card version; stopped')
    text = raw.decode('utf-8').replace('\r\n', '\n')
    candidate = patch_source(text)
    compile(candidate, 'long_parallel_dashboard.py', 'exec')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'parallel_dashboard.before_long.py').write_bytes(raw)
    path = args.output / 'long_parallel_dashboard.py'
    lines = candidate.splitlines(keepends=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for start in range(0, len(lines), 200):
            stream.writelines(lines[start:start + 200])
    diff = difflib.unified_diff(text.splitlines(True), candidate.splitlines(True),
                                fromfile='parallel_dashboard.py (cache+fixed cards)',
                                tofile='parallel_dashboard.py (no_grad candidate)')
    (args.output / 'long.patch').write_text(''.join(diff), encoding='utf-8')
    print(f'PREPARED: {path}; production unchanged')
    print(f'SHA256: {hashlib.sha256(path.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
