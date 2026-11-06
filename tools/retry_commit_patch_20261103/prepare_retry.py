"""Prepare a batch-result commit patch without editing production."""
import argparse
import ast
import difflib
import hashlib
from pathlib import Path

EXPECTED = 'ea0cf8245dbb97d4b2a2ea2ff63af89dc0ab81ffd99b7a9d3de4aab84cc7e34c'
SOURCE = Path('/yinghepool/tech-dcu/zhaoyao/model_design/muse_encoder/pretrain/parallel_dashboard.py')


def patch_source(text):
    start = text.index('\ndef _encode_items(')
    end = text.index('\ndef _encoding_job(', start)
    block = text[start:end]
    old = '        if not chunk_raw:\n            return\n        try:\n'
    new = ('        if not chunk_raw:\n            return\n'
           '        pending_ids = []\n        pending_embeddings = []\n        pending_tokens = []\n'
           '        try:\n')
    if block.count(old) != 1:
        raise ValueError('Unexpected chunk entry; stopped')
    changed = block.replace(old, new, 1)
    for before, after in [('ok_ids.append(', 'pending_ids.append('),
                          ('embeddings.append(', 'pending_embeddings.append('),
                          ('token_chunks.append(', 'pending_tokens.append(')]:
        if changed.count(before) != 2:
            raise ValueError(f'Unexpected output accumulation: {before}')
        changed = changed.replace(before, after)
    anchor = '            encode_chunk(chunk_raw[middle:], chunk_items[middle:])\n'
    if changed.count(anchor) != 1:
        raise ValueError('Unexpected retry branch; stopped')
    changed = changed.replace(anchor, anchor +
                              '        else:\n'
                              '            ok_ids.extend(pending_ids)\n'
                              '            embeddings.extend(pending_embeddings)\n'
                              '            token_chunks.extend(pending_tokens)\n', 1)
    result = text[:start] + changed + text[end:]
    before_ast, after_ast = ast.parse(text), ast.parse(result)
    for tree in (before_ast, after_ast):
        tree.body = [node for node in tree.body
                     if not isinstance(node, ast.FunctionDef) or node.name != '_encode_items']
    if ast.dump(before_ast) != ast.dump(after_ast):
        raise ValueError('Unexpected changes outside target function')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'candidate')
    args = parser.parse_args()
    raw = args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('Production version differs; stopped')
    text = raw.decode('utf-8').replace('\r\n', '\n')
    candidate = patch_source(text)
    compile(candidate, 'retry_parallel_dashboard.py', 'exec')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'parallel_dashboard.before_retry.py').write_bytes(raw)
    path = args.output / 'retry_parallel_dashboard.py'
    lines = candidate.splitlines(keepends=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for start in range(0, len(lines), 200):
            stream.writelines(lines[start:start + 200])
    diff = difflib.unified_diff(text.splitlines(True), candidate.splitlines(True),
                                fromfile='parallel_dashboard.py (deployed)',
                                tofile='parallel_dashboard.py (candidate)')
    (args.output / 'retry.patch').write_text(''.join(diff), encoding='utf-8')
    print(f'PREPARED: {path}; production unchanged')
    print(f'SHA256: {hashlib.sha256(path.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
