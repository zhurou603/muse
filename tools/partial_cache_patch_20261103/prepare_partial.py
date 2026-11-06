"""Prepare partial feature reuse on top of the four deployed repairs."""
import argparse
import ast
import difflib
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED = '6889ba8a4a722455fa7d07633fe7765e3f61d56f48e4c584e0aee7977db17b32'
SOURCE = Path('/yinghepool/tech-dcu/zhaoyao/model_design/muse_encoder/pretrain/parallel_dashboard.py')


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Patch context not unique; stopped')
    return text.replace(old, new, 1)


def patch_source(text):
    original = text
    helper = (HERE / 'partial_helper.py').read_text(encoding='utf-8')
    text = once(text, '\ndef encode_records_parallel(', '\n' + helper + '\n\ndef encode_records_parallel(')
    text = once(text, '    ids = [record["id"] for record in records]\n    central_path',
                '    ids = [record["id"] for record in records]\n'
                '    if len(ids) != len(set(ids)):\n'
                '        raise ValueError("输入ID重复，不能按ID复用特征")\n    central_path')
    text = once(text, '        if cached is not None:\n            _write_cache(central_path, central_key, ids, cached)',
                '        if cached is not None:\n'
                '            _validate_encoded_cache(cached, ids, cfg.embed_dim, token_level, require_complete=True)\n'
                '            _write_cache(central_path, central_key, ids, cached)')
    text = once(text, '    if cached is not None:\n        print(f"[fast-eval] {namespace}: central embedding cache hit", flush=True)',
                '    if cached is not None:\n'
                '        _validate_encoded_cache(cached, ids, cfg.embed_dim, token_level, require_complete=True)\n'
                '        print(f"[fast-eval] {namespace}: central embedding cache hit", flush=True)')
    text = once(text, '\n    jobs = [\n        (index, batch, external_base, external_cache,',
                '\n    partial_path = _cache_path(cache_root, "embedding_progress_v1", namespace, central_key)\n'
                '    partial = _read_cache(partial_path, central_key, ids)\n'
                '    reused_ids = set()\n'
                '    if partial is not None:\n'
                '        _validate_encoded_cache(partial, ids, cfg.embed_dim, token_level)\n'
                '        if not partial.ids or not partial.failed:\n'
                '            raise ValueError("部分特征缓存必须同时包含成功和失败记录")\n'
                '        reused_ids = set(partial.ids)\n'
                '        remaining = [record for record in records if record["id"] not in reused_ids]\n'
                '        raw_batches = [remaining[start:start + batch_size]\n'
                '                       for start in range(0, len(remaining), batch_size)]\n'
                '        normalization_keys = [_normalization_key(batch, external_base, data_version)\n'
                '                              for batch in raw_batches]\n'
                '        print(f"[fast-eval] {namespace}: reuse={len(reused_ids)} retry={len(remaining)}", flush=True)\n'
                '\n    jobs = [\n        (index, batch, external_base, external_cache,')
    text = once(text, '    if not encoded.failed:\n        _write_cache(central_path, central_key, ids, encoded)',
                '    pending_ids = [record["id"] for record in records if record["id"] not in reused_ids]\n'
                '    _validate_encoded_cache(encoded, pending_ids, cfg.embed_dim, token_level)\n'
                '    if partial is not None:\n'
                '        retained = D.EncodedSplit(partial.ids, partial.embeddings, partial.tokens, partial.offsets, [])\n'
                '        encoded, _ = _merge_encoded([(0, retained, 0, 0), (1, encoded, 0, 0)], cfg.embed_dim, [])\n'
                '        rows = {record_id: index for index, record_id in enumerate(encoded.ids)}\n'
                '        ordered_ids = [record_id for record_id in ids if record_id in rows]\n'
                '        if ordered_ids != encoded.ids:\n'
                '            order = [rows[record_id] for record_id in ordered_ids]\n'
                '            encoded.embeddings = encoded.embeddings[order]\n'
                '            if encoded.tokens is not None:\n'
                '                chunks = [encoded.tokens[encoded.offsets[index]:encoded.offsets[index + 1]]\n'
                '                          for index in order]\n'
                '                encoded.tokens = np.concatenate(chunks, axis=0)\n'
                '                encoded.offsets = np.zeros(len(chunks) + 1, dtype=np.int64)\n'
                '                encoded.offsets[1:] = np.cumsum([len(chunk) for chunk in chunks])\n'
                '            encoded.ids = ordered_ids\n'
                '    _validate_encoded_cache(encoded, ids, cfg.embed_dim, token_level)\n'
                '    if not encoded.failed:\n'
                '        _write_cache(central_path, central_key, ids, encoded)\n'
                '    elif encoded.ids:\n'
                '        _write_cache(partial_path, central_key, ids, encoded)')
    text = once(text, '        "central_hit": False,\n',
                '        "central_hit": False,\n'
                '        "partial_hit": partial is not None,\n'
                '        "reused_embedding_rows": len(reused_ids),\n'
                '        "retried_records": len(pending_ids) if partial is not None else 0,\n')
    before = {n.name: ast.dump(n) for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)}
    after = {n.name: ast.dump(n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    if set(after) - set(before) != {'_validate_encoded_cache'} or any(
            after[n] != node for n, node in before.items() if n != 'encode_records_parallel'):
        raise ValueError('Unexpected changes outside cache orchestration')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=HERE / 'candidate')
    args = parser.parse_args()
    raw = args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('Production version differs; stopped')
    text = raw.decode('utf-8').replace('\r\n', '\n')
    candidate = patch_source(text)
    compile(candidate, 'partial_parallel_dashboard.py', 'exec')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'parallel_dashboard.before_partial.py').write_bytes(raw)
    path = args.output / 'partial_parallel_dashboard.py'
    lines = candidate.splitlines(keepends=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for start in range(0, len(lines), 200):
            stream.writelines(lines[start:start + 200])
    diff = difflib.unified_diff(text.splitlines(True), candidate.splitlines(True),
                                fromfile='deployed/parallel_dashboard.py', tofile='candidate/parallel_dashboard.py')
    (args.output / 'partial.patch').write_text(''.join(diff), encoding='utf-8')
    print(f'PREPARED: {path}; production unchanged')
    print(f'SHA256: {hashlib.sha256(path.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
