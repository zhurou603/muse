def _validate_encoded_cache(encoded, ids, embed_dim, token_level, *, require_complete=False):
    success = encoded.ids
    failures = [item["id"] for item in encoded.failed]
    if (len(ids) != len(set(ids)) or len(success) != len(set(success))
            or len(failures) != len(set(failures)) or set(success) & set(failures)
            or set(success) | set(failures) != set(ids)):
        raise ValueError("特征缓存ID重复、缺失或成功失败重叠")
    if any(not isinstance(item.get("error"), str) or not item["error"] for item in encoded.failed):
        raise ValueError("特征缓存失败原因缺失")
    if require_complete and failures:
        raise ValueError("完整特征缓存包含失败记录")
    if (not isinstance(encoded.embeddings, np.ndarray)
            or encoded.embeddings.shape != (len(success), embed_dim)
            or not np.isfinite(encoded.embeddings).all()):
        raise ValueError("特征缓存embedding形状或数值无效")
    if not token_level:
        if encoded.tokens is not None or encoded.offsets is not None:
            raise ValueError("特征缓存token模式不匹配")
        return
    if not success and encoded.tokens is None and encoded.offsets is None:
        return
    if (not isinstance(encoded.tokens, np.ndarray) or encoded.tokens.ndim != 2
            or encoded.tokens.shape[1] != embed_dim or not np.isfinite(encoded.tokens).all()
            or not isinstance(encoded.offsets, np.ndarray)
            or encoded.offsets.shape != (len(success) + 1,)
            or not np.issubdtype(encoded.offsets.dtype, np.integer)
            or encoded.offsets[0] != 0 or encoded.offsets[-1] != len(encoded.tokens)
            or (np.diff(encoded.offsets) < 0).any()):
        raise ValueError("特征缓存token/offsets不对齐或数值无效")
