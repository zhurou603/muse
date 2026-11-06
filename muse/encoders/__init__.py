"""编码器插件包。统一约定：

encode_*(...) -> 在缓存目录写出 {split}.emb.npy (float32 [N, D]) 与 {split}.ids.txt。
逐残基任务额外写 {split}.tokens.npy 与 {split}.token_counts.txt（P4）。
"""
