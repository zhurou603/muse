"""ESM-2 650M 序列编码器（蛋白/肽）。

输入：氨基酸序列（或序列对、蛋白+配体SMILES）。
输出：池化表征 [1280]；逐残基任务另输出 tokens [L, 1280] 与 offsets。
序列对/蛋白+配体：各部分分别编码后拼接（维度 2560 / 1280+1024）。

v0 处理约定：
- 非标准字符（小写非天然残基等）映射为 'X' 并计数；
- 序列截断到 max_len（PEER/CATH 绝大多数远短于此；截断记录数）。
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/yinghepool/tech-dcu/zhaoyao")
ESM2_DIR = ROOT / "models/esm2-650m"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq):
    """大写；非标准残基替换为 X。返回 (序列, 替换数)。"""
    s = seq.upper()
    bad = sum(1 for c in s if c not in VALID_AA)
    if bad:
        s = "".join(c if c in VALID_AA else "X" for c in s)
    return s, bad


class ESM2Encoder:
    def __init__(self, model_dir=ESM2_DIR, device_id=0, max_len=1022):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.torch = torch
        self.device = torch.device(f"cuda:{device_id}")
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModel.from_pretrained(
            str(model_dir), torch_dtype=torch.float32).to(self.device).eval()
        self.dim = self.model.config.hidden_size

    def encode(self, sequences, batch_size=16, resume_path=None):
        """返回 (pooled [N, 1280] float32, tokens list[ np [L,1280] ] 或 None)。

        resume_path: 给定则每 25 批落盘一次断点，中断后从断点继续（仅池化表征；
        逐残基 tokens 不续跑——逐残基子集都很小，不需要）。
        """
        torch = self.torch
        rp = Path(resume_path) if resume_path else None
        done, pooled_parts = 0, []
        if rp and rp.exists():
            z = np.load(rp)
            done = int(z["done"])
            pooled_parts.append(z["pooled"])
            print(f"  从断点继续: {done}/{len(sequences)}", flush=True)
        tokens_all = []
        for i in range(done, len(sequences), batch_size):
            chunk = sequences[i:i + batch_size]
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_len + 2)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                hidden = self.model(**enc).last_hidden_state.float()
            mask = enc["attention_mask"].bool()
            # 去掉 BOS/EOS：attention_mask 含 special tokens，用序列内容区间
            pooled_rows = []
            for j in range(hidden.shape[0]):
                L = int(mask[j].sum().item()) - 2  # 去掉 <cls> <eos>
                res = hidden[j, 1:1 + L]           # [L, D]
                pooled_rows.append(res.mean(dim=0).cpu().numpy())
                tokens_all.append(res.cpu().numpy())
            pooled_parts.append(np.stack(pooled_rows))
            if rp and ((i // batch_size) % 50 == 49 or i + batch_size >= len(sequences)):
                np.savez(rp, done=i + len(chunk),
                         pooled=np.concatenate(pooled_parts, axis=0))
        pooled = np.concatenate(pooled_parts, axis=0).astype(np.float32)
        if rp and rp.exists():
            rp.unlink()
        return pooled, tokens_all


def encode_subset(subset, cache_dir, device_id=0, batch_size=16,
                  unimol2_embed_fn=None):
    """按 sample_type 分发编码，写 {split}.jsonl / .ids.txt / .emb.npy（+ tokens）。

    unimol2_embed_fn: protein_ligand 任务的配体编码器 (smiles list -> [N,1024])，
    由调用方注入以避免重复加载 Uni-Mol2。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    enc = None
    stats = {"replaced_residues": 0, "truncated_records": 0}

    def get_encoder():
        nonlocal enc
        if enc is None:
            enc = ESM2Encoder(device_id=device_id)
        return enc

    for split in ("train", "validation", "test"):
        emb_done = (cache_dir / f"{split}.emb.npy").exists() and \
            (cache_dir / f"{split}.ids.txt").exists()
        if emb_done:
            print(f"SKIP {subset.dataset}/{subset.subset}/{split}: 表征已存在", flush=True)
            continue
        records = [json.loads(l) for l in
                   subset.split_file(split).open(encoding="utf-8")]
        ids = [r["id"] for r in records]
        keys = subset.input_keys
        token_level = any(t.task_type == "sequence_classification" for t in subset.tasks)

        if keys == ["sequence"] or keys == ["context", "sequence"]:
            seqs, nbad = [], 0
            for r in records:
                s, b = clean_sequence(r["inputs"]["sequence"])
                seqs.append(s)
                nbad += b
            stats["replaced_residues"] += nbad
            rp = None if token_level else cache_dir / f"{split}.partial.npz"
            pooled, tokens = get_encoder().encode(seqs, batch_size, resume_path=rp)
        elif keys == ["sequence_1", "sequence_2"]:
            e = get_encoder()
            p1, _ = e.encode([clean_sequence(r["inputs"]["sequence_1"])[0]
                              for r in records], batch_size,
                             resume_path=cache_dir / f"{split}.p1.partial.npz")
            p2, _ = e.encode([clean_sequence(r["inputs"]["sequence_2"])[0]
                              for r in records], batch_size,
                             resume_path=cache_dir / f"{split}.p2.partial.npz")
            pooled, tokens = np.concatenate([p1, p2], axis=1), None
        elif keys == ["ligand_smiles", "protein_sequence"]:
            if unimol2_embed_fn is None:
                raise ValueError("protein_ligand 需要 unimol2_embed_fn")
            p_seq, _ = get_encoder().encode(
                [clean_sequence(r["inputs"]["protein_sequence"])[0]
                 for r in records], batch_size)
            p_lig = unimol2_embed_fn([r["inputs"]["ligand_smiles"] for r in records])
            pooled, tokens = np.concatenate([p_seq, p_lig], axis=1), None
        else:
            raise ValueError(f"未支持的 input_keys: {keys}")

        with (cache_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({"id": r["id"], "labels": r.get("labels", {})},
                                   ensure_ascii=False, allow_nan=False) + "\n")
        (cache_dir / f"{split}.ids.txt").write_text(
            "\n".join(ids) + "\n", encoding="utf-8")
        np.save(cache_dir / f"{split}.emb.npy", pooled.astype(np.float32))
        if token_level and tokens is not None:
            counts = [t.shape[0] for t in tokens]
            flat = np.concatenate(tokens, axis=0).astype(np.float32)
            np.save(cache_dir / f"{split}.tokens.npy", flat)
            np.save(cache_dir / f"{split}.token_offsets.npy",
                    np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))
        print(f"BIO-ENCODED {subset.dataset}/{subset.subset}/{split}: "
              f"{pooled.shape}", flush=True)
    (cache_dir / "encode_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    return stats
