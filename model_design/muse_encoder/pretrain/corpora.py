from __future__ import annotations

import hashlib
import json
import random
import threading
from dataclasses import dataclass
from itertools import islice, zip_longest
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from ..tokenizers import PolymerSmilesTokenizer, SmilesTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRETRAIN_ROOT = PROJECT_ROOT / "data" / "pretrain"
CLEAN_ROOT = PRETRAIN_ROOT / "corpora_clean"
DESCRIPTOR_ROOT = PRETRAIN_ROOT / "descriptors"


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    path: Path
    kind: str
    records: int


@dataclass(frozen=True)
class TextRecord:
    source: str
    key: str
    text: str
    row: int
    view: int = 0
    crop_start: int = 0
    crop_seed: int = 0


CORPORA = {
    "pubchem": CorpusSpec("pubchem", CLEAN_ROOT / "pubchem.tsv", "tsv", 40_000_000),
    "chembl": CorpusSpec("chembl", CLEAN_ROOT / "chembl.tsv", "tsv", 2_870_569),
    "pi1m": CorpusSpec("pi1m", CLEAN_ROOT / "pi1m.tsv", "tsv", 995_542),
    "uniref50": CorpusSpec("uniref50", CLEAN_ROOT / "uniref50.fasta", "fasta", 53_624_246),
    "pdb": CorpusSpec("pdb", CLEAN_ROOT / "pdb.fasta", "fasta", 630_326),
    "uniparc_peptides": CorpusSpec(
        "uniparc_peptides", CLEAN_ROOT / "uniparc_peptides.fasta", "fasta", 8_000_000),
}


def iter_tsv(spec: CorpusSpec, limit: int | None = None) -> Iterator[TextRecord]:
    with spec.path.open(encoding="utf-8") as stream:
        for row, line in enumerate(islice(stream, limit)):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2 or not fields[0] or not fields[1]:
                raise ValueError(f"{spec.path}:{row + 1}: 非法 TSV 记录")
            yield TextRecord(spec.name, fields[0], fields[1], row)


def iter_fasta(spec: CorpusSpec, limit: int | None = None) -> Iterator[TextRecord]:
    key = None
    chunks = []
    row = 0
    with spec.path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if line.startswith(">"):
                if key is not None:
                    if limit is not None and row >= limit:
                        return
                    if not chunks:
                        raise ValueError(f"{spec.path}:{line_number}: 空 FASTA 序列")
                    yield TextRecord(spec.name, key, "".join(chunks), row)
                    row += 1
                key = line[1:].split()[0]
                if not key:
                    raise ValueError(f"{spec.path}:{line_number}: 空 FASTA id")
                chunks = []
            elif line:
                if key is None:
                    raise ValueError(f"{spec.path}:{line_number}: FASTA 缺少 header")
                chunks.append(line)
    if key is not None and (limit is None or row < limit):
        if not chunks:
            raise ValueError(f"{spec.path}: 文件末尾为空 FASTA 序列")
        yield TextRecord(spec.name, key, "".join(chunks), row)


def iter_corpus(name: str, limit: int | None = None) -> Iterator[TextRecord]:
    spec = CORPORA[name]
    return iter_tsv(spec, limit) if spec.kind == "tsv" else iter_fasta(spec, limit)


def crop_seed(seed: int, record: TextRecord) -> int:
    digest = hashlib.sha256(f"{seed}\0{record.source}\0{record.key}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def crop_sequence(record: TextRecord, max_residues: int, seed: int) -> TextRecord:
    if max_residues < 1:
        raise ValueError("max_residues 必须为正数")
    if len(record.text) <= max_residues:
        return record
    window_seed = crop_seed(seed, record)
    start = random.Random(window_seed).randrange(len(record.text) - max_residues + 1)
    return TextRecord(record.source, record.key,
                      record.text[start:start + max_residues], record.row,
                      record.view, start, window_seed)


def random_smiles_views(smiles: str, count: int, seed: int) -> list[str]:
    if count < 1:
        return []
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit 无法解析 PSMILES: {smiles[:80]}")
    rng = random.Random(seed)
    views = [smiles]
    atom_count = mol.GetNumAtoms()
    while len(views) < count:
        order = list(range(atom_count))
        rng.shuffle(order)
        renumbered = Chem.RenumberAtoms(mol, order)
        views.append(Chem.MolToSmiles(renumbered, canonical=False, isomericSmiles=True))
    return views


def iter_pi1m_views(seed: int = 42, views_per_record: int = 5,
                    limit: int | None = None) -> Iterator[TextRecord]:
    if views_per_record != 5:
        raise ValueError("正式 TransPolymer 配方固定为每条总计 5 个视图")
    for record in iter_corpus("pi1m", limit):
        view_seed = crop_seed(seed, record)
        for view, smiles in enumerate(random_smiles_views(record.text, views_per_record, view_seed)):
            yield TextRecord(record.source, record.key, smiles, record.row, view)


def validate_row_alignment(text_path: Path, keys_path: Path,
                           limit: int | None = None) -> int:
    with text_path.open(encoding="utf-8") as text_stream, keys_path.open(encoding="utf-8") as key_stream:
        texts = islice(text_stream, limit)
        keys = islice(key_stream, limit)
        count = 0
        for row, (text_line, key_line) in enumerate(zip_longest(texts, keys)):
            if text_line is None or key_line is None:
                raise ValueError(f"语料与 descriptor keys 行数不一致: row={row}")
            fields = text_line.rstrip("\n").split("\t")
            if len(fields) < 2 or fields[0] != key_line.rstrip("\n"):
                raise ValueError(f"语料与 descriptor keys 在第 {row} 行错位")
            count += 1
    return count


class DescriptorStore:
    def __init__(self, name: str, root: Path = DESCRIPTOR_ROOT):
        self.name = name
        self.values_path = root / f"{name}.desc.f32.npy"
        self.keys_path = root / f"{name}.keys.txt"
        self._mmap = np.load(self.values_path, mmap_mode="r")
        # mmap 随机页读在共享文件系统上会偶发数分钟级停顿（拖垮 NCCL 集合）；
        # 内存充足时整体读入（pubchem 33GB，各 rank 持有独立内存副本）。
        self.values = np.asarray(self._mmap)
        if self.values.ndim != 2 or self.values.shape[1] != 208 or self.values.dtype != np.float32:
            raise ValueError(f"{self.values_path}: 期望 float32[N,208]，实际 {self.values.dtype}{self.values.shape}")

    def __len__(self):
        return self.values.shape[0]

    def validate(self, limit: int | None = None) -> int:
        count = validate_row_alignment(CLEAN_ROOT / f"{self.name}.tsv", self.keys_path, limit)
        if limit is None and count != len(self):
            raise ValueError(f"{self.name}: keys/语料 {count} 行，描述符 {len(self)} 行")
        if limit is not None and limit > len(self):
            raise ValueError(f"{self.name}: 描述符不足 {limit} 行")
        return count

    def get(self, rows: np.ndarray) -> np.ndarray:
        values = np.asarray(self.values[rows], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{self.name}: 描述符含 NaN/Inf")
        return values


def compute_global_norm(stores: Iterable[DescriptorStore], output: Path | None = None,
                        chunk_size: int = 65_536) -> dict:
    count = 0
    total = None
    total_sq = None
    for store in stores:
        for start in range(0, len(store), chunk_size):
            values = np.asarray(store.values[start:start + chunk_size], dtype=np.float64)
            if total is None:
                total = np.zeros(values.shape[1], dtype=np.float64)
                total_sq = np.zeros(values.shape[1], dtype=np.float64)
            total += values.sum(axis=0)
            total_sq += np.square(values).sum(axis=0)
            count += values.shape[0]
    if not count:
        raise ValueError("没有可用于 MTR 归一化的描述符")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    norm = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "mask": (std > 1e-12).tolist(),
        "n": count,
        "dim": len(mean),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(norm, separators=(",", ":")), encoding="utf-8")
    return norm


class DescriptorNormalizer:
    def __init__(self, norm: dict):
        self.mean = np.asarray(norm["mean"], dtype=np.float32)
        self.std = np.asarray(norm["std"], dtype=np.float32)
        self.mask = np.asarray(norm["mask"], dtype=bool)
        if not (self.mean.shape == self.std.shape == self.mask.shape == (208,)):
            raise ValueError("MTR norm 必须为 208 维")

    @classmethod
    def from_path(cls, path: Path):
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def __call__(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        safe_std = np.where(self.mask, self.std, 1.0)
        normalized = (values - self.mean) / safe_std
        normalized = np.where(self.mask, normalized, 0.0).astype(np.float32)
        mask = np.broadcast_to(self.mask, values.shape).copy()
        if not np.isfinite(normalized).all():
            raise ValueError("归一化描述符含 NaN/Inf")
        return normalized, mask


def build_chemical_vocab(records: Iterable[TextRecord], polymer: bool,
                         output: Path, min_freq: int = 2,
                         max_size: int = 4096) -> dict:
    tokenizer = PolymerSmilesTokenizer() if polymer else SmilesTokenizer()
    tokenizer.build_vocab((r.text for r in records), min_freq=min_freq, max_size=max_size)
    payload = {"name": tokenizer.name, "vocab": tokenizer.vocab,
               "min_freq": min_freq, "max_size": max_size}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def vocab_coverage(records: Iterable[TextRecord], vocab: dict, polymer: bool = False) -> dict:
    tokenizer = PolymerSmilesTokenizer(vocab) if polymer else SmilesTokenizer(vocab)
    total = with_unk = tokens = unk_tokens = 0
    for record in records:
        ids = tokenizer.encode(record.text, add_cls=False)
        n_unk = ids.count(tokenizer.unk_id)
        total += 1
        with_unk += int(n_unk > 0)
        tokens += len(ids)
        unk_tokens += n_unk
    return {
        "records": total,
        "records_with_unk": with_unk,
        "record_coverage": 1.0 - with_unk / max(total, 1),
        "tokens": tokens,
        "unk_tokens": unk_tokens,
        "token_coverage": 1.0 - unk_tokens / max(tokens, 1),
    }
