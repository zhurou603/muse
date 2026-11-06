"""四域 tokenizer。

- SmilesTokenizer / PolymerSmilesTokenizer：化学感知正则切分（参照 TransPolymer 的
  tokenizer 思想：原子团、括号结构、方括号表达式作为整体 token），'*' 为一等公民。
- ResidueTokenizer：残基级（20 标准氨基酸 + 常见特殊残基 + 特殊符号）。
- 全部支持语料统计构建词表（build_vocab），也可直接默认词表。

无第三方依赖（不依赖 RDKit），保证无网络环境下可用。
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List

PAD, UNK, MASK, CLS, SEP, EOS = '<pad>', '<unk>', '<mask>', '<cls>', '<sep>', '<eos>'
SPECIALS = [PAD, UNK, MASK, CLS, SEP, EOS]

# 化学感知 SMILES 切分正则（先长后短）：方括号表达式 > 双字符原子 > 环闭合数字 > 单字符
_SMILES_PATTERN = re.compile(
    r"(\[[^\]]+\])"                       # [NH4+] 等方括号原子
    r"|(Br|Cl|Si|Se|Na|Li|Mg|Ca|Fe|Zn|Al)"  # 常见双字符原子
    r"|(%\d{2}|\d)"                        # 环闭合标记
    r"|(\*\(\)|\*)"                        # 聚合物位点 '*' / '(*)'
    r"|(==|#|=|-|/|\\|\.|:|~)"             # 键
    r"|([A-Za-z])"                         # 单字符原子/芳香原子
    r"|([()@+])"                           # 结构符号
)


class BaseTokenizer:
    name = 'base'

    def __init__(self, vocab: Dict[str, int] | None = None):
        self.vocab: Dict[str, int] = vocab or {t: i for i, t in enumerate(SPECIALS)}
        self.inv_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        for t in SPECIALS:
            assert t in self.vocab, f'vocab 缺少特殊 token {t}'
        self.pad_id, self.unk_id, self.mask_id = self.vocab[PAD], self.vocab[UNK], self.vocab[MASK]
        self.cls_id, self.sep_id, self.eos_id = self.vocab[CLS], self.vocab[SEP], self.vocab[EOS]

    def __len__(self):
        return len(self.vocab)

    def tokenize(self, s: str) -> List[str]:
        raise NotImplementedError

    def encode(self, s: str, add_cls: bool = True) -> List[int]:
        ids = [self.vocab.get(t, self.unk_id) for t in self.tokenize(s)]
        return ([self.cls_id] + ids) if add_cls else ids

    def build_vocab(self, corpus: Iterable[str], min_freq: int = 2, max_size: int = 10_000):
        cnt = Counter()
        for s in corpus:
            cnt.update(self.tokenize(s))
        for tok, c in cnt.most_common():
            if c < min_freq or len(self.vocab) >= max_size:
                continue
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        return self


class SmilesTokenizer(BaseTokenizer):
    name = 'smiles'

    def tokenize(self, s: str) -> List[str]:
        toks = [m.group(0) for m in _SMILES_PATTERN.finditer(s)]
        return toks if toks else [UNK]


class PolymerSmilesTokenizer(SmilesTokenizer):
    """PSMILES：与 SMILES 同一切分规则，但保证 '*' 位点 token 存在。"""
    name = 'psmiles'

    def __init__(self, vocab: Dict[str, int] | None = None):
        super().__init__(vocab)
        for t in ('*', '(*)'):
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab)
                self.inv_vocab[self.vocab[t]] = t


# 20 种标准氨基酸 + 常见特殊残基
_CANONICAL_AA = list('ACDEFGHIKLMNPQRSTVWY')
_EXTRA_AA = ['B', 'Z', 'X', 'U', 'O']  # 歧义/稀有残基


class ResidueTokenizer(BaseTokenizer):
    name = 'residue'

    def __init__(self, vocab: Dict[str, int] | None = None):
        if vocab is None:
            vocab = {t: i for i, t in enumerate(SPECIALS + _CANONICAL_AA + _EXTRA_AA)}
        super().__init__(vocab)

    def tokenize(self, s: str) -> List[str]:
        return [c.upper() for c in s.strip() if not c.isspace()]

    def encode(self, s: str, add_cls: bool = True) -> List[int]:
        ids = [self.vocab.get(c, self.unk_id) for c in self.tokenize(s)]
        return ([self.cls_id] + ids) if add_cls else ids


def default_tokenizers():
    return {
        'smiles': SmilesTokenizer(),
        'psmiles': PolymerSmilesTokenizer(),
        'residue': ResidueTokenizer(),
    }


def load_tokenizers(root=None):
    import json
    from pathlib import Path

    root = Path(root) if root is not None else Path(__file__).resolve().parents[2] / 'data/pretrain/vocab'
    tokenizers = default_tokenizers()
    for name in ('smiles', 'psmiles'):
        payload = json.loads((root / f'{name}.json').read_text(encoding='utf-8'))
        tokenizers[name] = (SmilesTokenizer(payload['vocab']) if name == 'smiles'
                            else PolymerSmilesTokenizer(payload['vocab']))
    return tokenizers
