"""MuseEncoder 配置。

正式默认：共享主干 24 层 / hidden 1024 / 16 头 / FFN 2048 + 8 专家 MoLE，
统一读出维度 D=1024。所有尺寸在此集中声明，训练配置直接引用本 dataclass。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field


@dataclass
class MuseConfig:
    # ---- 词表（实际尺寸由 tokenizers 构建后回填） ----
    smiles_vocab_size: int = 715        # 实际语料构建词表（coverage.json 验证）
    psmiles_vocab_size: int = 74        # PI1M 全量词表（含 '*' 位点）
    residue_vocab_size: int = 32        # 氨基酸 + 特殊残基
    num_elements: int = 120             # 0=dummy，1–118=元素，119=原子 [MASK]
    atom_mask_id: int = 119
    num_bond_types: int = 8             # single/double/triple/aromatic/dummy/pbc/self/none
    num_spd_buckets: int = 16           # 最短路距离分桶
    num_dist_basis: int = 32            # 距离高斯基数

    # ---- 主干 ----
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    d_ffn: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 1024             # 统一 token 序列上限（含 [CLS]）
    use_rope: bool = True

    # ---- MoLE（UMA 式线性专家混合，按域软路由） ----
    num_experts: int = 8
    mole_every: int = 4                 # 每隔 mole_every 层把 FFN 换成 MoLE

    # ---- 域与模态 ----
    num_domains: int = 4                # small_molecule / polymer / crystal / bio
    num_modalities: int = 2             # 1D (无坐标) / 3D (有坐标)
    num_segments: int = 4               # 序列对 / 蛋白-配体段 id
    num_spacegroups: int = 231          # 0=缺失，1–230=空间群
    scalar_cond_dim: int = 4            # 组成比例、电荷、自旋、测量条件等标量
    # 原子序数(含mask)/手性/度数/电荷/氢数/自由基/杂化/芳香/环。
    atom_feature_vocab_sizes: tuple = (120, 8, 12, 12, 10, 6, 6, 2, 2)

    # ---- 读出 ----
    embed_dim: int = 1024               # D：所有领域统一表征维度（submission.json 的 embedding_dim）
    contrastive_dim: int = 256          # 跨视图对比投影维度
    pooling: str = 'cls_mean'           # 'cls_mean'（默认）| 'gru'（PepLand 消融更优，看板 A/B 后锁定）

    # ---- 预训练头 ----
    descriptor_dim: int = 208           # RDKit/Mordred 描述符回归维度
    # name -> {kind: graph/atom/bond, dim: int, task_type: regression/binary_classification}
    aux_specs: dict = field(default_factory=dict)
    denoise_sigma_min: float = 0.04     # Å
    denoise_sigma_max: float = 1.0      # Å
    pair_mask_prob: float = 0.5         # 去噪时连带掩掉原子辅助特征与对特征的概率
    mlm_mask_prob: float = 0.15
    conservative_force: bool = False    # 直接力阶段=False；保守力阶段 F=-∇E=True

    # ---- 晶体前端 ----
    crystal_cutoff: float = 6.0         # Å，PBC 邻居搜索截断
    crystal_max_neighbors: int = 32
    crystal_composition_features: bool = False
    crystal_composition_scale: float = 128.0

    # ---- 数值 ----
    layer_norm_eps: float = 1e-5
    use_sdpa: bool = True             # True=SDPA（有高效后端时省显存；本机 HIP 构建回落 math）
    gradient_checkpointing: bool = True   # 逐块重算激活（本机无 mem-efficient SDPA，必须开）

    def to_dict(self):
        return asdict(self)


# 领域常量（与 sample_type 对齐）
DOMAIN_SMALL_MOLECULE = 0
DOMAIN_POLYMER = 1
DOMAIN_CRYSTAL = 2
DOMAIN_BIO = 3

MODALITY_1D = 0
MODALITY_3D = 1
