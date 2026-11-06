# muse_encoder —— MUSE 赛题统一 Encoder 参考实现

对应《MUSE四域SOTA调研与预训练方案设计.md》§5–§6 的架构与预训练算法设计。
**全部参数从头训练**（无外部权重加载路径），四域共享单一 checkpoint，统一输出维度 D。

## 结构

| 文件 | 内容 |
|---|---|
| `config.py` | `MuseConfig`：主干 24L / d=1024 / 16 头 / FFN 2048 + 8 专家 MoLE，D=1024（完整配置约 386M 参数）；`pooling` 可选 `cls_mean`/`gru` |
| `tokenizers.py` | SMILES / PSMILES（化学感知正则，`'*'` 一等公民）/ 残基 tokenizer，支持语料建词表 |
| `featurize.py` | 四域前端特征化；RDKit 可选（无则降级路径）；晶体端内置纯 numpy PBC 超胞邻居搜索；逐条独立处理 |
| `layers.py` | RoPE、对偏置注意力（bond/SPD/距离高斯基 → 每头加性偏置）、MoLE FFN（域嵌入软路由） |
| `model.py` | `MuseEncoder`：前端嵌入 → 共享主干 → CLS⊕mean-pool（或 GRU 池化）→ D；可选晶体组成计数/原子数投影仍输出同一 D；六组预训练头与损失；MLP 探针参数合规校验 |
| `masking.py` | 掩码策略（报告 §5.1）：序列 15% 随机 MLM（80/10/10）、原子级 15% 元素掩码、**肽结构语义掩码**（Bulk/SideChain/Fragment 按 30/40/30 混合，PepLand 配方；缺标注时 span 降级） |
| `inference.py` | 冻结编码：`ids.txt` + `embeddings.npy`（+ 逐残基 `tokens.npy`/`token_offsets.npy`），float32，有限性断言 |
| `demo_smoke_test.py` | 端到端烟测（8 项，全绿）：`python -m muse_encoder.demo_smoke_test` |

## 预训练损失（`pretraining_losses`）

MLM/掩码原子（分域输出头）、坐标去噪（多 σ score matching，50% 概率连带掩对特征防捷径）、
能量/力/应力（Huber，力:能 ≈ 30:1）、描述符蒸馏（中部+尾部双读出）、
跨视图 InfoNCE（τ=0.07）、辅助性质多任务。权重按报告 §5.2 课程表分阶段配置。
晶体可启用 train-only 元素参考能残差目标；参考能文件与统计文件按 SHA256 握手，旧配置默认仍使用原始总能量。

## 烟测覆盖

1. 四域前向与读出形状；2. **批次无关性**（同一记录换 batch 组成，max|Δ|<1e-5）；
3. 六组损失前向+反向传播；4. 推理输出文件格式（含逐残基 offsets）；
5. MLP 探针 ≤2 层 ≤100K 参数校验（D=1024、hidden=48：回归 49,249 / 21 类 50,229）；
6. 完整配置参数量 386M；7. 三种掩码策略（序列期望掩码率 0.15、肽结构语义掩码
含降级路径、CLS 永不掩）；8. GRU 池化读出形状与批次无关性。

## 下一步（接训练管线时）

- 装 RDKit 后小分子/聚合物前端自动切换到完整图特征路径；
- 词表：用外部语料（PubChem/PI1M/UniRef50）调 `build_vocab` 后回填 `MuseConfig` 词表尺寸；
- 数据侧补 `aux_labels` 解析器（能量/力/应力/描述符 → `pretraining_losses` 的 targets 约定）；
- B 榜 MLP 权重转 safetensors（`0.weight/0.bias/2.weight/2.bias` 命名）。
