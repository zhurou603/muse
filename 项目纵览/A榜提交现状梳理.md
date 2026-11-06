# A 榜提交全流程现状梳理（截至 2026-10-31）

## 1. 流程定义（已固化）

正式流程已由 `官方提交/官方流程提交SOP.md` 固化，五步：

1. **冻结 Encoder**：`model_design/runs/s3-residual-pilot-02/checkpoints/final.pt`
   （S3 残差 pilot，step=100 / micro_step=400 / samples=12800，d_model=1024、24 层、
   MoE 8 专家、4 领域；sha256 前缀 `38a4104c…`，run_state 中有完整握手记录）。
2. **生成预测**：`model_design/launches/eval16-20261027/fast_eval.py`，
   `--second-split test --single-seed --train-cap 50000`，单种子 MLP（48 隐层，≤100K 参数）。
3. **合并分片**：`merge_and_score.py` 校验各分片的 checkpoint sha / 模型配置 /
   单种子 / manifest（board=A, split=test, heads={}）一致后硬链接合并 90 个任务预测，
   缺任务直接报错退出。
4. **官方计分**：用官方 `data/muse-a/eval/MUSE-eval/score.py` 生成 `官方提交/scores.json`。
5. **提交前核验**：`score.py check` + 机器断言（90/90、complete=true、check 无问题）。

合规要点已内建于工具链：换 checkpoint/配置/split 必须换输出目录；缓存握手
（`run_state.json`）；合并脚本拒绝多种子、拒绝非官方任务预测、拒绝 checkpoint 不一致的分片。

## 2. 实际执行现状：分片策略 + 当前进度

单卡跑全量 OOM（见 `official_a_test.log` 尾部 `torch.OutOfMemoryError`，GPU5 打满），
因此实际执行改成了**多目录分片跑 + 最后合并**的路线：

| 分片目录 | 覆盖 | 预测数 | 状态 |
|---|---|---|---|
| `official_a_test`（主 run） | 10 个数据集 | 65 | OOM 中断，partial_scores 41/90 |
| `official_a_test_remaining` | PEER | 1 | 中断 |
| `official_a_progress_report` | **全部 13 个数据集** | 84 | 进度快照，无 scores |
| `official_a_tail_16` | CATH / Polymer-Genome / QM9 | 11 | FAST-DONE（final 8.70，仅 11 任务） |
| `official_a_bindingdb(_fixed)` | PEER/bindingdb | 1 | 修复 bindingdb 任务 |
| `official_a_peer_16` / `node0_final` / `node1_final` | PEER | 各 2–3 | 跨节点 PEER 分片 |
| `official_a_after_bindingdb` / `official_a_peer_node0_tail` | — | 0 | 空壳/中断 |

**跨分片去重后已覆盖 85/90 个任务**，分片间 checkpoint sha、单种子、manifest
均为同一握手（已用 dashboard 官方任务清单逐一核对）。

**尚缺 5 个 PEER 子任务**（都在远端节点 172.17.0.178 上）：

- `PEER/solubility`
- `PEER/stability`
- `PEER/subcellular_localization`
- `PEER/thermostability`
- `PEER/yeast_ppi`

本机有一个后台看护脚本（今早 08:14 启动）在轮询等待 node1 的 solubility 预测落盘，
然后 ssh 到 node1 启动 `official_a_peer_node1_tail` 跑这最后几个子集。
截至梳理时 `official_a_peer_node1_tail` 目录尚未生成，**5 个任务仍在排队**。

## 3. 提交物现状

- `官方提交/` 目录目前**只有 SOP 和赛题网页，没有 `scores.json`、没有 `check.json`**。
- 即：完整 A 榜 test 的 90 任务官方得分文件**尚未生成、尚未提交**。
- 历史 partial 分数参考（均非完整提交物）：
  - `official_a_test/partial_scores.json`：41/90，final 6.06
  - `official_a_tail_16/official_scores.json`：11/90，final 8.70
  - `official_a_bindingdb_fixed/official_scores.json`：1/90（修复 bindingdb 用）

## 4. 剩余步骤清单（按 SOP）

1. 等 node1 跑完最后 5 个 PEER 子任务（看护脚本会自动拉起 tail 分片，需确认）。
2. 跑 `merge_and_score.py`（shards-root = `runs/s3-residual-pilot-02`，output 用新空目录），
   它会强制校验 90/90 + complete 后才产出 `official_scores.json`。
3. 用官方 `score.py tasks` + `score.py score` 重新生成 `官方提交/scores.json`。
4. `score.py check` + SOP 第 6 节全部断言。
5. 上传赛事平台（每日 18:00 前最后一次提交为准）。

## 5. 风险与注意点

- **不要直接拿任何一个现有目录当提交物**：只有合并后的 90/90 目录才合规；
  partial_scores 均标 `complete: false`。
- `official_a_progress_report` 虽有 84 个预测且覆盖全部 13 个数据集，但与其他分片
  存在任务重叠，合并脚本按目录去重，需留意同名文件以哪个分片为准（脚本按
  排序后的分片顺序取第一个）。
- 远端 node1 的任务是本链路唯一外部依赖；若 ssh 或远端环境出问题，需要手动
  在本机重跑这 5 个子集（可 `--subsets PEER/solubility,...` 指定）。
- 每次换 checkpoint 必须同步换 `CKPT/RUN/NAME` 并新建输出目录，避免审计混淆。

## 6. 工程侧代码结构

A 榜链路的代码分两套：**`model_design/muse_encoder/`（自研 Encoder + 冻结评估主链路，当前提交走这里）**
和 **`muse/`（早期的外包编码器基线管线，仅供对照）**。

### 6.1 主链路：`model_design/muse_encoder/`（约 16k 行）

**模型本体**

| 文件 | 职责 |
|---|---|
| `model.py` | `MuseEncoder`：四域前端 + 共享 Transformer 主干 + 统一 D 读出 + 预训练头；含 `PairBiasEncoder`（对偏置编码 3D 信息）、`PretrainHeads` |
| `layers.py` | RoPE、`PairBiasAttention`、`MoleFFN`（MoE 8 专家，自定义 autograd function）、`Block` |
| `featurize.py` | `RecordFeaturizer` / `FeaturizedRecord`：把四种模态（SMILES / 3D 构象 / CIF 晶体 / 蛋白序列）统一特征化 |
| `tokenizers.py` / `masking.py` / `structures.py` / `peptide.py` | 分词、掩码策略、结构解析、肽序列处理 |
| `inference.py` | 冻结编码推理接口（B 榜推理包的基础） |
| `config.py` | `MuseConfig`（d_model=1024、24 层等，与 run_state 握手） |

**冻结评估链路（A 榜提交的核心，约 4.5k 行）**

| 文件 | 职责 |
|---|---|
| `pretrain/dashboard.py`（1562 行） | 主编排 `run_dashboard()`：读官方 board → 逐子集编码 → 训练冻结 MLP 探针（`FixedProbe`，`fit_probe`，单种子 `seeds=(0,)`，48 隐层 ≤100K 参数）→ 写预测 → 官方 scorer 打分。还含 `load_board`/任务清单模型、`normalize_record`、CIF/SDF/pickle 读取、外部结构解析器、run 级缓存（`encode_split_cached`，按 checkpoint_sha + config_sha 握手） |
| `pretrain/parallel_dashboard.py` | 多设备并行编码：三层缓存（normalize → prepared → encoded），缓存 key 由数据 sha、checkpoint sha、特征化/归一化版本（模块代码 digest）共同组成；`encode_records_parallel` 用多进程池把编码任务摊到多张卡 |
| `pretrain/a_train.py` | 冻结 MLP 在 A 榜 train 子集上的训练实现 |
| `pretrain/crystal_dataset.py` / `crystal_audit.py` / `crystal_references.py` | 晶体 CIF 数据集、周期性边界审计、参考文献指纹 |
| `pretrain/molecule3d.py` / `conformers.py` / `scaffolds.py` / `omc25.py` | 3D 构象、构象生成、scaffold、OMC25 外部语料 |

**预训练侧（checkpoint 的来源）**

`pretrain/run_pretrain.py`（DDP 入口）→ `trainer.py`（`Trainer` / `CheckpointManager`，含 rng 状态、确定性种子）→ `dataset.py` / `corpora.py` / `batching.py` / `sampling.py`；配置在 `model_design/configs/pretrain_s3_residual_pilot.yaml`（S3 残差 pilot），runs 落在 `model_design/runs/<name>/`（checkpoints + config.resolved.yaml + node*.log）。

### 6.2 A 榜提交入口：`model_design/launches/eval16-20261027/`

| 文件 | 职责 |
|---|---|
| `fast_eval.py`（132 行） | **正式提交入口**。monkey-patch `dashboard.encode_split_cached`（train 截断 50k、validation 换 test、多卡并行编码）+ SDF 容错读取 + scorer 换成 test split；最后把 manifest 标记为 split=test 并重新跑官方打分 |
| `shard_eval.py` | 分片评估包装：train 截断但 validation 全量；`_read_sdf_selected_tolerant` 让单条坏 SDF 不再拖垮整个 split（fast_eval 复用） |
| `merge_and_score.py`（126 行） | 分片合并 + 强制校验（checkpoint sha / 模型与探针配置 / 单种子 / manifest / 非官方预测拒绝），缺任务直接报错；产出 `official_scores.json` 并断言 90/90、complete=true |
| `launch_shards.py` / `shards.json` | 把 90 任务按成本切成 16 个分片（shards.json 记录每片的 train_raw / validation 规模） |
| `make_shards.py` / `make_fallback.py` | 分片清单与兜底清单生成 |
| `bench_*.py` / `profile_compute.py` / `test_chain*.py` | GPU/数据管线基准与全链路冒烟测试（含 test_chain 系列 log） |

### 6.3 缓存与审计设计

- **跨 run 持久缓存**：`data/pretrain/cache/frozen_eval/`（归一化/特征化/编码三层）和 `data/pretrain/cache/omc25/ase_lmdb`（外部结构）；命中条件 = checkpoint sha + 数据版本 sha + 特征化代码 digest，缺一不可（对应 SOP 第 2.6 条缓存握手）。
- **run 内缓存**：`$RUN/embeddings/<dataset>/<subset>/<split>.{embeddings.npy,tokens.npy,ids.json,encoding.json}`，含 checkpoint_sha、config_sha、失败记录与耗时审计。
- **握手文件**：`$RUN/run_state.json`（dashboard 状态 v1，记录 checkpoint 路径+sha+训练步数、model_config、probe_config、数据 manifest sha），是合并脚本与复现审查的依据。
- **产物结构**：`$RUN/predictions/`（90 个 `test.csv.gz` + manifest.json）、`task_cache/`（每个任务的冻结 MLP 训练记录）、`official_scores.json` / `results.json`。

### 6.4 历史基线：`muse/`（早期对照管线）

`run.py`（convert → encode → train → score 驱动，结果追加 `scoreboard.csv`）+ `encoders/`（composition / copolymer / electrolyte / `esm2_seq.py` / `unimol2_mol.py` 外包编码器）+ `pretrain/`（数据准备：dedup / deleak / 描述子 / 指纹）。`results/v0-*` 是 v0 checkpoint 的对照实验。**A 榜正式提交不走这套**，正式链路完全在 `model_design/` 内闭环。
