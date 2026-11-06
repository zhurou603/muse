# S5 四领域连续训练完整交付

这是一个安装在既有 `muse_encoder` 项目上的增量工程，不是独立可运行的替代仓库。目标是在最后四天内持续训练同一个、完全自研且从头训练的 Encoder，定期保存可恢复 checkpoint，用固定 validation 协议筛选候选，并在任何候选退化时保留父模型和历史最佳。

**先读结论：**本工程不能保证 A/B 榜必然收益，也不能保证四个领域每天严格单调上涨。它能保证的是数据来源与标签边界可审计、课程确定、训练可续跑、checkpoint 有血缘、评估只读、候选失败不覆盖最佳、正式提交与高频选模分离。晶体已接近满分，应优先守住；小分子、聚合物和生物按官方加权总分与遗忘门共同筛选。

## 1. 使用前必须满足的条件

1. 原项目是已审查的 S1–S3 代码基线，存在 `code/model_design/muse_encoder/model.py`。
2. 最终 Encoder 的全部参数均由团队从随机初始化训练；父 checkpoint 必须来自自己的 S1–S4 训练轨迹。
3. 不加载 Uni-Mol2、ESMC、ESM-2、UMA、eSEN、EquiformerV2、polyBERT、CheMeleon 等第三方预训练权重。
4. 官方 train、官方明确允许的 aux 和既有外部无标签语料已经按原项目流程准备好。外部语料的 release、许可证、原始 SHA256 和下载日期仍需在原仓库补证。
5. validation/test 标签不参与训练、归一化或选模；validation/test 输入只用于去泄漏指纹。A 榜 test 不作为反复调参工具。
6. 不在已有正式 run 目录上直接启动新实验，不覆盖现有最佳模型或正式提交文件。

数据内容、记录数、清洗、去泄漏、描述符、索引和单位处理见 **[训练数据与预处理](docs/训练数据与预处理.md)**。逐日操作见 **[四天执行手册](docs/四天执行.md)**，已运行与未运行项目见 **[验证记录](docs/验证记录.md)**。

## 2. 交付内容

| 路径 | 作用 |
|---|---|
| `config/s5.yaml` | 24 层、1024D、16 头、MoLE 模型配置；数据路径、20-step 课程、训练与评估门槛 |
| `config/external_provenance.yaml` | 五类外部语料的血缘补证模板；未知项必须 fail closed |
| `s5/curriculum.py` | 四领域 6/5/6/3 槽位和 DBAASP 精确微课程 |
| `s5/data.py` | 官方 train、aux、外部 replay、OMC25 的统一装配与单任务过滤 |
| `s5/robust_targets.py` | 记录内 median、train-only median/MAD、std/1.0 回退 |
| `s5/long_sequence.py` | 训练侧确定性 token 裁剪，不污染推理完整序列 |
| `s5/prepare_data.py` | 目标统计、单进程索引和数据 SHA256 清单 |
| `s5/train.py`、`s5/runtime.py` | 严格初始化、warmup-stable LR、DDP、checkpoint、RNG 和 resume 血缘 |
| `s5/robust_dashboard.py` | 固定两层 MLP、固定种子、90 项 validation 和 shadow-B |
| `s5/evaluate_candidate.py` | 冻结副本、完整性与遗忘门、best_fixed 接受/拒绝 |
| `s5/watch_evaluation.py` | 独立异步评估；评估失败不杀训练，不自动提交 |
| `scripts/` | 官方下载、安全解包、安装、训练、评估和失败记录聚合 |
| `tests/` | 课程、目标、损失隔离、初始化、恢复、评估门控和长序列测试 |

## 3. 安装

以下命令中的两个路径都必须替换为真实绝对路径：

```bash
export S5_PACKAGE=/path/to/S5完整连续训练交付
export REPO_ROOT=/path/to/项目根目录

bash "$S5_PACKAGE/scripts/install.sh" "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/model_design:$S5_PACKAGE:${PYTHONPATH:-}"
```

安装器会：

- 将模块复制到 `code/model_design/muse_encoder/s5`；已有目录则拒绝覆盖。
- 将配置复制为 `code/model_design/configs/s5_continuous.yaml`；已有配置则拒绝覆盖。
- 备份 `featurize.py.pre_s5`，并且只在原文件 SHA256 等于已审查基线时应用长序列补丁。
- 安装外部血缘模板，但不伪造未知 release、许可证或 hash。

如果 `featurize.py` 已被修改，不要强行打补丁。先人工比较 `patches/long_sequence.patch`，把“训练裁剪、推理完整分窗”的两个边界合并到当前版本，再重新做集成测试。

## 4. 第一次启动前的验收

### 4.1 静态与单元测试

```bash
python3 -m compileall -q \
  "$REPO_ROOT/code/model_design/muse_encoder/s5" "$S5_PACKAGE/scripts"
python3 -m unittest discover -s "$S5_PACKAGE/tests" -p 'test_*.py' -v
bash -n "$S5_PACKAGE"/scripts/*.sh
```

### 4.2 长序列集成测试

```bash
python3 "$S5_PACKAGE/tests/integration_long_sequence.py"
```

交付包中的合成测试使用内置词表，以便在没有生产词表的审查快照中验证模型和窗口逻辑。原机还必须执行 `python3 "$S5_PACKAGE/tests/integration_long_sequence.py" --production-tokenizers`，对生产 `load_tokenizers()` 再跑一次；不要把合成词表结果当成生产词表验收。

### 4.3 数据清单与外部血缘

```bash
python3 -m muse_encoder.s5.prepare_data \
  --config "$REPO_ROOT/code/model_design/configs/s5_continuous.yaml"
```

如果 `s5_a_train_targets.json` 已存在且确认是当前数据生成，不要加 `--recompute-targets`。需要重算时先备份旧文件和日志。准备过程会顺序预建 A-train offsets/metadata，避免多个 DDP rank 同时写索引，并生成 `s5_data_manifest.json`。

```bash
python3 -m muse_encoder.s5.audit_featurization \
  --source pi1m --limit 10000 \
  --output "$REPO_ROOT/data/pretrain/audit/s5_pi1m_featurization.json" \
  --max-degraded-fraction 0.05

# 当前模板含“待核验”，预期返回非零；补齐真实证据后才应通过。
python3 -m muse_encoder.s5.audit_external \
  "$REPO_ROOT/data/pretrain/manifest/external_provenance.yaml"
```

训练启动时会复核数据清单：小文件和关键统计文件重新计算 SHA256，大文件默认核对字节数。需要全量复核时，在训练命令增加 `--full-hash`。字节数检查不是内容 hash 的等价替代，正式复现至少做一次全量复核。外部血缘文件也被纳入数据清单；补证或修改它之后必须重新运行 `prepare_data` 更新 hash，不能手工伪造清单。

## 5. 父模型、baseline 与新 run

父模型只能选择自己的合法训练结果：

1. 优先使用已经通过完整 90 项 validation、shadow-B 和四领域非退化验收的 S4 replay-only control。
2. 如果 control 尚未完成验收，使用已验证的 DBAASP step 300。
3. 不使用 S4 treatment，不做 checkpoint 权重平均，不把 `latest/final` 当作最佳模型。

为父模型建立一次固定协议 baseline。以下命令会跑完整 90 项 validation 和 8 项 shadow-B，不是 A-test：

```bash
export RUN_DIR=/data/runs/S5_$(date +%Y%m%d_%H%M%S)
export EVAL_ROOT=/data/eval/S5_fixed
export BEST_STATE=/data/eval/S5_fixed/best_fixed_state.json
export PARENT=/path/to/自己的父checkpoint.pt

mkdir -p "$RUN_DIR" "$EVAL_ROOT"
python3 -m muse_encoder.s5.evaluate_candidate \
  --config "$REPO_ROOT/code/model_design/configs/s5_continuous.yaml" \
  --checkpoint "$PARENT" --evaluation-root "$EVAL_ROOT/parent_bootstrap" \
  --mode full --device cuda:7 --state "$BEST_STATE" --bootstrap
```

`--bootstrap` 只能在 best_fixed 尚不存在时建立父模型基线，不能覆盖已有最佳。若评估机只有一张可见卡，用 `CUDA_VISIBLE_DEVICES=7` 配合 `--device cuda:0`，或直接把 `--device` 改为实际设备。

## 6. 启动连续训练与异步评估

默认配置为 7 卡训练、1 卡评估；先按执行手册完成 1-step、50-step 和 resume 小验收，再长跑。

```bash
# 终端/tmux 会话 1：训练。TARGET 是绝对 optimizer step，不是本段新增步数。
TRAIN_GPUS=0,1,2,3,4,5,6 NPROC=7 \
  bash "$S5_PACKAGE/scripts/start_training.sh" "$RUN_DIR" "$PARENT" 1000000

# 终端/tmux 会话 2：只读评估。
EVAL_GPU=7 bash "$S5_PACKAGE/scripts/start_evaluation.sh" \
  "$RUN_DIR" "$EVAL_ROOT/candidates" "$BEST_STATE"
```

训练脚本发现 `latest.json` 时自动 resume，否则从指定父模型初始化。失败按递增间隔重启，连续失败 20 次后停止并保留现场。配置文件在 resume 时必须逐字一致，只允许增大 `--stop-after`，避免通过续跑偷改课程或 LR 轨迹。

评估器只读取 checkpoint 并复制为带 SHA256 的冻结副本：

- 每 100 step 的 sentinel 只用于观察，结果始终标记 smoke，不接受为最佳，也不是完整总分。
- 每个任务只训练一个固定种子的 `Linear→ReLU→Linear` 探针，不做多种子预测集成；参数预算按单个完整头计算，分类输出只使用规定的 sigmoid/softmax。
- full 默认每 24 小时最多一次，必须 90/90、非 smoke、shadow-B 完整、Encoder 缺失/回退比例达标。
- 候选的 `J_encoder_fixed` 至少比当前 best 高 0.02，且四领域与 shadow-B 遗忘均不超门限，才更新 best_fixed。
- 被拒绝候选只写审计记录，不修改 best_fixed，不停止训练，不自动运行 A-test。

## 7. 20-step 课程与算法边界

每个 optimizer step 只选择一个来源；梯度累积内保持同一来源。所有 DDP rank 共享 source，仅 row/view 不同。一个 20-step 大周期为：

| 领域 | 步数 | 内容 |
|---|---:|---|
| 小分子 | 6 | 2 步量子 aux；1 步 GEOM 能量＋3D；1 步 QMugs/QM9 A-train；1 步 PubChem/ChEMBL MLM＋MTR；1 步 FreeSolv/TDC |
| 聚合物 | 5 | 3 步 PI1M MLM＋descriptor replay；1 步 Polymer Genome 两个带隙轮换；1 步 TransPolymer/Copolymer/PolyGraphMT 电子任务轮换 |
| 生物 | 6 | 2 步 DBAASP 微课程；1 步 MegaScale；1 步 PEER；1 步 CATH；1 步 UniRef/PDB 长短序列 replay |
| 有机晶体 | 3 | OMC25 元素残差能量＋力，source 权重 energy=10、force=30、stress=1 |

两个 DBAASP 槽位在十个大周期内精确实现 main/hemo/nonmic/short_uniref/short_pdb/long_pdb = 45%/10%/10%/15%/5%/15%。这不是“随机采样长期期望接近比例”。

已有参数 LR 为 `1e-5`，仅新增 aux head 为 `1e-4`；AdamW weight decay 0.01，warmup 50 步后恒定 LR，grad clip 1.0，BF16。不要在这四天内扩大模型、换 tokenizer、重做 S4 treatment 或频繁改课程。

## 8. 正式 A 榜、交付和止损

- 高频选模只比较同一 validation、同一 robust probe、同一固定单种子和完整 90 项的 `J_encoder_fixed`。旧多种子集成看板与 S5 单头协议不是同一口径，不直接比较数值。
- 每天最多一次，仅对当前已接受 best_fixed 的冻结 Encoder 跑既有正式 A-test/提交流程。validation 分数与 A-test 分数不能直接当成同一数字比较。
- A-test 只用于每日一次交付估计，不据此反复改训练目标。B 榜阶段不得更新 Encoder；train/validation/test 表征必须来自同一个锁定推理包。
- 遇到数据清单/hash 失败、非有限 loss/梯度、非法来源、生产特征化异常或持续领域退化，先停止自动重启并保留现场。任何新候选都不能覆盖父模型或 best_fixed。
- checkpoint 包含模型和 AdamW 状态，386M 参数下单个文件可能达到数 GB；100-step 周期存档会持续占用磁盘。长跑前按实测文件大小和步速做容量预算，定期把 `checkpoints.jsonl`、course 日志、评估结果与最佳冻结副本备份到独立存储。不要在未核对血缘前批量删除 checkpoint。

历史 23479 条失败/回退记录需要单独聚合：

```bash
python3 "$S5_PACKAGE/scripts/aggregate_failures.py" \
  /path/to/历史日志目录 \
  --output "$REPO_ROOT/data/pretrain/audit/s5_failure_groups.json"
```

该输出是记录级原因统计，不等于 90/90 缺失，也不等于 Encoder coverage；三者不得混写。
