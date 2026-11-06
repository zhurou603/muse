# 官方流程提交 SOP

## 1. 适用范围

本流程用于生成**可直接提交的官方 A 榜得分文件**。正式提交物是官方 `score.py` 对完整 A 榜 test 预测生成的 `scores.json`，不是 validation 分数、晶体单任务分数或手工整理的 JSON。

依据：

- 赛题网页：`官方提交/muse.htm`
- 官方评分说明：`data/muse-a/eval/MUSE-eval/README.md`
- 官方评分程序：`data/muse-a/eval/MUSE-eval/score.py`

A 榜覆盖 13 个数据集、78 个子集、90 个子任务。最终文件必须满足：

- `evaluation.board == "A"`
- `evaluation.split == "test"`
- `num_tasks == 90`
- `scored_tasks == 90`
- `complete == true`
- `missing_predictions == []`
- `invalid_predictions == []`
- 官方 `score.py check` 返回 `consistent: true` 且 `problems: []`

每队每日最多一次 A 榜正式提交，以当日 18:00 前最后一次提交为准；不需要每天重复提交。

## 2. 合规边界

1. 全部任务使用同一个冻结 Encoder，所有领域输出相同的表征维度 D。
2. Encoder 不得在任何下游任务上继续训练；每个任务只训练独立轻量 MLP。
3. 正式流程使用单种子 `Linear(D,48) -> ReLU -> Linear(48,K)`，最多两个 Linear，参数量不超过 100K。不要直接把多个 48 隐层的种子集成用于提交，因为合并后可能超过参数上限。
4. MLP 只使用对应子集的 A 榜 train 标签训练。validation/test 标签只用于评估，不得用于训练、早停、校准或选择权重。
5. test 预测必须来自同一个 checkpoint；不得混合不同 Encoder 的预测或表征。
6. 本地缓存只能复用输入解析、特征化和与 checkpoint 完全匹配的表征。换 checkpoint、输入数据、模型配置或特征化代码后必须按缓存握手重新计算。
7. 得分文件必须由官方 `score.py` 生成，不得手工修改。

## 3. 固定路径与环境

在项目根目录执行：

```bash
cd /yinghepool/tech-dcu/zhaoyao
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/yinghepool/tech-dcu/zhaoyao/model_design:/yinghepool/tech-dcu/zhaoyao
export OMP_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python
DATA=/yinghepool/tech-dcu/zhaoyao/data/muse-a/MUSE-A/data
EVAL=/yinghepool/tech-dcu/zhaoyao/data/muse-a/eval/MUSE-eval
```

本次示例：

```bash
CKPT=model_design/runs/s3-residual-pilot-02/checkpoints/final.pt
RUN=model_design/runs/s3-residual-pilot-02/official_a_test
NAME=s3-residual-pilot-02
```

更换 checkpoint 时同时更换 `CKPT`、`RUN` 和 `NAME`，避免不同实验的审计产物混淆。

## 4. 生成完整 A 榜 test 预测

以下命令默认遍历全部 A 榜子集，不设置 `--subsets`。每个子集最多读取 50,000 条 train 记录；少于该数量的子集使用全部 train。test 完整评估，不截断。

```bash
$PY model_design/launches/eval16-20261027/fast_eval.py \
  --checkpoint "$CKPT" \
  --output-dir "$RUN" \
  --cache-root data/pretrain/cache/frozen_eval \
  --external-cache data/pretrain/cache/omc25 \
  --device cuda:0 \
  --devices 1,2,3,4,5,6,7 \
  --cpu-workers 14 \
  --batch-size 64 \
  --train-cap 50000 \
  --second-split test \
  --single-seed \
  --team zhaoyao \
  --name "$NAME" \
  > "$RUN.log" 2>&1
```

说明：

- `--single-seed` 是正式提交流程的固定要求，不用于改变 Encoder。
- GPU 0 保留主进程，GPU 1–7 执行编码，避免同一张卡叠加主模型和长序列 worker 的显存高水位。
- `--second-split test` 使第二个编码切分从 validation 改为 test，并将预测 manifest 标记为 test。
- 省略 `--subsets` 才能形成完整 A 榜；指定它只用于调试，不能作为完整提交。
- 输出目录已有不同 checkpoint、配置或 split 的 `run_state.json` 时，必须换一个新目录。
- 同一运行中断后可使用相同命令和目录恢复；预处理缓存位于 `data/pretrain/cache/frozen_eval/`。缓存命中不代表跳过赛事检查，最终仍须重新执行官方计分与核验。

### 4.1 两节点 16 卡互斥分片（本次已验证）

16 卡交付时不要重复启动同一子集。先按任务规模拆分，并在每个输出目录完成后再严格合并。本次有效分片为：

| 分片 | 子任务数 | 说明 |
|---|---:|---|
| `official_a_test` | 65 | 主体任务 |
| `official_a_test_remaining` | 1 | `binary_localization` |
| `official_a_bindingdb_fixed` | 1 | 修复后的 `bindingdb`；旧 `official_a_bindingdb` 禁止使用 |
| `official_a_peer_16` | 3 | `fluorescence`、`fold`、`gb1` |
| `official_a_tail_16` | 11 | CATH、QM9、Polymer-Genome |
| `official_a_peer_node0_final` | 2 | `human_ppi`、`pdbbind` |
| `official_a_peer_node1_final` | 3 | `ppi_affinity`、`secondary_structure`、`solubility` |
| `official_a_peer_node1_tail` | 2 | `thermostability`、`yeast_ppi` |
| `official_a_peer_node0_stability` | 1 | `stability` |
| `official_a_peer_node1_subcell` | 1 | `subcellular_localization` |

单个输出目录因会话退出成为孤儿后，先以真实进程、GPU 和预测文件确认状态；若中断且没有 prediction，换新分片目录恢复，不能继续向可能不一致的旧状态写入。node1 在 8 个 worker 同时加载模型时出现过 HIP 显存竞争，已验证 `devices 0,1,2,3` 是稳定配置。

合并前建立只包含有效分片的 `official_a_shards/`，绝不纳入旧 `official_a_bindingdb`。若某些分片缺少 `predictions/manifest.json`，在确认其 `run_state.json`、checkpoint SHA256、单种子配置和 prediction 文件均正确后，按正式 manifest 补写 `team/name/board=A/split=test/heads={}/smoke=false/limit_per_split=null`。

```bash
$PY model_design/launches/eval16-20261027/merge_and_score.py \
  --shards-root model_design/runs/s3-residual-pilot-02/official_a_shards \
  --output-dir model_design/runs/s3-residual-pilot-02/official_a_merged \
  --split test \
  --team zhaoyao \
  --name s3-residual-pilot-02 \
  --train-cap 50000
```

合并器会强制检查同 checkpoint、同模型/MLP 配置、单种子、每个官方任务恰好一个来源、无重复、无缺失、manifest 合法以及 `scored_tasks == 90`。合并通过后，第 5 步使用 `official_a_merged/predictions`。

主要中间产物：

```text
$RUN/
├── predictions/           90 个子任务的 test 预测与 manifest.json
├── embeddings/            本子集、本 checkpoint 的表征与审计信息
├── task_cache/            冻结 MLP 训练记录
├── run_state.json         checkpoint、配置和数据版本握手
├── official_scores.json   由同一官方评分模块生成的分数
└── results.json           完整运行结果
```

## 5. 用官方 score.py 生成最终提交文件

即使内部流程已经生成 `official_scores.json`，正式提交仍统一通过官方命令行再生成一次：

```bash
mkdir -p 官方提交

$PY "$EVAL/score.py" tasks \
  --data-root "$DATA" \
  --board A \
  --out "$RUN/tasks.json"

$PY "$EVAL/score.py" score \
  --predictions model_design/runs/s3-residual-pilot-02/official_a_merged/predictions \
  --data-root "$DATA" \
  --board A \
  --split test \
  --out 官方提交/scores.json
```

`官方提交/scores.json` 才是上传赛事平台的文件。不要修改其内容、队伍名、提交名、分数或哈希。

## 6. 提交前强制核验

```bash
$PY "$EVAL/score.py" check \
  --scores 官方提交/scores.json \
  --data-root "$DATA" \
  --board A \
  > 官方提交/check.json
```

再检查机器可读结果：

```bash
RUN="$RUN" $PY - <<'PY'
import json
import os
from pathlib import Path
scores = json.loads(Path('官方提交/scores.json').read_text())
check = json.loads(Path('官方提交/check.json').read_text())
assert scores['schema_version'] == 'muse.scores.v1'
assert scores['evaluation']['board'] == 'A'
assert scores['evaluation']['split'] == 'test'
assert scores['num_domains'] == 4
assert scores['num_datasets'] == 13
assert scores['num_tasks'] == scores['scored_tasks'] == 90
assert scores['complete'] is True
assert scores['missing_predictions'] == []
assert scores['invalid_predictions'] == []
assert check == {'consistent': True, 'problems': []}
manifest = json.loads((Path(os.environ['RUN']) / 'predictions/manifest.json').read_text())
assert manifest['board'] == 'A' and manifest['split'] == 'test'
assert not manifest.get('heads')
print({'final_score': scores['final_score'], 'complete': scores['complete'],
       'scored_tasks': scores['scored_tasks'], 'check': check})
PY
```

只有上述断言全部通过才上传。若任何任务缺失或非法，应回到 `$RUN/results.json`、对应 `task_cache/` 和运行日志定位原因，修复预测后重新执行第 5、6 步；不得手工把 `complete` 改为 `true`。

## 7. 验证集与测试集的区别

- `--second-split validation`：用于开发期模型诊断，不能作为正式 A 榜提交。
- `--second-split test`：用于正式 A 榜提交，最终得分文件必须显示 `split=test`。
- 晶体单任务的 `official_scores.json` 即使来自官方 scorer，也因缺少其他 89 个任务而不是完整 A 榜提交物。
- B 榜需要另行准备 `inference_package/`、锁定 Encoder、MLP heads 和训练复现材料；本 SOP 不替代 B 榜流程。
