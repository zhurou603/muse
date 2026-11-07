# S5 六机续训交接：47 rank + 独立 A-test

更新时间：2026-11-07 00:01（Asia/Shanghai）。

## 0. 当前真实状态

- **正式 47-rank 长训已启动。** session：`train-start_1793968162699724949`；正式 orchestrator PID `3540148`，验收/长训管线 PID `3332826`，独立后台运行，不依赖应用连接。
- 20+1 步及恢复验收通过：`acceptance.json` 为 `smoke_and_resume_pass`，47 ranks、无新跳过、两代各47份 RNG。正式段从固定 17500 源点开始，不混入 smoke 权重。
- **首个正式 checkpoint 已核验**：step 17600，SHA256 `ad1b1dd46fe9dbc3e7d3f362f77d4a38b5b2728297023611a19dc8d76a187f1e`，47 份 RNG、optimizer/scheduler、有限参数和血缘均通过。报告 `control/formal_checkpoint_17600_verified.json`。
- 20:40 健康为 **47/47、课程一致、无 issues**。截至首个 checkpoint 检查时有 8 条新增异常数据（6 GEOM、2 QM9）已记录并换样，训练未退出。实时状态以 `runs/main/health/sessions/` 和 `runs/main/latest.json` 为准。
- 171 的**物理第 7 卡专用于 A-test**，训练只可见物理 0–6 卡。其余五机各用 8 卡，训练全局 batch=376。
- step16900+v3 A-test **已完成90/90**，得分 **58.08142791678441**，complete=true、missing/invalid=[]、官方check一致。相比step10800+v2的59.2906418076102 **下降1.20921389082579**；聚合物提高1.3147，其他三域下降。156个failed.json全部为空，未上传或覆盖正式提交。产物 `official_a_test_step16900_repair_v3/`。
- **step26700+47rank A-test 已完成**：**59.778077940648124**，90/90、complete=true、missing/invalid=[]、官方check一致；156个failed.json全部为空，进程已结束。相比16900提高1.696650023863711，相比10800提高0.48743613303792444，是当前本地已知最高完整A-test分数。未上传或覆盖正式提交，未停止训练或改变参数。产物 `official_a_test_step26700_47rank_v3/`。
- 六机 48-rank 小张量 RoCE 通信已通过；它不替代真实 S5 验收。首次导入失败和 r2 拓扑回溯停滞均保留现场，当前 r3 包含完整依赖和搜索上限。
- 长时稳定性、后续 checkpoint 持续写入和赛事提分仍需继续验证；不得仅凭当前 healthy 宣称已稳定运行10小时或已提分。

## 1. 当前候选与血缘

候选根目录：

```text
/yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_step17500_47rank_r3_20261106
```

- 源 checkpoint：`model_design/runs/s5-continuous-four-domain-v1/main/checkpoints/step_00017500.pt`，SHA256 `6e7d1e7d417c56271a6ff2e50044bab19d16f1f7627e48fde0c5c2c1d5fcff2d`。
- 固定副本：`handoff/source.pt`，不再追随旧 latest。
- `source_state.step=micro_step=17500`，原 15 份 RNG 完整；17500 之后未保存的更新不纳入新段。
- 生产源码 commit：`f588216229d43e0cdc0dd18511deb0f4a3c83461`。
- 工具快照 commit：`42275b7c4a666849e7674960830e2e18a3daf666`。该提交只改变管线/测试，未改训练 Python；已完成 smoke 后的更新、47份 RNG 和全快照复核记录为 `control/pipeline_patch_42275b7c4a666849e7674960830e2e18a3daf666.json`。
- 数据清单 SHA256：`fafe5a6b0f97bbc35e64b5af3585627a2fcab5e4aedcff1e7c82af374c396ee8`。
- 继承的坏样本清单 SHA256：`37bb9f53edefbc7a65f5d7144e8a233d6cd92e8e6725edefb0839a47d976082b`，含旧 run 全部记录和 r2 新增两条非法 GEOM；追加前缀已验证。
- 固定课程、模型结构和 `config.yaml`；不加载第三方权重，不使用 validation/test 标签训练或归一化 Encoder，不混用 S4 或其他父 checkpoint。
- `tools/s5_six_node_step17500_20261106` 是 48 卡工具与通信审计；`tools/s5_six_node_step17500_47rank_20261106` 是首次导入失败现场；`tools/s5_six_node_step17500_47rank_r2_20261106` 是拓扑回溯停滞现场。均不混入当前候选。

## 2. 拓扑与超参数

| 节点 | 训练物理卡 | global ranks |
|---|---|---|
| 172.17.2.17 | 0–7 | 0–7 |
| 172.17.2.184 | 0–7 | 8–15 |
| 172.17.2.46 | 0–7 | 16–23 |
| 172.17.1.84 | 0–7 | 24–31 |
| 172.17.0.171 | **0–6；第7卡给 A-test** | 32–38 |
| 172.17.0.178 | 0–7 | 39–46 |

| 项目 | 原 15 rank | 当前 47 rank |
|---|---:|---:|
| 每 rank micro batch / accumulation | 8 / 1 | 8 / 1 |
| 名义全局 batch | 120 | **376** |
| existing / new_aux_heads LR | 1e-5 / 1e-4 | 完整恢复，不按卡数放大 |
| weight decay | 0.01 | 0.01 |
| scheduler | last_epoch=17500 | 恢复，不重新 warmup |
| BF16 / grad clip | true / 1.0 | 不变 |
| 每 rank prefetch worker | 4 | 1，降低共享存储压力 |
| checkpoint / log interval | 100 / 20 step | 不变 |
| 最大训练 token/残基裁剪 | 1024 | 保留已验收的训练裁剪，推理不截断 |

不缩小每卡 batch 的原因是保持本卡 InfoNCE 负样本数。全局 batch 增大约 3.13 倍会改变梯度统计，不能承诺与旧轨迹或分数等价。当前 LR 较小，扩卡已是独立变量，故不做缺少验证依据的线性/开方放大。后续仍按固定 robust validation、90/90、shadow-B、coverage、遗忘门和血缘选模，不能按 latest 或预训练 loss 选模。

相同样本预算下，固定绝对 step 会改变数据暴露量；采样有放回，不能把步数换算成 epoch 百分比。`samples` 是 rank0 历史展开行数，不是全局去重覆盖率。四域 20-step 课程和 DBAASP 微课程不变。

## 3. 已完成验证与边界

- 真实 17500 checkpoint CPU 恢复：模型、AdamW、两组 LR、scheduler、计数器逐张量一致，参数有限。
- 15→47 扩展恢复：新增 rank 按 `(seed,step,rank)` 重播种子；47→47 严格读取本 rank RNG，下一步确定性更新一致。
- 原 S5 回归 40/40；候选控制/拓扑测试 8/8；管线门测试 3/3。
- **慢样本根因与修复**：r2 在 step17502、rank41、GEOM aux row 27567566 上停滞，worker 持续约99% CPU。`_topology_mapping` 旧代码只限制完整匹配，不限制部分回溯状态。现对全部搜索状态限制 10000 节点，超限不伪造映射。该记录随后通过官方构象图验证（同分子式/连通性、直接重原子映射、有限坐标、非2D降级），CPU重放约2秒，合法输入不误删；无合法表示时按数据异常记录补采。证据 `tools/s5_six_node_step17500_47rank_r2_20261106/rank41_bounded_prefetch.json`。
- structures/featurize/dashboard 回归 60/60；真实17500 CPU恢复在r3再次通过。
- 真实双进程 Gloo 协调停止：同一优化步边界保存 final 和完整 RNG。
- 六机 `muse.tasks` 与训练模块真实导入通过；47 个 rank 心跳文件已生成，NCCL 初始化显示 `nranks=47`。
- **已完成**：真实20步、47份RNG保存、47-rank恢复1步、20+1验收、正式长训启动及首个正式checkpoint核验。管线曾因完成状态缺少`mode`字段而停止放行，已改为验证不可变session，3/3测试通过；只修编排，未改训练模型。
- **持续边界**：20步未覆盖全部DBAASP 200-step微课程，也不证明10小时稳定或赛事提分。`control/first_healthy.json` 是 smoke 的历史标志，正式健康必须读 `runs/main/health/sessions/train-start_1793968162699724949.summary.json`，不要误用旧标志。

## 4. 源码一致性与运行持久化

- 训练 PYTHONPATH 为候选内 `model_design` 与候选根目录；`muse_encoder`、`muse`、启动器均逐文件 hash 校验，不读取可随时变化的生产 Python 包。
- 数据仍通过固定绝对路径和 manifest 使用，不复制或改写原始数据。
- rank 间核对 Python、PyTorch/HIP、RDKit、NumPy、YAML、GPU 型号、session、配置、断点和 batch；171 要求 7 个可见设备，其他节点 8 个。
- 训练与节点 driver 使用独立 session；应用退出后由 PID 1 接管。节点断电、存储故障等仍依赖 checkpoint 恢复，不保证零中断。
- 四台新节点已为 `zhaoyao` 配置公钥；不需要在训练进程中保存密码。任务 known_hosts 仅包含六台公开主机指纹。

## 5. 试跑与正式续训命令

```bash
C=/yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_step17500_47rank_r3_20261106
PY=/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python
TOOL=$C/tools/s5_six_node_candidate/cluster.py
export S5_KNOWN_HOSTS=/yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_runtime_20261106/access/known_hosts
```

资源检查（保留 171 第7卡，只检查训练卡显存）：

```bash
$PY -B $TOOL probe --candidate $C --require-free-mib 55000
```

20 步试跑及恢复验收均已完成，正式段正在运行，**不要重复执行 smoke-start、smoke-resume 或 train-start**。`control/pipeline_state.json` 为 `formal_training`，当前 orchestrator PID `3540148`、管线 PID `3332826`。管线日志为 `logs/pipeline_orchestrator.log`，首段日志为 `logs/smoke_start_orchestrator.log`。

管线已执行并通过（仅供审计，不要与管线并行执行）：

```bash
$PY -B $TOOL run --candidate $C --mode smoke-resume --stop-after 17521 --max-attempts 1
$PY -B $C/tools/s5_six_node_candidate/check_acceptance.py --candidate $C
```

验收已确认：47/47 rank 的 host/device/session 正确、两个 session 完成、20+1 步、两代 checkpoint 各 47 份 RNG、optimizer/scheduler 正确、loss/grad 有限、无新坏样本。出现新跳过时先审查 `smoke_review.json`，不能忽略。20 步覆盖课程槽位，但不覆盖全部 DBAASP 200-step 微课程。

正式段由管线启动，实际审计命令如下，**当前不要重复执行**：

```bash
setsid $PY -B $TOOL run --candidate $C --mode train-start --stop-after 1000000 \
  --max-attempts 20 > $C/logs/formal_orchestrator.log 2>&1 < /dev/null &
```

正式 main 从固定的 17500 源点恢复，**不混入 smoke 的 20+1 步权重**。正式启动的 checkpoint/健康证据分别为 `control/formal_checkpoint_17600_verified.json` 和 `runs/main/health/sessions/train-start_1793968162699724949.summary.json`。

## 6. 稳定性、停止与恢复

- 心跳 15 秒，stale 阈值 60 秒；无进度 900 秒，启动宽限 1800 秒，checkpoint 宽限 3600 秒；NCCL/rendezvous timeout 900 秒。
- 健康按 session 隔离，旧日志不掩盖新启动失败。汇总 47 rank 进度、课程一致性、有限 loss/grad、checkpoint hash、RNG 和六机 GPU。
- checkpoint 使用临时文件、flush/fsync、原子 rename 和目录 fsync；全部 47 份 RNG barrier 后才发布 latest。
- 普通 dataset/metadata 错误记录并补采，不丢弃整个DDP step；单条最多替换256次，日志跨进程加锁并fsync。排除在当前worker立即生效、所有rank在恢复时加载，**不是同session全局实时去重**。OOM、NCCL、I/O、非有限loss/grad、源码/hash/配置错误不被当作坏数据。
- 正式段最多 20 次 attempt，退避 30–300 秒；确定性错误停止待查。没有新 checkpoint 时保留失败现场，只从同一 17500 源点重建，不混用其他父模型。
- 应用退出不影响已启动进程；异常节点故障最多丢失未保存尾部更新，latest 不是最佳模型。

查看：

```bash
$PY -B $TOOL status --candidate $C
```

健康：`runs/<smoke或main>/health/sessions/<session>.summary.json`。最近完整断点：`runs/main/latest.json`。驱动日志：`logs/drivers/`；逐 rank 日志：`logs/<session>/`。

正常停止，不直接杀 orchestrator：

```bash
$PY -B $TOOL stop --candidate $C --grace-seconds 900 --reason 'operator planned stop'
```

请求会在所有 rank 的同一优化步边界生效，保存 final 和 47 份 RNG，写 `runs/main/control/stop_ack.json`。审阅 ack 后，使用 `run --mode train-resume --stop-after 1000000 --acknowledge-stop` 恢复。

## 7. 评估与回退边界

- A-test 保留在 171 物理第 7 卡，训练启动器不会向它分配训练 rank，也不会自动启动另一个 A-test。
- 不上传赛事平台、不覆盖 `官方提交/` 或 best_fixed。
- 原 15-rank run 和 17500 checkpoint 保留，可按原两机方式回退。47-rank optimizer/RNG 不得交给未适配的旧启动器。
- 失败试跑、旧 48 卡通信和首次导入失败现场均保留，不能删除历史或用新 session 掩盖旧错误。
