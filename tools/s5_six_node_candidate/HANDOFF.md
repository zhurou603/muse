# S5 六机 48 卡长训交接

更新时间：2026-11-06 18:06（Asia/Shanghai）。本版取代此前的 16900 候选说明。

Git 主线合并提交：`ab29bf8c50f513b4f4ff976a40cb7c6b71af7645`；最终本地回归为 S5 40/40、候选控制 7/7，工作区干净。

## 0. 当前结论（先看这里）

**旧两机 15-rank 长训已停止，最终完整断点为 step 17500。六机源码、恢复和后台管理工具已准备，并已通过真实 checkpoint 的 CPU 完整恢复及真实双进程停止测试。四台新节点的 SSH 认证尚未就绪，六机 48 卡通信、20+1 步真实验收和正式长训尚未启动。不得将本文件理解为“48 卡已经跑起来”。**

- 旧 run 保留在原处：`model_design/runs/s5-continuous-four-domain-v1/main/`。
- 旧源 checkpoint：`main/checkpoints/step_00017500.pt`。
- checkpoint SHA256：`6e7d1e7d417c56271a6ff2e50044bab19d16f1f7627e48fde0c5c2c1d5fcff2d`。
- `step=micro_step=17500`，原 15 份 RNG 完整。17520 有训练日志但没有后续完整 checkpoint，不能冒充保存进度。
- 新候选：`/yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_step17500_20261106`。
- 固定副本：`handoff/source.pt`；禁止追随旧 `latest.json` 或旧 16900 候选。
- 生产源码 commit：`499e65f29ac377ab106b918f76bf51d03d4946ec`；tree：`22aae33a2e7eb137051d1ec8d3af712ce33babc9`。
- 六机工具分支：`feat/s5-six-node-20261106`；冻结工具 commit：`d351046de952456992f6e7270375ee94cb60b613`。
- 数据清单 SHA256：`fafe5a6b0f97bbc35e64b5af3585627a2fcab5e4aedcff1e7c82af374c396ee8`。
- 继承的坏样本排除清单 SHA256：`ad67acde58ed969fa449deacb1ff66fdc8bdc91150383da22f7cd364dbe524f9`。
- 历史 `tools/s5_six_node_runtime_20261106` 是 16900 旧候选，保留审计，不用于本次续训。

## 1. 超参数决定及原因

| 参数 | 原 15 卡 | 六机 48 卡 | 决定与理由 |
|---|---:|---:|---|
| 每 rank micro batch | 8 | 8 | 保持；本卡 InfoNCE 的负样本数不变 |
| accumulation | 1 | 1 | 保持；断点恰在优化步边界 |
| 名义全局 batch | 120 | 384 | 扩容自然增大 3.2 倍，不假装与旧轨迹等价 |
| existing LR | 1e-5 | 1e-5 | 完整恢复，不按卡数盲目线性放大 |
| new_aux_heads LR | 1e-4 | 1e-4 | 完整恢复；保持两组参数比例 |
| weight decay | 0.01 | 0.01 | 保持 |
| warmup / scheduler | 50 / stable | 恢复 last_epoch=17500 | 不重做 warmup，不改后续 LR 轨迹 |
| grad clip | 1.0 | 1.0 | 保持；日志 grad_norm 是裁剪前指标，不能误读为未裁剪 |
| BF16 / dropout | true / 0.1 | true / 0.1 | 保持 |
| 每 rank prefetch worker | 4 | 1 | 降低 48 rank 同时扫描共享存储的 CPU/IO 压力 |
| checkpoint / log interval | 100 / 20 step | 100 / 20 step | 保持 |
| 最大训练 token/残基裁剪 | 1024 | 1024 | 保留已验收的长蛋白训练裁剪；推理不截断 |
| 目标 | 1000000 | 显式传入 1000000 | 延续既有目标，不表示无限资源或最佳模型 |

**不建议本次同时放大 LR。** 当前学习率已较小，扩卡本身就是独立变量，直接乘 3.2 或开方缩放都缺少固定 validation 证据。减少每卡 batch 来保持全局 120 又会改变本卡对比学习负样本，因此不作为无代价替代。全局 384 的统计变化必须通过固定 robust validation、90 项完整性、shadow-B、coverage 和遗忘门评估，不能用预训练 loss 或 latest step 选模。

相同绝对 step 目标下，剩余阶段的基础样本暴露量约为原来的 3.2 倍；采样有放回，不称为 3.2 个 epoch。`samples` 是 rank0 的历史展开行计数，不是全集群去重覆盖率。CPU/IO 可能成为瓶颈，不承诺 3.2 倍吞吐。

模型结构、四域 20-step 课程（6 小分子、5 聚合物、6 生物、3 晶体）、DBAASP 微课程、loss 权重均不变。不读取 validation/test 标签训练或归一化 Encoder，不加载第三方预训练权重，不混用 S4 或其他父 checkpoint。

## 2. 已完成的验证

1. **真实 17500 checkpoint CPU 恢复**：`actual_checkpoint_cpu_verification.json` 为 `status=pass`。在扩展 rank47 上逐张量核对模型、AdamW 状态、两组 LR、weight decay、scheduler 和计数器完全一致；所有参数有限。只验证恢复，不代表 GPU 训练或分数通过。
2. **15→48 / 48→48 合成恢复**：`check_resume_cpu.py` 通过。首次扩卡明确按 `(seed,step,rank)` 重播种子；48→48 严格读取本 rank RNG，缺文件即拒绝。恢复后下一次确定性 optimizer 更新一致。
3. **原 S5 回归**：40/40 通过。新增候选控制测试 7/7 通过。
4. **真实双进程 Gloo 停止**：两个 rank 在同一优化步边界停止，final step=2，两个 RNG 完整。不是 mock 的广播测试。
5. **快照完整性**：训练 Python、配置、contract、排除清单和编排工具逐文件 SHA256 绑定；每节点启动再次检查。rank 间核对 Python、PyTorch/HIP、RDKit、NumPy、YAML、GPU 型号和数量、session、断点 hash、batch 与目标。
6. **尚未验证**：六机 48-rank RoCE 数值通信、真实 S5 20 步、48 份 RNG 保存后重启 1 步、长时稳定性和赛事分数。

## 3. 节点与固定配置

| node_rank | host / IP | global ranks |
|---:|---|---|
| 0 | VM-2-17-tencentos / 172.17.2.17 | 0–7 |
| 1 | VM-2-184-tencentos / 172.17.2.184 | 8–15 |
| 2 | VM-2-46-tencentos / 172.17.2.46 | 16–23 |
| 3 | VM-1-84-tencentos / 172.17.1.84 | 24–31 |
| 4 | VM-0-171-tencentos / 172.17.0.171 | 32–39 |
| 5 | VM-0-178-tencentos / 172.17.0.178 | 40–47 |

- master：`172.17.2.17:29688`；通信测试用 `29687`，不与旧训练端口复用。
- 每机 8 卡全部可见，按 `LOCAL_RANK` 绑卡；控制面 `eth0`，RoCE 使用四个 `mlx5_bond_*:1`、GID3。
- NCCL timeout=900 秒，静态 rendezvous=900 秒。
- Python：`/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python`。
- 训练入口是冻结快照内的 `model_design/muse_encoder`，不再从可随时变化的共享生产 Python 路径导入。
- 新长训输出：`runs/main`；试跑输出：`runs/smoke`。不覆盖原 S5 历史 run。

## 4. SSH 与权限边界

当前从 171 到四台新机器的 `zhaoyao`/`root` 公钥登录均返回 `Permission denied`。需要实际账号认证后才能由编排器统一启动；仅知道“六台密码相同”并不等于本会话持有密码。

优先配置 `zhaoyao` 公钥。若明确授权临时密码，使用进程环境注入，不写入 Git、脚本、命令参数或日志：

```bash
export S5_SSH_PASSWORD_AUTH=1
export S5_SSH_PASSWORD='由负责人在安全输入渠道提供'
export SSH_ASKPASS=/yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_step17500_20261106/tools/s5_six_node_candidate/ssh_askpass.py
export SSH_ASKPASS_REQUIRE=force
export DISPLAY=none
# 非 zhaoyao 账号时额外设置：export S5_SSH_OPTIONS='-l 实际账号'
export S5_KNOWN_HOSTS=/yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_runtime_20261106/access/known_hosts
```

密码不得写入上述文件或 `HANDOFF.md`。`known_hosts` 仅含新四机公开主机指纹；不把密码当作主机身份验证。训练进程会清除不需要的 SSH 密码环境。应用退出后进程可继续，但不保证节点断电、系统故障或存储不可用时仍计算；此类情况依赖 checkpoint 恢复。

## 5. 启动顺序（不可跳步）

设：

```bash
C=/yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_step17500_20261106
PY=/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python
TOOL=$C/tools/s5_six_node_candidate/cluster.py
```

### 5.1 六机资源检查

```bash
$PY -B $TOOL probe --candidate $C --require-free-mib 55000
```

必须六机各有 8 卡、无遗留 `muse_encoder.s5.train`、每卡至少 55000 MiB 空闲。原 A-test 若仍占本机第 7 卡，需要任务负责人停止或等其完成，不能强行抢占。报告：`preflight.json`。

### 5.2 48 卡小模型通信

```bash
setsid $PY -B $TOOL comm --candidate $C > $C/communication_driver.log 2>&1 < /dev/null &
```

等 `communication_acceptance.json` 为 `ok=true`。六机都必须正常退出、各有 8 个 `RANK PASS`、观察到 `NET/IB`，主机有 `DDP COMMUNICATION PASS: 48 ranks`。旧四机 32 卡通过记录不能替代此门。

### 5.3 真实 S5 20 步与保存后恢复 1 步

```bash
setsid $PY -B $TOOL run --candidate $C --mode smoke-start --stop-after 17520 \
  > $C/smoke_start_driver.log 2>&1 < /dev/null &
```

六机正常结束后：

```bash
setsid $PY -B $TOOL run --candidate $C --mode smoke-resume --stop-after 17521 \
  > $C/smoke_resume_driver.log 2>&1 < /dev/null &
$PY -B $C/tools/s5_six_node_candidate/check_acceptance.py --candidate $C
```

必须生成 `acceptance.json`，状态为 `smoke_and_resume_pass`，两代 checkpoint 各有 48 份 RNG，48 rank 的 host/device/session 正确、optimizer/scheduler 步数正确、loss/grad 有限、无新坏样本。20 步覆盖 20 个课程槽位但不覆盖全部 DBAASP 200-step 微课程，也不证明长时稳定。出现新跳过时先审查 `smoke_review.json`，不得忽略。

### 5.4 正式长训

仅在上一步通过后执行：

```bash
mkdir -p $C/logs
setsid $PY -B $TOOL run --candidate $C --mode train-start --stop-after 1000000 \
  --max-attempts 20 > $C/logs/formal_orchestrator.log 2>&1 < /dev/null &
echo $! > $C/logs/formal_orchestrator.pid
```

 orchestrator 和各节点 driver 都是独立 session，应用退出后由 PID 1 接管；不得依赖六个交互终端。首次 `control/first_healthy.json` 写入且健康摘要为 48/48 后，才称正式启动验收通过。

## 6. 稳定性机制

- **心跳**：每 rank 15 秒；60 秒无事件为 stale。15 秒汇总一次六机 GPU、48 rank 进度和课程。
- **进度阈值**：无进度 900 秒告警；启动宽限 1800 秒；checkpoint 宽限 3600 秒。持续 stalled 120 秒后编排器停止该次 attempt，保留现场。
- **有效 session**：旧 rank 日志不会掩盖新 session 的未启动或错误；堆栈通过 SIGUSR1 抓取且 `chain=False`。
- **保存**：临时文件 + flush/fsync + 原子 rename + 目录 fsync；48 份 RNG barrier 后才发布 latest。latest 是最近状态，不是最佳模型。
- **恢复**：有完整 latest 就 48→48 严格恢复；若首个新 checkpoint 前失败，旧 attempt 移到 `runs/history`，只从同一 17500 源点重建并继承合法跳过记录。不得混用其他父模型。
- **自动重启**：最多 20 次 attempt，退避 30 秒递增至 300 秒。源码/hash/配置/manifest/RNG、非有限 loss/grad、OOM 等确定性错误不自动重试。
- **坏数据**：普通 dataset/metadata 错误记录 `data_skips.jsonl` 并补采；多进程 append 加锁并 fsync。I/O、OOM、NCCL/CUDA/HIP/ProcessGroup 等系统错误仍 fatal，不伪装成坏数据。
- **异常停止**：可丢失未保存的尾部更新，回退到最近完整 checkpoint；不得把 final/latest 当作赛事最佳候选。

## 7. 查看与停止

```bash
$PY -B $TOOL status --candidate $C
```

关键文件：

- `control/cluster_state.json`：当前编排器、attempt 和 session。
- `runs/main/health/sessions/<session>.summary.json`：当前 48 卡真实健康，不看旧 session。
- `runs/main/latest.json`：最近完整 checkpoint。
- `runs/main/train.jsonl`、`runs/main/data_skips.jsonl`。
- `logs/formal_orchestrator.log`、`logs/drivers/`、`logs/health/`。

正常停止不要直接杀 orchestrator：

```bash
$PY -B $TOOL stop --candidate $C --grace-seconds 900 --reason 'operator planned stop'
```

控制文件在所有 rank 的同一优化步边界生效，保存 final 和 48 份 RNG，写入 `runs/main/control/stop_ack.json`。超时只终止登记的节点 driver；没有完整保存时不宣称正常停止。审阅 ack 后，正式恢复需显式：

```bash
setsid $PY -B $TOOL run --candidate $C --mode train-resume --stop-after 1000000 \
  --acknowledge-stop > $C/logs/resume_orchestrator.log 2>&1 < /dev/null &
```

## 8. 回退与评估边界

- 原 15 卡 checkpoint 和 run 未覆盖；可使用原两机启动器从 17500 恢复。48 卡 optimizer/RNG 不得交给未适配的旧启动器硬恢复。
- 新工具不自动运行 A-test、不上传赛事平台、不覆盖 `官方提交/` 或 best_fixed。
- 先前完整 A-test：step10800+v2 为 59.2906418076102；step16900+v3 的独立评估状态需查实际产物，不能借用旧分数宣称新长训提分。
- 长训候选仍须固定 robust validation、90/90、shadow-B、coverage、遗忘门和血缘验收。
