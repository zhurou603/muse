# S5 六机 48 卡扩容交付文档

更新日期：2026-11-06（北京时间）。面向训练负责人及服务器操作人员。

**交付结论：新增四机环境及四机32卡通信已通过；当前S5六机候选已在服务器准备成功，CPU合成模型恢复检查已通过。六机48卡通信、真实S5训练和保存后重启尚待执行，因此当前是“候选方案交付”，不是“正式训练验收完成”。**

验证依据为用户回传的服务器终端日志。助手完成代码准备与静态核对；所有服务器执行均由用户完成。原生产源码和checkpoint没有被本候选覆盖，旧任务没有被本工具停止。

## 交付摘要：接手先看这里

目标是将当前 `muse_encoder.s5.train` 从原两机15个训练进程扩展到六机48个进程。以前的S1A双机16卡启动器是另一阶段的方案，不能直接用于当前S5。

| 项目 | 当前交付状态 | 验证边界 |
|---|---|---|
| 新四机Python环境迁移 | 已完成 | 从171的环境包复制到各节点本地 |
| 新四机逐卡FP32/BF16及单机8卡DDP | 已通过 | 各机8张卡，基础计算/通信 |
| 新四机联合32rank通信 | 已通过 | 3轮广播、all-reduce、DDP平均梯度；四机日志确认RoCE |
| 原171/178环境与网络参数 | 核对完成 | Python/PyTorch版本及RDMA/GID；不是当前48卡联合运行验证 |
| 六机候选源码及恢复方案 | 已准备并部署 | 独立目录；源代码与checkpoint检查通过 |
| CPU恢复验证 | 已通过 | 小型合成模型的15->48和48->48恢复逻辑 |
| 六机48rank通信 | 未执行/未回传通过 | 需先协调原两机任务 |
| 真实S5 20步＋重启1步 | 未执行 | 真实数据、模型、显存和保存恢复仍待验收 |
| 正式续训、性能和质量 | 未验收 | 不承诺扩容后线性加速或分数提高 |

**当前下一步：负责人协调171/178的训练、评估及48卡使用窗口，然后执行第6节六机通信测试。已完成的prepare和CPU检查无需重复。**

### 已固定的候选训练状态

准备命令在 `VM-2-17-tencentos` 上读取了以下源文件：

```text
/yinghepool/tech-dcu/zhaoyao/model_design/runs/s5-continuous-four-domain-v1/main/checkpoints/step_00016900.pt
```

| 字段 | 已回传值 |
|---|---:|
| step / micro_step | 16900 / 16900 |
| samples（原rank0本地历史计数） | 148728 |
| existing 参数组学习率 | 0.00001 |
| new_aux_heads 参数组学习率 | 0.0001 |
| 两组weight_decay | 0.01 |
| 名义全局batch变化 | 120 → 384 |

这不是实时追随旧任务的 `latest.json`。源checkpoint的SHA256已由准备脚本核验并写入候选 `contract.json`，终端摘要没有显示该值，本文件不猜写。

### 共享盘交付位置

| 用途 | 服务器绝对路径 |
|---|---|
| 候选工具目录 | `/yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_candidate` |
| 六机训练启动器 | `/yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_candidate/launch.py` |
| 已准备的候选根目录 | `/yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_runtime_20261106` |
| 独立训练源码 | `/yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_runtime_20261106/model_design/muse_encoder` |
| 配置副本 | `/yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_runtime_20261106/config.yaml` |
| 固定恢复信息 | `/yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_runtime_20261106/contract.json` |
| 固定checkpoint副本 | `/yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_runtime_20261106/handoff/source.pt` |
| 六机通信启动脚本 | `/yinghepool/tech-dcu/zhaoyao/tools/start_cluster_comm.sh` |
| 通信数值检查脚本 | `/yinghepool/tech-dcu/zhaoyao/tools/check_ddp_runtime.py` |
| 六机通信日志 | `/yinghepool/tech-dcu/zhaoyao/tools/cluster_comm6_logs` |

以上位于共享盘，仅部署一次。候选路径中的 `runs/smoke`、`runs/main` 和 `acceptance.json` 是后续执行才产生的输出，当前不应认为已存在或已通过。

### 环境与部署来源

- 新四机：TencentOS Server 3.1、x86_64，`/opt/dtk`指向`/opt/dtk-26.04`；每台8张BW1000_H，每卡实测约63.98GiB。
- Python：3.10.22；解释器路径为 `/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python`。
- `pip show torch`版本：`2.10.0+das.opt1.dtk2604.2608131652.g28498c`；新节点运行时显示torch 2.10.0、HIP 6.3.26113。
- 原171/178回传的Python和PyTorch包版本与上述一致；环境对齐不代表当前所有卡已空闲。
- 环境包：`/yinghepool/tech-dcu/zhaoyao/tools/muse-pretrain-env_20261106_143512.tar.gz`。新四机已解压到各自本机环境目录并执行`conda-unpack`，不需要再次安装另一套PyTorch。
- 新四机`/yinghepool`挂载核对为Lustre，来源`172.17.1.223@tcp:/7f515aed/cfs/`；数据通过共享盘复用，无需逐机上传。

### CPU恢复验证记录

修正版`check_resume_cpu.py`由用户在2.17执行，输出：

```text
PASS: rank47 restores exact model, optimizer moments/LR, scheduler and counters without old RNG47
PASS: 48->48 restores RNG; next deterministic optimizer update exactly matches
CPU RESUME CHECK PASS; synthetic states only. Real S5/DDP/save/restart still pending.
```

初版测试遗漏了模拟rank0写主checkpoint，导致`next.pt`缺失；已修正测试，同时模拟rank47写随机状态和rank0写主文件。只改测试脚本，没有为使测试通过而改动训练保存规则。修正版文件SHA256为：

```text
27e9dbf11c1caefac587241748fde73905b34e983762b8f4b99e0c5b2c294e02
```

不要用旧`s5_six_node_candidate_20261106.zip`中的测试覆盖修正版；后续重新部署可使用`s5_six_node_candidate_20261106_v2.zip`或当前修正版目录。本文件的状态已更新到CPU检查通过，早期压缩包内说明只代表打包时状态。

末尾出现`Could not open /var/log/hylog/.`日志访问提示，没有阻止本次CPU测试输出通过；不据此修改系统日志目录权限。CPU结果不代表真实48卡模型恢复已经验证。

## 1. 这次具体准备了什么

- 当前 S5 专用六机启动器；每机 8 个长期存活的训练进程，共 48 rank。
- 独立源码副本、配置副本、输出目录；保留现有 S5 数据、损失、优化器和课程实现。
- 第一次从 15 rank 检查点恢复；随后 48->48 正常恢复。不是从头训练，不是冻结 Encoder 训练 MLP。
- 训练前绑定 LOCAL_RANK。模型在每个进程内初始化一次并加载一次 checkpoint 权重，随后连续处理 batch。
- 保留已审核的评估代码，包括固定卡修复；本启动器不自动启动评估，也不把训练卡另行分给评估。
- 源码哈希、模型配置、数据清单、optimizer/scheduler、step 边界及恢复来源检查；不支持未经核验的其他 world size。
- CPU 小模型恢复验证、真实 S5 20 步试跑、保存后再恢复 1 步及 48 rank 日志验收。

## 2. batch 和训练语义

| 项目 | 当前观察到的旧 S5 | 六机候选 |
|---|---:|---:|
| rank 数 | 15（171 为7、178 为8） | 48 |
| 每 rank 基础样本数 | 8 | 8 |
| 梯度累积 | 1 | 1 |
| 名义全局 batch | 120 | 384 |
| CPU 预取 worker/rank | 配置4 | 启动参数1 |
| 学习率 | 旧 checkpoint 的实际值 | 完整恢复，不自动放大 |

保留每卡8的理由：S5 的 InfoNCE 对比在本卡 batch 内计算，改成3会改变负样本数。全局384仍然改变了梯度统计，不保证与旧120相同效果；后续要看验证指标和吞吐。不是把旧 S1A 的全局2048配置搬过来。

预取降至1是为了避免48×4=192个CPU进程同时读共享数据；不是性能最优结论。当前已有 SDF 逐条扫描等问题，本候选没有顺便重写数据读取器；扩容不保证3.2倍提速。

保持同一个绝对stop-after会让扩容后的剩余阶段抽取约3.2倍基础样本；不能把“恢复step不变”理解成“总数据预算不变”。本候选不擅自换算课程步数，正式终点由负责人按数据/时间预算确定。

基础样本可展开成2D/3D多个视图，故384不是每步 tensor 行数。原 `samples` 是 rank0 本地展开行计数，保留其历史值，不把它冒充全局样本数或去重覆盖率。采样有放回，也不能据步数推算训练了一个 epoch 的百分之几。

## 3. 恢复方案

1. 在共享盘上复制并固定一个完整 checkpoint，用 SHA256 校验。即使输入是 `latest.json`，最终启动 session 也只引用已固定文件。
2. 严格恢复 Encoder/Embedding/全部头、AdamW 参数组及动量、scheduler、step/micro_step/samples。原累积为1，故不换算 step，不重启 warmup，不调整 loss 权重。
3. 15->48时旧文件不可能有 rank15..47 的 RNG。所有48rank按固定的 `(配置seed, 恢复step, 新rank)` 重新播种并记录；明确放弃与旧15rank逐步数值一致的承诺。
4. 数据采样保留原 `(seed, stage, micro_step, rank)` 算法；新增rank扩展随机流，不重放所谓未完成epoch。采样与dropout随机状态是两套机制。
5. 48->48时严格读取本rank RNG sidecar，缺少任何文件就报错；不会偷偷重新播种。
6. 新输出继承旧 `data_skips.jsonl` 的固定副本，避免已知坏样本排除列表丢失；继承时刻和checkpoint并非严格一致的数据快照，尤其不能保证旧进程继续运行后的排除状态相同。
7. 原数据替换机制保留并记日志。试跑出现新跳过记录时只输出待审报告，不自动放行正式启动。

正式交接前由原任务负责人停止/协调旧训练和评估并确认最终checkpoint，然后重新prepare到一个新目录。工具不会杀进程。提前用旧checkpoint做候选试跑是分叉实验，不能声称接上了随后仍在运行的旧任务的全部进度。

## 4. 节点与通信

| node_rank | 节点 | local_rank | global rank |
|---:|---|---|---|
| 0 | 172.17.2.17 | 0..7 | 0..7 |
| 1 | 172.17.2.184 | 0..7 | 8..15 |
| 2 | 172.17.2.46 | 0..7 | 16..23 |
| 3 | 172.17.1.84 | 0..7 | 24..31 |
| 4 | 172.17.0.171 | 0..7 | 32..39 |
| 5 | 172.17.0.178 | 0..7 | 40..47 |

master=172.17.2.17:29688；与旧训练29828、六机通信29687隔离。eth0控制通信，四个 mlx5_bond_*:1、GID3、RoCE。静态 rendezvous 等待900秒，无自动重启。新增四机32rank通信已有用户实测通过；六机48rank还没有实测结论。

所有命令由用户执行。共享盘上部署一次即可，各机Python环境仍是各自本地的 `/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python`。本启动器会占用六台全部8卡，请先协调资源。不要以瞬时利用率0判断可抢占。

## 5. 上传和准备（CPU，不启动训练）

**本节在当前候选目录已完成，以下保留为重建流程。不要在已存在的输出目录重复prepare，也不要为了重跑而删除当前候选。** 如果正式切换时旧任务已超过16900步，应由负责人确定最终checkpoint，再选择新的输出目录重新准备，并将后续命令中的候选路径同步替换。

把整个 `s5_six_node_candidate` 文件夹上传为 `/yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_candidate`。如果上传本目录的ZIP，在tools解压一次即可。

在任一节点执行，输出目录必须不存在：

```bash
/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python -B /yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_candidate/prepare_candidate.py --checkpoint /yinghepool/tech-dcu/zhaoyao/model_design/runs/s5-continuous-four-domain-v1/main/latest.json --output /yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_runtime_20261106
```

这会CPU读取完整checkpoint，复制checkpoint及源码到独立目录，需要足够内存和磁盘。`data`只是指向共享原始数据的软链接，不复制整个数据集。源码来自11月6日13:36快照的67个Python文件和配置；任一源码变化会明确失败，不强套旧补丁。失败目录不算准备完成，查明原因后使用新目录重新准备。

```bash
/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python -B /yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_candidate/check_resume_cpu.py --candidate /yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_runtime_20261106
```

CPU检查运行小型合成模型的几次optimizer更新，只写临时目录；比较恢复前后完整参数、AdamW动量、学习率、scheduler与计数，验证48->48随机恢复和下一个更新一致，验证错误配置/缺失RNG显式失败。它不替代真实模型验证。

## 6. 六机小通信、真实试跑和重启

六机资源协调好后，先在六个终端分别执行已有通信脚本：

开始前确认`start_cluster_comm.sh`是支持`--six-nodes`的版本，且`check_ddp_runtime.py`已在共享tools目录。先查看脚本支持项，不要把四机模式误当六机验收：

```bash
grep -n -- '--six-nodes' /yinghepool/tech-dcu/zhaoyao/tools/start_cluster_comm.sh
```

```bash
bash /yinghepool/tech-dcu/zhaoyao/tools/start_cluster_comm.sh --six-nodes
```

通信验收须同时满足：本轮主节点日志含`DDP COMMUNICATION PASS: 48 ranks`；本轮六台日志均有RoCE网络选择记录；六台命令正常结束，无失败/超时。只看到`ProcessGroupNCCL destructor`或`heart beat monitor thread joined`不能判定成功。对照本轮六个日志，不要拿旧32rank结果代替。

六机通信全部正常结束且确认48rank数值检查通过后，在一台节点准备真实试跑的session：

```bash
/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python -B /yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_candidate/launch.py prepare --candidate /yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_runtime_20261106 --mode smoke-start
```

它会打印一条完整的 `launch.py start --session ...` 命令。**把同一条命令在六台分别执行**，不要六台分别prepare。前启动的等待后面的；保持终端连接。试跑训练20个真实更新步，只写候选 `runs/smoke`。原模型仍不变。

对当前16900步副本，首次试跑应从16900更新到16920，恢复试跑再到16921。若以后改用了新的源checkpoint，这些终点由脚本按新源step自动计算，不要手填16920/16921。

全部六台结束后，在一台准备重启session：

```bash
/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python -B /yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_candidate/launch.py prepare --candidate /yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_runtime_20261106 --mode smoke-resume
```

仍把打印的同一条启动命令在六台执行，恢复后再更新1步。结束后在一台验收：

```bash
/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python -B /yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_candidate/check_acceptance.py --candidate /yinghepool/tech-dcu/zhaoyao/tools/s5_six_node_runtime_20261106
```

检查48rank设备/节点绑定、两个会话完成、20+1步、两代checkpoint及各48份RNG、optimizer/scheduler进度、有限loss和新跳过记录。仅全部通过才写 `acceptance.json`。失败/重试留下的额外会话要求人工审查，不用简单最后一行PASS掩盖早前失败。

20步覆盖课程槽位，但DBAASP子课程要200步才走完所有选项；这不是全数据或所有极端长度验证，也不是模型质量和长时稳定性结论。

## 7. 正式候选续训与回退

`launch.py prepare` 还有 `train-start`、`train-resume`，二者都要求试跑验收文件和显式绝对 `--stop-after`。由负责人决定训练终点，例如从固定checkpoint再训练100步，传入“源step+100”的整数；不在交接中擅自沿用1000000这个旧终点。

`train-start` 从固定旧checkpoint恢复到独立 `runs/main`，试跑20+1步的权重不混入正式输出。`train-resume` 从这个main的最新已存checkpoint恢复。session固定检查点、目录和终点，六台必须使用同一session。CPU worker固定1/rank，训练模型不会由CPU预取worker加载。

日志：候选 `logs/<session>/<hostname>_*.log` 和 `runs/<smoke或main>/health/rank_*.jsonl`。每个rank应输出一次 `RESUME VERIFIED`。正常停止优先到设定stop-after，异常只保留已完成checkpoint，不伪装完成。

回退：停止候选由任务负责人协调，用原15rank启动配置恢复原生产目录的原checkpoint。不能把新48rank的optimizer/checkpoint拿给未经扩容适配的旧代码硬恢复。原文件从未被覆盖。

## 8. 已知限制和评审材料

- 候选源码改变3个文件：s5/train.py、s5/runtime.py、pretrain/trainer.py；新增s5/expansion.py。不是改写另一套训练器。
- checkpoint每rank在启动时读取一次payload并复用于恢复，避免旧入口/runtime/manager三次重复torch.load。SHA256核验仍需要读文件；48rank同时读大checkpoint仍可能给共享存储造成压力。
- 所有卡保持可见，LOCAL_RANK绑定模型；延续原sidecar记录全部8个可见设备的RNG格式，不混用“每进程只可见一卡”的格式。
- 日志loss沿用原rank0本地汇总，不是重新实现的48rank全局loss统计。
- 启动器使用本机/tmp中的当前用户锁，阻止同用户同时启动两份本候选；它不替代任务调度器，不能识别/停掉其他启动器的任务。中断时只终止自己新建的进程组，旧任务没有自动停止。
- 数据、源配置、checkpoint若与当前快照不符，先更新核对，再出补丁；不能通过删检查绕过。
- DDP梯度跨rank平均与每进程绑定设备的依据：[PyTorch DDP官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。具体S5损失及采样语义以本次审阅的源码为准。

## 9. 交接责任与最终验收清单

| 角色 | 接手事项 |
|---|---|
| 原训练负责人 | 协调旧171/178训练和评估，确认最终checkpoint、可用48卡窗口、是否接续16900或更新断点 |
| 操作执行人 | 在六台启动同一通信模式和同一训练session，保留六机日志及验收输出 |
| 训练负责人 | 确认全局batch增至384、原学习率不变、正式stop-after与数据预算，决定后续性能及质量评估 |
| 候选维护方 | 依据实际报错修复候选，保持生产独立；不得把未通过检查包装成完成 |

- [x] 新四机环境迁移与基础计算通过。
- [x] 新四机32rank RoCE通信通过。
- [x] 当前S5源码、配置和checkpoint核对，16900候选准备成功。
- [x] 修正版CPU恢复检查通过。
- [ ] 原171/178资源和最终断点完成协调。
- [ ] 六机48rank通信数值检查与六机正常退出通过。
- [ ] 真实S5 20步、完整保存、48rank重启1步通过。
- [ ] 验收脚本生成`acceptance.json`；新跳过数据如有则先审查。
- [ ] 负责人确认正式batch/学习率/训练终点，创建正式候选session。

收到交付后的首个动作是协调六机测试窗口。环境和CPU恢复已经有通过证据；真实六机训练仍需以上剩余项目，不能仅凭本文件宣称正式扩容完成。