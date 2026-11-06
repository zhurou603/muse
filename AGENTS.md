# Kimi Agent 实验续接入口

当前唯一训练主线为 `s5-continuous-four-domain-v1`。每次开始或恢复工作前，先阅读：

1. `model_design/runs/s5-continuous-four-domain-v1/KIMI_AGENT_LOG.md`：人类可读的当前状态、决策、命令和验收结果。
2. `model_design/runs/s5-continuous-four-domain-v1/events.jsonl`：追加式机器可读事件日志。
3. 当前 run 下各实验的 `config.*`、`train.jsonl`、`checkpoints/checkpoints.jsonl` 和评估产物。

S4 已归档，历史入口为 `model_design/runs/s4-overnight-polymer-electronic-consolidation-v1/`；不得把 S4 treatment/control 候选混入 S5。

工作约束：

- 本地 `model_design/` 是唯一代码主线；外部 ZIP 仅作参考，不作为运行仓库。
- 不覆盖历史 run；不混用不同父 checkpoint 的候选；不读取 validation/test 标签训练 Encoder；不加载第三方预训练权重。
- 仅执行固定 S5 20-step 课程：小分子 6、聚合物 5、生物 6、晶体 3 槽位，以及既定 DBAASP 微课程；不恢复失败的 S4 treatment。
- 1-step、50-step、真实中断恢复和 7 卡 DDP 验收未通过，不启动正式连续训练。父模型与候选只按固定 robust validation、90 项完整性、shadow-B 和遗忘门选模，不按预训练 loss 或 latest step 选模。
- 每次训练、评估、代码修改、门控决策或阻塞都要更新 S5 日志，记录命令、哈希、输出路径和验证边界。

## Git 工作规范

- 仓库根目录为 `/yinghepool/tech-dcu/zhaoyao`，默认主分支为 `main`。开始工作先运行 `git status --short --branch`，确认当前分支、未提交修改和运行中的任务，再继续阅读上述 S5 日志。
- 只提交源码、测试、配置、启动脚本、文档和必要的文本清单。原始数据、模型权重、checkpoint、缓存、虚拟环境、日志、ZIP、运行产物及秘密文件不得提交；忽略规则不能用来掩盖应提交的源码。新增大文件前先检查体积和用途。
- 不提交 `.env`、私钥、令牌、凭据或未知二进制文件；不把外部仓库、ZIP 或修复包直接当成运行代码。外部补丁必须记录来源路径与 SHA256，逐项审查 diff 后再集成。
- 后台训练或评估仍运行时，不直接修改其 Python 导入路径内的代码。需要并行修复时，从已提交基线建立命名分支和独立 worktree；测试使用独立输出目录与缓存命名空间，不能复用或覆盖正在运行任务的产物。
- 功能分支使用 `fix/<主题>` 或 `feat/<主题>`。一个提交只包含一类可解释变更；提交前检查完整 diff，运行覆盖该变更的项目测试，并记录测试结果。禁止为了通过验收而删除或放宽既有安全约束。
- 不执行 `git reset --hard`、`git clean -fd`、强制推送或覆盖历史 run。撤销操作优先使用可逆的新提交；不得擅自丢弃用户未提交的修改。
- 本地提交由任务需要触发；不向远端推送、不创建远端 PR、不修改全局 Git 配置，除非用户另行明确要求。需要身份时只配置仓库级身份。
- 合并前确认相关后台任务已结束或已明确隔离，工作区干净，回归测试通过。合并后再次验证交付代码，并更新 S5 的 `KIMI_AGENT_LOG.md`、`events.jsonl` 和 `HANDOVER.md`，记录分支、commit、测试、命令和验证边界。
- 结束工作前检查 `git status --short --branch`。不得遗留测试生成的缓存、临时文件或无关修改；历史产物只可新增归档，不可删除以清理空间。
