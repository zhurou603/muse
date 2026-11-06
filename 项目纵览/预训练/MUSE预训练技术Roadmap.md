# MUSE 预训练技术 Roadmap（Encoder 锁定前 · 全程可打勾 TODO）

> 范围：从数据管线到 11-08 Encoder 锁定之间的**预训练**全部技术工作（MLP 探针工程与 B 榜流程另有清单，本文只在其与预训练选型耦合处标注）。
> 用法：每个 `[ ]` 是一个可验收 TODO，标注 `[GATE]` 的为阶段验收门——未通过不得进入下一阶段。标注 `[ABLATE]` 的为需要小实验验证的技术点（对应报告 §5.3 消融优先级）。依据标注：【R】=研究报告章节，【P】=论文证据，【Q】=赛题规则/Q&A，【C】=代码库 muse_encoder。

---

## Phase 0：地基与度量工具（先于一切训练，W1 上半周）

### 0.1 数据管线骨架
- [ ] 解析官方 `data/` JSONL（13 数据集 × 3 切分），建立统一 `Record` 中间表示（id / sample_type / inputs / labels / external_inputs 指针）【R§4.1】
- [ ] 解析 `external_inputs/`：SDF/3D 构象 → 原子+坐标数组；CIF → 晶胞+分数坐标；OMC25 ASE-LMDB → 帧迭代器；建立 `id → 结构` 索引并缓存（LMDB/parquet）【R§4.1】
- [ ] 解析 `aux_labels/`：QMugs 37 字段、GEOM 构象能量、QM9 8 性质、OMC25 力/应力（.npy）、PubChemQC/PCQM4Mv2、MegaScale、DBAASP 非 MIC 标签 → 统一 `targets` 字典（与 `model.pretraining_losses` 的 targets 约定对齐）【C: model.py】
- [ ] RDKit 安装与全量解析率统计：SMILES/PSMILES 解析失败率须 <0.5%，失败样本进降级路径并计数留证（无 RDKit 时 `featurize.py` 降级路径仅用于开发，正式训练禁用）【C: featurize.py】
- [ ] **[GATE]** 管线吞吐基准：单机 CPU 特征化 ≥ 5 万条/小时/进程；GEOM 3,200 万构象全量缓存 < 3 天完成；随机抽 1,000 条人工目检结构正确性

### 0.2 去泄漏审计（合规红线，先于任何预训练数据落盘）
- [ ] 分子域指纹查重：标准异构 SMILES + InChIKey-1，外部语料（PubChem/ZINC/ChEMBL）对 A 榜 val/test 双向查重，命中剔除外部侧【R§4.3】
- [ ] 聚合物域：规范化 PSMILES 查重（PI1M 对 PO-A-1/2/3 val/test）【R§4.3】
- [ ] 晶体域：约化化学式+空间群+母分子 InChIKey 三元组查重；沿用 OMC25 官方按起始堆积的切分【R§4.3】
- [ ] 蛋白/肽域：UniRef50/UniProt 语料对 CATH/PEER/DBAASP val/test 做 k-mer 预筛 + MMseqs2 复核（≥90% 同一性剔除）；肽的记录同时按母分子 InChIKey 查重（DBAASP 非天然肽走原子级前端时适用）【R§4.3, P: PepLand】
- [ ] 官方辅助标签的哈希校验（主办方已保证不含 val/test 物质，仍跑一遍留证）
- [ ] **[GATE]** `data_manifest.jsonl` 全量生成：每条外部数据含 name/version/source_url/license/usage/num_records/sha256；剔除日志+指纹文件 hash+种子写入复现材料【Q: 复现规范】

### 0.3 回归看板（一切"不掉点"承诺的度量工具，第 0 优先级）
- [ ] 固定探针协议脚本：输入任意 checkpoint → 输出 13 数据集 × 90 子任务的 A 榜 val 分数表；种子/轮数/早停全锁死，MLP 结构固定（D×48→1/K）【R§6.4】
- [ ] 看板基线列：ECFP4+RF、随机权重 Encoder、逐域单任务基线（先建立"地板分"，之后的预训练收益都相对它度量）【R§2.6 冻结嵌入常输指纹基线的警告】
- [ ] "性质留出影子集"构建：从辅助标签中留出 5–8 类性质（如偶极矩、某 3 个 ADMET、MegaScale 一个子家族）全程不进 Encoder 训练，作为 B 榜泛化的代理指标【R§6.4】
- [ ] **[GATE]** 看板端到端跑通一次（随机权重 → 分数表 → 存档），确认与官方 `score.py` 在 val 上的口径一致

### 0.4 工程地基
- [ ] Docker 训练镜像锁定（digest 固定）；无网络环境端到端演练脚本【Q】
- [ ] checkpoint 规范：可续跑（含 optimizer/scheduler/RNG 状态）、`checkpoints.jsonl` 记录路径/sha256/进度/loss【Q】
- [ ] 全局种子与确定性开关写入配置；混合精度 bf16 + FlashAttention + 序列打包基准测试【R§7】
- [ ] **[GATE]** 用 `muse_encoder` 小配置（4L/128d）在 1 万条合成混合数据上完成一次 100 步训练 → 看板出分 → 复现脚本重跑 loss 曲线逐位一致

---

## Phase 1：词表与前端标定（W1 下半周）

- [ ] SMILES 词表：PubChem+ZINC 抽 500 万条语料统计构建（`tokenizers.build_vocab`），校验覆盖率：目标语料 99.5% 分子无 `<unk>`；与 TransPolymer 化学感知切分对齐【C: tokenizers.py, P: TransPolymer】
- [ ] PSMILES 词表：PI1M 全量统计；`*`/`(*)` 位点 token 存在且语义正确；对比 polyBERT canonicalize 流程的规范化一致性【P: TransPolymer/polyBERT】
- [ ] 残基词表：26–30（20 标准 + B/Z/X/U/O + 特殊符号）；**[ABLATE]** 肽记录路由规则标定：纯标准残基序列 → 序列前端；含修饰/HELM → 原子级前端（转 SMILES）——抽样 500 条 DBAASP 训练集验证路由覆盖率【P: PepLand】
- [ ] 原子级前端特征：元素/键/SPD/距离高斯基维度核对；RDKit 特征与降级特征在同一 SMILES 上的输出一致性测试（只允许图结构差异，不允许形状差异）【C: featurize.py】
- [ ] 晶体前端 PBC 图：超胞扩展的正确性测试（截断球内邻居数 vs pymatgen 参考实现，误差 0）；`max_neighbors=32` 在 OMC25 上的截断率统计（目标 <1% 原子被截断）【C: featurize.py: build_pbc_graph】
- [ ] 共聚物前端：单体拼接 + 组成比嵌入的张量形状与块对角键矩阵正确性测试【C: featurize.py: featurize_polymer】
- [ ] **[GATE]** 四域前端各 100 条真实官方数据 → collate → 一次前向（随机权重），无 NaN、形状正确、token 长度分布统计落盘

---

## Phase 2：序列域预训练 S1（W1 末 – W3）

> 目标：SMILES/PSMILES/蛋白/肽四条序列线出基线分；完成 MTR 有效性的关键消融。

### 2.1 语料与采样
- [ ] SMILES 语料 6,000 万条落盘（PubChem 4,000 万 + ZINC 2,000 万），骨架温度采样 τ=0.005 的三级采样器（域→数据集→骨架）实现【P: Uni-Mol2】
- [ ] PI1M 100 万 PSMILES + 5× 增强（TransPolymer 配方：随机 SMILES 枚举）【P: TransPolymer】
- [ ] UniRef50 全量（约 5,400 万序列）；蛋白长度截断策略：≤1024 直接进，超长随机裁窗（记录窗口种子）【P: ESM-2】
- [ ] **[ABLATE]** 肽两步课程子实验：UniProt ≤50 残基子集（约 800 万）先训 → CycPeptMPDB/PDB 非天然肽（仅结构，无标签）续训 vs 单步混合，看板比肽域探针分（PepLand 证据：两步全胜）【P: PepLand, R§6.4】

### 2.2 损失与训练
- [ ] MLM 损失接入：15% / 80-10-10；序列域期望掩码率单测（烟测 [7] 已覆盖，训练侧再验证 batch 级统计）【C: masking.py】
- [ ] **[ABLATE·最高优先]** MTR 描述符蒸馏开/关对照：RDKit 200 维描述符离线预计算（z-score，统计量只来自预训练语料）、中部+尾部双读出；12 小时级小模型对照，看板比 SM 域探针分差（ChemBERTa-2/CheMeleon 证据支持开，但需自查收益-成本）【P: ChemBERTa-2/CheMeleon, R§5.3】
- [ ] 肽结构语义掩码接入：RDKit 原子角色标注（肽键骨架 vs 侧链）+ AdaFrag 片段 id 生成管线；Bulk/SideChain/Fragment 30/40/30 混合采样单测【C: masking.py, P: PepLand】
- [ ] 训练超参基线：AdamW、lr 1e-4（warmup 1 万步）、cosine 衰减、batch 等效 2048 条、梯度裁剪 1.0；loss 曲线与吞吐（token/s/卡）记录进 `logs/`【Q】
- [ ] **[GATE]** S1 退出：四条序列线的看板探针分 ≥ 对应单域文献基线的 80%（如 SMILES 线 vs ChemBERTa-2 公开分数、蛋白线 vs PEER 论文冻结设定分数）；影子留出集分数建档

---

## Phase 3：原子-3D 域预训练 S2（W2 – W4，与 S1 并行）

- [ ] 构象语料：GEOM 3,200 万 + QMugs/QM9/PubChemQC 子采样合计 4,000 万构象；同一分子多构象归组（对比学习正样本对用）【R§4.4】
- [ ] 坐标去噪：多 σ 对数均匀采样（0.04–1.0 Å）+ σ 条件嵌入 + 50% 概率连带掩对特征（防捷径）——三个细节逐一写单测验证（σ 分布直方图、掩特征开关、标签=真实噪声）【P: Uni-Mol2, C: model.py/config.py】
- [ ] 掩码原子 15% 与去噪的联合调度：同一样本两种破坏的组合概率标定（Uni-Mol2 配方：去噪样本上 50% 连带掩原子/键特征）【P: Uni-Mol2】
- [ ] 能量监督接入：GEOM 构象能量（z-score 按分子内统计 or 全局统计——**[ABLATE]** 两种口径对照，构象相对能量理论上应按分子内标准化）【R§5.2】
- [ ] 2D↔3D 跨视图对比：同分子 SMILES 视图 vs 构象视图 InfoNCE（τ=0.07），批内负样本按骨架去重；**[ABLATE]** 开/关对照看 SM-A-1/2（纯 SMILES 任务）探针分变化【P: 3D Infomax/GraphMVP】
- [ ] **[GATE]** S2 退出：SM-A-3/4/5 看板分显著超 S0 地板分；QM9 四性质探针 MAE 进入文献同协议区间的 1.5 倍以内；S1 四线分数波动 <1 个点（回放 15–30% 已生效）

---

## Phase 4：晶体域 S3（W3 – W5，最重算力块）

- [ ] OMC25 子采样策略：25.1M 帧 → 6–8M 帧；按能量/受力离群度过滤（|z|>10 剔除）+ 按起始堆积分层抽样（保持化学空间覆盖）【R§2.3/4.4】
- [ ] 能量/力/应力三头联训：力:能 权重比 **[ABLATE]** 10:1 vs 30:1 vs 100:1（力为逐原子监督，信号密度高两个数量级）；应力项 z-score；Huber δ=1.0【R§5.2, P: UMA/eSEN】
- [ ] 直接力 → 保守力两阶段：第一阶段非保守头（快）；收尾 1–2 天接保守力微调（F=−∇E，成本约 3× 但窗口允许）；能量守恒性作为 OOD 诊断指标记录【P: eSEN 两阶段省 40%】
- [ ] 晶胞/空间群嵌入与标量条件（电荷/自旋/温度）通路的前向验证【C: model.py: scalar_cond】
- [ ] **[GATE]** S3 退出：OC-A-1 探针 MAE 超"组成均值基线"（用化学式统计量的平凡模型）；PO-A-4 出分；S1/S2 各域看板波动 <1 个点

---

## Phase 5：全域联合退火 S4（W5 – W6 初）

- [ ] 四域混采配比 小分子40/晶体25/蛋白20/聚合物15 起步，按看板微调；旧域回放 15–30% 防遗忘（灾难性遗忘监控：每 5,000 步跑一次全看板）【R§4.4/6.4】
- [ ] MoLE 门控健康检查：8 专家的域条件负载均衡（熵统计），若坍塌到 ≤2 专家则加负载均衡辅助损失或降专家数【P: UMA, C: layers.py: MoleFFN】
- [ ] **[ABLATE]** A 榜标签轻量混入 0% vs 5%：独立任务头、只取 train split；判定标准不是 A 榜分，而是**影子留出集**分数——掉则回退到 0%【R§5.2/5.4 风险④】
- [ ] **[ABLATE]** GRU 池化 vs CLS⊕mean-pool：全看板 A/B（PepLand 证据限于肽域，是否全域启用由数据决定），锁定其一后不再变【P: PepLand, C: config.pooling】
- [ ] 读出维度最终确认 D=1024；表征标准化仿射折入 Encoder（固定统计量，来自训练集）【Q: MLP 规范】
- [ ] **[GATE]** S4 退出：全 13 数据集看板全绿（相对各阶段峰值掉点 <1）；影子留出集分数不低于 S2 结束时的 95%；推理吞吐实测（8×H200 模拟：GEOM 32M + OMC25 25M 全量编码时间）达标【R§7】

---

## Phase 6：封板与复现演练（W6 – W7，11-08 前）

- [ ] 权重冻结 + sha256 登记；`start_checkpoint.json` 确认为空（从零训练声明）【Q】
- [ ] `method.md` 逐阶段定稿：每阶段数据/目标函数/步数/学习率/消融结论【Q】
- [ ] `compute.json`：H200 卡时实测汇总，给出主办方一周复现所需算力预估【Q】
- [ ] 干净环境完整复现演练：Docker digest 固定 → `reproduce_encoder.sh` 重跑关键阶段（或抽样段）→ loss 轨迹与 checkpoint 对齐【Q】
- [ ] 推理包演练：`encode.sh` 在无网络、输入只读条件下的全量 dry-run；`embeddings.npy` float32/有限性/顺序一致性校验；逐残基子集 `tokens.npy`+`token_offsets.npy` 格式校验【Q, C: inference.py】
- [ ] 合规自检表 10 项逐项打勾归档（报告 §6.5）
- [ ] **[GATE]** 11-08 24:00 前：Encoder 锁定提交 + 训练复现材料提交完成

---

## 附 A：技术点 → 验证方式速查

| 技术点 | 出处 | 验证 | 状态 |
|---|---|---|---|
| 温度采样 τ=0.005 压骨架长尾 | Uni-Mol2 | 采样分布直方图 + 看板 | 待做 |
| 50% 连带掩对特征防去噪捷径 | Uni-Mol2 | 单测 + 去噪 loss 曲线形态 | 代码已就绪【C】 |
| MTR 描述符蒸馏（双读出） | ChemBERTa-2/CheMeleon | **[ABLATE]** 开关对照 | 头已就绪，待实验 |
| 肽 Bulk/SideChain/Fragment 掩码 30/40/30 | PepLand | 单测（烟测[7]已过）+ 肽域看板 | 代码已就绪，待真实标注管线 |
| AdaFrag 片段化（保酰胺键+BRICS） | PepLand | 恢复率 ≥97% 复算 | 待做（需 RDKit） |
| 肽两步课程（天然→非天然） | PepLand | **[ABLATE]** 单步 vs 两步 | 待做 |
| GRU 池化读出 | PepLand | **[ABLATE]** 全看板 A/B | 代码已就绪 |
| 力:能 = 30:1 量级 | OC20/UMA 惯例 | **[ABLATE]** 三档对照 | 待做 |
| 直接力→保守力两阶段 | eSEN | 训练时长与能量守恒性 | 待做 |
| 多 σ 去噪 + σ 条件嵌入 | score matching/Uni-Mol2 | σ 分布单测 | 代码已就绪 |
| 2D↔3D InfoNCE | 3D Infomax/GraphMVP | **[ABLATE]** 开关对照 | 待做 |
| MoLE 域软路由 8 专家 | UMA | 门控熵监控 | 代码已就绪，待监控 |
| OMC25 离群过滤 + 分层子采样 | sAlexandria/OMC25 | 过滤统计报告 | 待做 |
| UniRef50 聚类均匀采样 | ESM-2 | 采样器实现 | 待做 |
| 蛋白 MLM 规模安全区（~300M） | Compute-Optimal PLM | 架构已定 386M | 已落实【R§6.3】 |
| A 榜标签 ≤5% 混入 + 影子留出集 | 本方案纪律 | **[ABLATE]** + 看板门禁 | 待做 |
| 批次无关性 | 赛规 | 烟测[2][8]已过（max\|Δ\|<1e-5） | ✅ 已验证 |
| MLP ≤2 层 ≤100K | 赛规 | 烟测[5]已过（49,249 / 50,229） | ✅ 已验证 |

## 附 B：风险触发器（任一命中即启动预案）

1. S1 结束时 SMILES 线探针分低于 ChemBERTa-2 公开值的 60% → 降级：主干缩至 12L/512d，先保全域覆盖再追单域高度。
2. S3 晶体前端工程延期超 3 天 → 降级：晶体域退化为"组成+局部环境"无 PBC 表征保基线分，PBC 版后续追赶。
3. MTR 消融显示收益 <2 个点 → 关闭描述符组，省出约 30% 序列域算力转投 S3。
4. MoLE 门控坍塌（熵 < ln(3)）→ 降专家数至 4 并加重启。
5. 影子留出集在 S4 掉分 >5% → A 榜标签混入权重 5%→0%，联合退火配比回滚至上一看板全绿版本。
6. 全量推理吞吐不达标 → embed_dim 不变、max_seq_len 分桶 + 原子域按原子数排序批处理；仍不达标则主干降 16 层重训退火段。
