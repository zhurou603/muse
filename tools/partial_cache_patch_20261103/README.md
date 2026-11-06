# 部分成功特征复用候选

当前正式基础为6889ba8a…17b32，前4项修复保留；本包只准备候选，尚不安装。
以前一条失败就不保存整次最终特征，下次成功项也重新计算。本补丁单独保存部分成功结果，下次只处理未成功项；失败仍返回failed，partial_hit不冒充central_hit。

## 范围与兼容性

- 完整缓存键不变，旧完整缓存仍能读取；预测头设置仍不影响冻结特征复用。
- 只有同一完整输入请求、数据版本、模型权重、配置、处理代码、词表、batch与token模式相同才复用。不是跨任意数据集逐条共享缓存。
- 新目录embedding_progress_v1存部分结果；不会把含失败的结果写入完整embeddings缓存。
- 缓存ID必须完整覆盖本次请求且成功失败不重叠；embedding、token、offsets必须对齐且有限。损坏时明确报错，不静默跳过。
- 补回成功记录后按请求输入ID顺序合并，编号、特征、token同时移动。下游按ID关联标签。
- 不改正式数据、模型训练、调度或旧结果；不恢复历史未保存的特征。进程中断前尚未写出的结果不在此补丁保障范围。

## 准备

解压至tools。保留同级fixed_card_patch_20261102与long_no_grad_patch_20261102，验证复用其工具。

```bash
/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python -B /yinghepool/tech-dcu/zhaoyao/tools/partial_cache_patch_20261103/prepare_partial.py
```

只生成candidate，不覆盖正式文件。已有candidate时停止。

## 单卡验证

178物理1号卡空闲时：

```bash
env PYTHONDONTWRITEBYTECODE=1 HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 timeout --signal=TERM --kill-after=10s 600s /home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python -u /yinghepool/tech-dcu/zhaoyao/tools/partial_cache_patch_20261103/verify_partial.py --output "/yinghepool/tech-dcu/zhaoyao/tools/partial_cache_verify_178_$(date +%Y%m%d_%H%M%S)"
```

使用4条真实fluorescence记录，临时使中间1条归一化失败；不改变数据本身。分别检查整条特征、逐token两种模式：

1. 原版首次编码3条，再运行又编码3条。
2. 候选首次编码3条，失败持续时再运行应新增0条编码，仍报告1条失败。
3. 撤掉故障，只补算1条；再运行应命中完整缓存，不再编码。
4. 对照原版无故障结果，检查成功特征数值、token、ID顺序及长度；检查旧完整缓存、改头设置仍命中。

脚本使用独立诊断缓存、单进程单卡计数；不测试多卡调度，不运行MLP训练。600秒是上限，不是预计耗时。
本地10项CPU测试已通过，覆盖归一化/编码失败、持续失败、逐步恢复、token对齐、身份变更、损坏拒绝及旧完整缓存。
只有status=completed才代表本组服务器验证完成；正式全90任务收益未测。
