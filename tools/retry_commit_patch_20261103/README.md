# 拆批重试重复结果修复候选

每批先暂存编号、embedding、tokens；整批成功才加入最终结果。失败则沿用拆批重试，暂存结果不提交。
仅改并行入口 parallel_dashboard._encode_items，不改模型、训练、数据、调度、缓存；前3项修复保留。
它保证结果不重复，但失败拆批时仍可能重算成功样本，不是消除所有重算的性能优化。

## 准备

当前正式文件应为 ea0cf8245dbb97d4b2a2ea2ff63af89dc0ab81ffd99b7a9d3de4aab84cc7e34c。
包解压到 tools；保留同级 fixed_card_patch_20261102 和 long_no_grad_patch_20261102，复用验证工具和真实样本。

```bash
/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python -B /yinghepool/tech-dcu/zhaoyao/tools/retry_commit_patch_20261103/prepare_retry.py
```

只生成candidate，不覆盖正式文件。已有candidate目录则停止。
候选SHA256：6889ba8a4a722455fa7d07633fe7765e3f61d56f48e4c584e0aee7977db17b32。

## 单卡验证

178的物理1号卡空闲时执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 timeout --signal=TERM --kill-after=10s 600s /home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python -u /yinghepool/tech-dcu/zhaoyao/tools/retry_commit_patch_20261103/verify_retry.py --output "/yinghepool/tech-dcu/zhaoyao/tools/retry_commit_verify_178_$(date +%Y%m%d_%H%M%S)"
```

- 18组小型CPU检查：长/短、token开关、嵌套重试、全成功/失败、空输入、预处理错误、批次失败但拆开后全部成功。
- 再用真实模型处理1024长度长序列及237长度短序列。原始数据不改；仅在验证进程中临时让短序列计算入口报错。
- 原版必须得到两份长序列结果；候选只能保留一份，短序列错误仍需完整记录。
- 比较无错误时的输出、重试后保留的成功特征、去掉注入错误后的恢复结果；整条和逐token两种模式都检查。
- 检查权重未更新、正式文件不变。600秒是停止上限，不是预计时间。

失败或超时表示验证未完成。只有status=completed才表示本组通过。
当前仅候选；验证通过后再准备正式安装。本包没有安装操作。
dashboard.encode_records独立入口有同类写法，此包未修改；不要将本项称为全项目所有入口修完。
旧缓存或旧结果中的重复记录不会被此测试自动修复，也未运行全90任务。
