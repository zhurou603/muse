# 长序列推理梯度修复候选

仅在 parallel_dashboard.py 的 _encode_items 前增加 @torch.no_grad()。
普通推理入口已有这个设置；并行长序列入口漏了，导致输出带梯度记录，转 NumPy 报错。
不改训练函数、权重、分窗算法、缓存键；保留固定卡和缓存修复。

## 使用条件

- 解压到 /yinghepool/tech-dcu/zhaoyao/tools。
- 保留同级 fixed_card_patch_20261102，复用其中的 bench_encoder_workers.py。
- 正式文件 SHA256 必须为 99d22c6c8f3f0f3f3954043f2f6b6f3dc4c30901a7d3a483095c7bb44a4f30b3。
- prepare_long.py 仅生成 candidate 副本，不覆盖正式文件；已有 candidate 时停止。
- 验证仅用一张可见 GPU，不训练、不改正式缓存；输出目录必须新建。

## 执行

```bash
/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python -B /yinghepool/tech-dcu/zhaoyao/tools/long_no_grad_patch_20261102/prepare_long.py
```

确认 PREPARED 后，在空闲的 178 物理 1 号卡执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 timeout --signal=TERM --kill-after=10s 600s /home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python -u /yinghepool/tech-dcu/zhaoyao/tools/long_no_grad_patch_20261102/verify_long.py --output "/yinghepool/tech-dcu/zhaoyao/tools/long_no_grad_verify_178_$(date +%Y%m%d_%H%M%S)"
```

600 秒是停止上限，不是预计耗时。

## 验收内容

1. 三条原始失败记录来自已下载的 failure_all_data_20261031_v2.zip，长度 1024、2048、7406，历史错误为 requires grad；记录未修改。
2. 最短长序列先走正式函数，必须复现同类错误。
3. 三条长序列及一条 fluorescence 短序列分别走候选，对比原有带 no_grad 的 dashboard.encode_records。
4. 检查整条/逐 token 特征一致、长度完整、无失败；另查仅整条特征模式。
5. 确认前向关闭梯度、退出后恢复调用方状态、参数版本与 requires_grad 不变、没有参数梯度。

本地 5 项 CPU 行为测试通过，使用真实函数和小型张量；服务器完整模型验证尚未执行。
候选 SHA256：ea0cf8245dbb97d4b2a2ea2ff63af89dc0ab81ffd99b7a9d3de4aab84cc7e34c。
只有 report.json 的 status=completed 表示本组验证通过；报错或超时都是未完成。
本包不含安装操作，服务器验证通过后再正式替换。
这不是全部失败数据恢复测试，不解决 OOM、HIP 内核错误或批次失败后重复追加的问题。
