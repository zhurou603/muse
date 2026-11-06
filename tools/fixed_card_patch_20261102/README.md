# 固定编码进程用卡：独立候选包

仅修复编码进程池分配；不替换生产文件，不修改数据、模型或正式缓存。
prepare.py严格核对原parallel_dashboard SHA256后，在包内candidate目录保存原文件备份、候选完整源码和局部差异。
候选增加每个worker初始化时固定设备，编码job使用固定设备；CPU归一化不分配卡。
固定编码池有进程退出时明确失败，不接受静默替换后继续运行。
单进程调用保留原逻辑，缓存键和编码结果合并逻辑不变。

## 171准备（不占显卡）

上传zip到 /yinghepool/tech-dcu/zhaoyao/tools/，在tools目录执行：

```bash
unzip -n fixed_card_patch_20261102.zip
```

```bash
/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python /yinghepool/tech-dcu/zhaoyao/tools/fixed_card_patch_20261102/prepare.py
```

必须出现PREPARED。candidate已存在会报错，防止覆盖之前的候选和备份。

## 后续两卡验证（确认用卡后执行）

```bash
env PYTHONDONTWRITEBYTECODE=1 HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1 timeout --signal=TERM --kill-after=10s 600s /home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python -u /yinghepool/tech-dcu/zhaoyao/tools/fixed_card_patch_20261102/verify.py --output /yinghepool/tech-dcu/zhaoyao/tools/fixed_card_verify_171_01
```

600秒是上限，不是预计耗时。先跑原版、再跑真正候选池；同32条、batch4、准备缓存相同。失败即停止，原生产OOM等待和拆批降级不作为通过结果。
主进程不加载GPU模型。报告必须completed、candidate加载2次/切卡0次、2个活跃worker分别占用两卡、features_allclose=true。报告同时保留精确相等和最大误差。
测试仅验证短序列两卡编码；未修复长序列梯度状态、异常重试重复结果、缓存失效或跨切分池重建。
不自动执行14卡测试、正式文件替换或回滚。正式替换步骤在验收后单独给出。
