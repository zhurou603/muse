# A-test 编码失败最小复现包

本目录包含 step 4100 完整 A-test 中全部 **1846 条编码失败**的独立复现单元。

## 运行

```bash
python reproduce.py --output verification.json
```

只复现单个数据集或单条记录：

```bash
python reproduce.py --dataset QM9 --output qm9-verification.json
python reproduce.py --id '<record_id>' --output one-case-verification.json
```

退出码为 0 表示所有选中用例都复现了原错误；若有未复现项则退出码为 1，并在输出中列出。运行不需要模型权重，也不需要执行完整 A-test。

## 内容

- `failures.jsonl`：所有失败的数据集、子集、split、ID、原错误、原始记录 hash 和结构文件位置。
- `records/`：按数据集/子集/split 拆分的原始输入。**labels/targets 已剥离**，禁止将这些输入用于训练或选模。
- `structures/`：QM9 原始 SDF 结构块和 GEOM 对应官方构象的 RDKit 二进制对象。
- `reproduce.py`：独立复现入口。
- `vendor/structures.py`：与生成包时一致的严格结构对齐实现。
- `manifest.json`：数量、环境、源 checkpoint 和源生产代码 hash。
- `CHECKSUMS.sha256`：包内全部文件的 SHA256。
- `verification.json`：交付前实际复现结果。

## 错误范围

- RDKit 无法解析的 SMILES。
- QM9 严格 SDF 解析或结构对齐失败。
- GEOM 官方构象图与 SMILES 表示不一致。
- DBAASP 当前路由不支持的修饰肽记号。

修复时不得强行对齐不同分子式或连通性，不得用 validation/test 标签训练、校准或选模。修复后应先用本包验证，再运行完整 90 项 A-test。

## 代码管理

生成器为 `build_handoff.py`。重新生成时必须有完整的六类定向失败复现产物，并检查总数恰好为 1846。任何代码或数据变更都会改变 `manifest.json` / `CHECKSUMS.sha256`，请保留新的验证结果，不覆盖历史证据。
