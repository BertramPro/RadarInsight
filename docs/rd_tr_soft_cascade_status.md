# 当前 TR-RD 软级联状态

本次实现已完成两条独立支路的迁移和航迹级软级联封装：

- TR：B01 Transformer 航迹分支，固定 F split validation，`Accuracy=93.10%`、`Macro-F1=86.95%`。
- RD：R2 Contrast + 900 列 RD 分支，先做同航迹多帧概率平均，`Accuracy=91.38%`、`Macro-F1=85.26%`。
- 固定 RD 权重 0.2：`Accuracy=93.10%`、`Macro-F1=86.95%`，本次没有相对 TR 的 rescue，也没有 harm。

结果文件：

`artifacts/fusion_b01_r2w900_fixed02_fsplit_val/metrics.json`

`artifacts/fusion_b01_r2w900_fixed02_fsplit_val/trajectory_decisions.jsonl`

当前固定融合只是诊断基线，不能宣称已经带来增益。下一步若要启用质量感知类别门控，必须先按 [rd_tr_soft_cascade.md](rd_tr_soft_cascade.md) 生成训练集 OOF 分数，再运行 `scripts/train_fusion_gate.py`；验证集只用于方案筛选，测试集暂不参与门控拟合。
