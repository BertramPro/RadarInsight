# 原项目 B01 与当前 TR-only 复现对照

**验证状态：VERIFIED**。在固定 F split validation 上，当前结果与原 B01 的 232 条逐航迹预测完全一致。

## 对照范围

- 原项目：`K:\radar\main\artifacts\f_protocol\20260728-183147\b01_transformer_seed42\evaluation_report.json`
- 当前项目：`artifacts/tr_only_b01_fsplit_reproduction/validation_best.json` 与 `trajectory_decisions.jsonl`
- 评估划分：固定 F split 的 validation，共 232 条航迹；测试集未启用。
- 类别顺序：`DroneTarget / BirdTarget / BalloonTarget / ClutterTarget / UnknownTarget`。
- 当前项目的 `other` 与原项目的 `UnknownTarget` 是同一类别，以下统一写作 `Unknown`。

## 混淆矩阵

行是真实类别，列是预测类别：

| 真实 \\ 预测 | Drone | Bird | Balloon | Clutter | Unknown |
|---|---:|---:|---:|---:|---:|
| Drone | 102 | 2 | 0 | 1 | 0 |
| Bird | 2 | 46 | 0 | 0 | 4 |
| Balloon | 0 | 1 | 51 | 0 | 0 |
| Clutter | 0 | 1 | 0 | 7 | 0 |
| Unknown | 0 | 4 | 0 | 1 | 10 |

原 B01 与当前复现的矩阵逐项一致，没有出现某一类样本排序或标签映射导致的偏移。

## 指标对照

| 指标 | 原 B01 | 当前 TR-only | 差值 |
|---|---:|---:|---:|
| Accuracy | 0.9310345 | 0.9310345 | 0 |
| Macro-F1 | 0.8694954 | 0.8694954 | 0 |
| Drone F1 | 0.9760766 | 0.9760766 | 0 |
| Bird F1 | 0.8679245 | 0.8679245 | 0 |
| Balloon F1 | 0.9902913 | 0.9902913 | 0 |
| Clutter F1 | 0.8235294 | 0.8235294 | 0 |
| Unknown F1 | 0.6896552 | 0.6896552 | 0 |

## 全部错分航迹

以下编号均为 `cq08|track|编号` 中的航迹编号：

| 真实 → 预测 | 数量 | 航迹编号 |
|---|---:|---|
| Drone → Bird | 2 | 3960, 4238 |
| Drone → Clutter | 1 | 1258 |
| Bird → Drone | 2 | 528, 1301 |
| Bird → Unknown | 4 | 845, 851, 2367, 2956 |
| Balloon → Bird | 1 | 3109 |
| Clutter → Bird | 1 | 2903 |
| Unknown → Bird | 4 | 2119, 2709, 2862, 4171 |
| Unknown → Clutter | 1 | 3878 |

总错分数为 16/232，正确数为 216/232。

## Bird 与 Unknown 重点分析

- Bird 真值共 52 条，正确 46 条；错分 6 条，其中 2 条判为 Drone（528、1301），4 条判为 Unknown（845、851、2367、2956）。Bird recall 为 `46/52 = 0.8846`。
- Unknown 真值共 15 条，正确 10 条；错分 5 条，其中 4 条判为 Bird（2119、2709、2862、4171），1 条判为 Clutter（3878）。Unknown recall 为 `10/15 = 0.6667`。
- Bird 与 Unknown 之间的双向错分为 8 条：Bird→Unknown 4 条、Unknown→Bird 4 条，净数量为 0，但两类边界都不稳定。
- 若按真实类别统计，Bird/Unknown 自身的错分共 11 条（Bird→Drone 2、Bird→Unknown 4、Unknown→Bird 4、Unknown→Clutter 1），是当前主要误差来源；其中最直接的 Bird↔Unknown 错分占 8 条。
- 若按错分两端只要涉及 Bird/Unknown 统计，还包括 Drone→Bird 2、Balloon→Bird 1、Clutter→Bird 1，因此共有 15 条；剩余 1 条为 Drone→Clutter。

## 是否发生了复现差异

按航迹编号建立索引后，原项目 `predictions[].label_name_true/label_name_pred` 与当前 `trajectory_decisions.jsonl` 的 `true_label/prediction_label` 对比结果为：

- 航迹 ID 交集：232/232
- 真实标签差异：0
- 预测标签差异：0
- 错分集合差异：0

因此当前页面或当前项目看到的 B01 混淆矩阵，和原项目 B01 是同一组 validation 预测，不是重新训练后偶然得到的近似结果。后续分析 Bird/Unknown 时，可以直接使用上述 13 条重点错分航迹作为固定样本集。
