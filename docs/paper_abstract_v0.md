# 双语摘要初稿

## English Abstract

Low-slow-small radar targets such as unmanned aerial vehicles, birds, balloons, clutter, and unknown objects often exhibit overlapping motion and echo characteristics, making five-class recognition difficult. This study extends track-motion-based UAV and bird recognition by introducing a dual-domain framework that combines radar track sequences with range-Doppler (RD) images. A B01-compatible trajectory branch encodes a 15-dimensional track sequence together with 22 physical statistics, while an RD branch aggregates frame-level probabilities into trajectory-level evidence. The two branches are aligned on a fixed grouped split and evaluated separately before fixed-weight and quality-aware gated fusion. Experiments use 1,084, 232, and 233 trajectories for training, validation, and testing, respectively. On the fixed validation split, the TR-only branch reproduces the authoritative B01 result with 93.10% accuracy and an 86.95% Macro-F1, whereas the R2-900 RD branch achieves 91.38% accuracy and an 85.26% Macro-F1. A fixed RD weight of 0.2 reaches the same overall validation metrics as TR-only and does not produce an observable net rescue in the current diagnostic run. Bird/Unknown analysis indicates two directional error patterns: Bird-to-Unknown cases tend to show broader velocity responses, while a few Unknown-to-Bird cases exhibit narrow and sharp Bird-like responses. The results demonstrate the importance of trajectory-level alignment, leakage-controlled gating, and class-specific error analysis rather than assuming that multimodal fusion is always beneficial.

**Keywords**: passive radar; low-slow-small targets; trajectory recognition; range-Doppler image; gated fusion; class-wise error analysis

---

## 中文摘要

针对无人机、飞鸟、气球、杂波和未知目标在低慢小探测场景中存在运动特征与回波结构相似、类别边界易混淆的问题，本文在航迹运动特征识别无人机与飞鸟的研究基础上，引入距离-多普勒图信息，构建航迹与 RD 双域特征识别框架。航迹分支采用与 B01 协议兼容的 Transformer 结构，同时编码 15 维航迹序列和 22 维物理统计特征；RD 分支先进行帧级分类，再将同一航迹内的多帧概率聚合为航迹级 RD 证据。两条支路在固定分组划分上分别评价，并进一步比较固定权重和质量感知门控融合。实验采用 1084、232 和 233 条航迹作为训练、验证和测试划分。固定验证集结果表明，TR-only 分支准确率为 93.10%，Macro-F1 为 86.95%，成功复现原 B01 结果；R2-900 RD 分支准确率为 91.38%，Macro-F1 为 85.26%；固定 RD 权重 0.2 的融合结果与 TR-only 相同，当前诊断中未观察到净 rescue。Bird/Unknown 分析发现，Bird→Unknown 错例更倾向于表现为较宽的速度响应，而少量 Unknown→Bird 错例具有窄而尖锐的 Bird-like 响应。研究结果表明，双域融合的有效性取决于两路错误的互补程度、航迹级对齐和门控防泄漏设计，不能预先假设融合必然提升识别性能。

**关键词**：外辐射源雷达；低慢小目标；航迹识别；距离-多普勒图；门控融合；类别级错分分析

## 摘要质量检查

| 项目 | English | 中文 |
|---|---:|---:|
| 长度 | 约 212 words | 约 430 字 |
| 背景、目的、方法、结果、意义 | 5/5 | 5/5 |
| 关键词数量 | 6 | 6 |
| 是否包含未经验证的性能提升 | 否 | 否 |
| 是否包含引用 | 否 | 否 |

