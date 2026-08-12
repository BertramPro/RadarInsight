# 论文摘要 v1

## 中文摘要

针对外辐射源雷达低慢小目标中无人机、飞鸟及未知目标易混淆的问题，提出一种基于航迹运动特征与距离-多普勒（range-Doppler，RD）图像双域证据的五分类识别框架。以航迹为基本评价单位，在固定 F split 和分组防泄漏协议下，分别构建 B01-compatible Transformer 航迹分支和 RD 卷积神经网络分支，并将同一航迹内多帧 RD 概率聚合为航迹级证据。进一步比较固定权重融合与质量感知类别门控，分别保存两条支路及融合结果，用 rescue/harm 统计分析错误互补性。验证集上，TR-only 基线逐航迹复现原 B01，Accuracy 和 Macro-F1 分别为 93.10% 和 86.95%；R2-900 RD 分支的 Macro-F1 为 85.26%，固定权重融合未超过 TR 基线。一组未参与门控校准的测试集诊断中，RD-only Macro-F1 为 87.46%，验证集校准门控为 86.13%，尚未证明门控融合具有稳定净增益。Bird/Unknown 归因分析显示，错误可能同时表现为 Bird 响应过宽和 Unknown 呈现 Bird-like 窄尖响应。研究结果表明，双域融合的有效性取决于错误互补性、概率校准和防泄漏训练协议，而不能由特征叠加本身保证。

**关键词：** 外辐射源雷达；低慢小目标；航迹运动特征；距离-多普勒图；航迹级融合；质量感知门控

## English Abstract

Low-slow-small radar targets such as drones, birds, clutter, and unknown objects often exhibit overlapping kinematic and echo patterns. This study develops a five-class recognition framework that evaluates trajectory motion features and range-Doppler (RD) maps as two aligned evidence branches. A B01-compatible Transformer trajectory branch and an RD convolutional branch are implemented under a fixed grouped split without cross-partition leakage. Frame-level RD probabilities are aggregated to the trajectory level before fusion. Fixed-weight fusion and quality-aware classwise gating are compared while preserving the independent decisions of both branches. On the validation split, the migrated trajectory branch exactly reproduces the original B01 trajectory-level results, reaching 93.10% accuracy and 86.95% Macro-F1. The R2-900 RD branch obtains 85.26% Macro-F1, while fixed-weight fusion does not exceed the trajectory baseline. In a held-out test diagnostic whose samples were not used for gate calibration, RD-only reaches 87.46% Macro-F1 and validation-calibrated gating reaches 86.13%, providing no evidence of a stable net gain over RD. Bird/Unknown attribution suggests two opposite error patterns: overly broad Bird-like responses and narrow, peaked Unknown responses. The findings indicate that dual-domain fusion depends on error complementarity, probability calibration, and leakage-controlled training rather than feature concatenation alone.

**Keywords:** passive radar; low-slow-small targets; trajectory motion features; range-Doppler map; trajectory-level fusion; quality-aware gating
