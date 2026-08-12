# TR-RD 软级联方法

## 两条独立支路

TR 支路直接借鉴 `K:\radar\main` 的 B01 Transformer 航迹模型。每条航迹输入 15 维序列，并计算 22 维物理统计特征；序列编码和物理编码在 TR 支路内部融合后输出五分类 logits。加载的权威 checkpoint 是 **TR-only**，不包含 RD 输入。

RD 支路使用当前项目的 RD CNN。RD 的每一帧先独立得到五分类 logits，再按 `trajectory_id` 聚合为航迹级概率。聚合使用同一航迹内帧概率的平均值，同时保存帧数、帧间一致性和是否有 RD 证据。

## 航迹级软级联

两条支路在同一固定 grouped split、同一航迹粒度上计算概率。固定诊断模式使用每类 RD 权重：

`p_fused,c = alpha_c * p_RD,c + (1 - alpha_c) * p_TR,c`

`alpha_c` 可以是单个固定值、五个类别值，或由质量感知门控网络按航迹动态产生。门控输入包含两路概率、熵、top-1/top-2 margin、RD 帧间一致性和 `log(1 + frame_count)`。RD 不可用时权重强制为 0。

推理结果不会覆盖任一支路：`trajectory_decisions.jsonl` 对每条航迹同时记录真值、TR 判断、RD 判断、融合判断、三套概率、RD 质量和 rescue/harm 标记；`metrics.json` 分别给出 TR、RD、融合三套指标与混淆错例编号。

## 门控训练协议

质量感知门控只能用训练集 OOF 分数拟合。`scripts/train_fusion_gate.py` 要求元数据显式包含：

- `score_origin: OOF`
- `source_partition: train`
- `split_sha256`
- `tr_checkpoint` 与 `rd_checkpoint`

验证集和测试集分数会被拒绝，避免门控泄漏。门控 checkpoint 保存 `fusion_state`、OOF 文件哈希、配置和训练历史。当前固定权重 `0.2` 的结果只用于诊断，不视为正式学习型融合结果。

## 评价约束

默认只评价固定 F split 的 validation（1084/232/233）。测试集必须显式传 `--allow-test`，且仅在模型和门控方案筛选完成后执行。横向比较前必须检查 RD 训练成员是否与 F split 的 train 完全一致。
