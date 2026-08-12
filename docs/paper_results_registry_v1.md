# 论文主结果注册表 v1

本文件用于约束论文正文使用的实验结果，避免混用不同 checkpoint、RD 预处理、划分或门控来源。所有结果均按航迹级统计，类别顺序为 `Drone / Bird / Balloon / Clutter / Unknown`。

## 1. 固定 F split

- split 文件：`K:\radar\main\data\manifests\cq08_grouped_split_f.json`
- split SHA-256：`c9424d7b9b7ec7373fdd7989b66e30c9ce11dc0d2de8bf04172ebfa48758667c`
- 分区数量：Train 1084、Validation 232、Test 233
- 原始类别计数：

| 分区 | Drone | Bird | Balloon | Clutter | Unknown |
|---|---:|---:|---:|---:|---:|
| Train | 489 | 245 | 245 | 35 | 70 |
| Validation | 105 | 52 | 52 | 8 | 15 |
| Test | 105 | 53 | 53 | 7 | 15 |

## 2. 已验证主基线

### TR-only B01 validation

- 来源：`docs/b01_tr_only_error_comparison.md`
- 当前项目与原项目逐航迹预测完全一致
- Accuracy：`0.9310345`
- Macro-F1：`0.8694954`
- F1：Drone `0.9760766`、Bird `0.8679245`、Balloon `0.9902913`、Clutter `0.8235294`、Unknown `0.6896552`
- Bird→Unknown：4；Unknown→Bird：4

### RD-only R2-900 validation

- 来源：`docs/rd_tr_soft_cascade_status.md`
- 共同速度区间：`[-90, +89] m/s`
- 统一输出：`31×900`
- 输入：`rd_contrast`
- 重采样：`db_linear`
- 归一化：`global_z`
- Accuracy：`0.9138`
- Macro-F1：`0.8526`

### 固定权重融合 validation

- RD 权重：`0.2`
- Accuracy：`0.9310`
- Macro-F1：`0.8695`
- 结论：与 TR-only 相同，不能写成融合提升。

## 3. 最新配对测试结果

- artifact：`artifacts/fusion_202608122222/test_trajectory_metrics.json`
- 配置：`quality_classwise`
- TR checkpoint：`artifacts/tr_202608122127_rerun/best.pt`，epoch 30
- RD checkpoint：`artifacts/rd_ablation_R2_contrast_w900_registry_seed42_rerun/best.pt`，epoch 44
- 门控 checkpoint：`artifacts/gate_cal_202608122218/fusion_gate.pt`
- 门控校准来源：Validation；测试集未用于校准
- 测试航迹数：233

| 方法 | Accuracy | Macro-F1 | Drone F1 | Bird F1 | Balloon F1 | Clutter F1 | Unknown F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TR-only | 0.9185 | 0.8169 | 0.9750 | 0.8214 | 0.9811 | 0.5882 | 0.5333 |
| RD-only | 0.9485 | 0.8746 | 0.9809 | 0.8762 | 0.9907 | 0.6667 | 0.8235 |
| 质量感知门控 | 0.9528 | 0.8939 | 0.9858 | 0.9412 | 0.9811 | 0.8000 | 0.7500 |

错误互补统计：同时正确 208 条；仅 RD 正确 13 条；仅 TR 正确 6 条；同时错误 6 条；门控相对 TR 修正 9 条、引入误判 1 条，净变化 +8 条。

## 4. 证据等级与正文写法

1. B01 validation 复现：可作为论文中的可靠基线结果。
2. R2-900 validation：可作为 RD 受控结果，但应绑定完整预处理协议。
3. 固定权重 validation：只能说明当前权重没有超过 TR，不能宣称融合提升。
4. 最新测试结果：测试集未参与门控校准，属于“验证集校准后的独立测试评估”；它不是严格 OOF 门控结果，正文不应写成 OOF 泛化结论。
5. 分区内虚拟航迹扩增指标：只能作为诊断结果，不得替代原始 validation/test 主指标。

## 5. 当前阻塞项

- 若要形成严格 OOF 证据，仍需另行用训练集 OOF 分数拟合门控，并在同一 Test 233 条航迹上完成配对评估。
- 需要逐条核验 [3]、[5]、[7]、[8] 的正式书目信息和 DOI。
- 在 OOF 门控结果确定前，不应在摘要或结论中写“融合显著提升”。
