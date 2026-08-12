# RD 训练中断记录：学习率 0.01 对照实验

- 原实验：`r1_rd_cnn_vr360_lr001_seed42`
- 发生日期：2026-08-06
- 状态：中途失败，保留原始产物
- 已完成：9 个 epoch；第 10 个 epoch 训练到中途
- 最佳验证 Macro-F1：0.8055606666（epoch 8）
- 最佳验证准确率：0.8922413793（epoch 8）
- 无最终测试集结果

## 根因

训练程序的旧版 `write_progress()` 先写入 `progress.json.tmp`，再通过
`Path.replace()` 替换 `progress.json`。HTML 监控页面同时读取该文件时，Windows
拒绝替换操作并抛出：

```text
PermissionError: [WinError 5]
progress.json.tmp -> progress.json
```

该错误属于训练进度文件的并发读写冲突，不是模型、数据、CUDA 或学习率导致的训练发散。

## 修复

当前 `radar_rd/train.py` 已改为直接写入 `progress.json`；监控端将不完整 JSON
视为瞬时状态并在下一次刷新时重试。重试实验使用独立输出目录，避免覆盖本次失败现场。
