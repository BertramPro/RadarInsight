# 前端实验接口

监控服务保留原有的 `GET /api/trainings` 数组接口，同时提供统一接口：

- `GET /api/experiments`：返回 `{api_version: 2, items: [...], generated_at}`。
- `GET /api/experiments/{name}`：返回单个实验 `{api_version: 2, item}`。

每个实验条目新增：

- `experiment_type`：`rd_only`、`tr_only`、`tr_rd_soft_cascade` 或 `unknown`；
- `branches`：TR、RD、fusion 是否启用及角色；
- `evaluation`：分区、指标来源、主指标和是否允许测试集；
- `outputs`：实际存在的配置、指标、checkpoint 和逐航迹判断文件。

前端分类页根据 `experiment_type` 渲染：

- `tr_only`：显示 TR-only 验证指标、分类报告、混淆矩阵和 checkpoint 来源；
- `rd_only`：沿用 RD 训练进度、曲线和 RD 指标；
- `tr_rd_soft_cascade`：并列显示 TR、RD、最终融合三套结果，并按对应分支展示混淆错例。

这样不会把 TR-only 实验误显示成 RD 配置，也不会把融合结果覆盖成任一单支路结果。
