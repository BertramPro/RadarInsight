# RD 五分类实验

仅使用雷达数据集 MAT 文件中的 `data_proc_MTD_result_db`，在航迹级进行五分类：无人机、鸟、气球、杂波、其他。

## 首个实验：R1

R1 是轻量 2D CNN 单帧分类基线。训练阶段采用每航迹最多固定数量的均匀采样帧；评估阶段对测试或验证航迹中的所有帧预测概率取平均，得到航迹级类别。

训练、验证和测试均按照航迹 ID 划分。绝不允许同一航迹的 MAT 帧跨集合。

```powershell
& 'C:\Users\Surfa\AppData\Local\Programs\Python\Python39\python.exe' -m radar_rd.train --dataset-root 'K:\23所雷达数据\CQ-08中国航天科工二院二十三所-低空监视雷达目标智能识别技术研究数据集' --output-dir artifacts\r1_rd_cnn_seed42 --epochs 50 --seed 42
```

输出包含 `manifest.csv`、`split.json`、训练集归一化统计量、每轮指标、最优模型、验证/测试集航迹级指标和混淆矩阵。
