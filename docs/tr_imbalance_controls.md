# TR 类别不均衡控制：实现契约

本文件适用于 `scripts/train_tr_only.py` 的 CQ-08 TR-only 训练。

## 三条独立主线

1. **采样层（sample boost）**

   `WeightedRandomSampler(replacement=True)` 先为扩增后的每一类计算
   每条记录的基础权重 `1 / n_class`，再乘 Bird、Clutter、Unknown 的
   extra sample boost。boost=`1` 只表示不附加偏置，基础逆频率采样仍在。

2. **损失层（class loss weight）**

   `inverse_sqrt` 与 `class_balanced` 二选一，均根据扩增后的训练记录
   计数计算五类交叉熵权重。可选 `clutter_loss_weight` 只覆盖 Clutter
   那一项，并不是另一条独立路线。

3. **数据层（train-only 虚拟副本）**

   `unknown_augmentation_copies` 和 `bird_augmentation_copies` 是**确切
   追加副本数**。副本仅加入本次训练的内存记录列表；原始 CSV、grouped
   split、validation 和 test 均不改动。副本经连续删帧、整轨幅度缩放和
   整轨 SNR 平移后重新运行 `encode_track`，因此重新得到 15 维序列和
   22 维物理特征。

## 计数派生项

数据层追加副本后，采样层的 `1/n_class`、损失层的类别权重，以及 source
normalization 统计量都会基于扩增后的训练记录重新计算。故“只验证数量扩增”
的规范设置是：固定全部 extra sample boost、固定一种 class loss rule，
只改变追加副本数；报告须说明以上派生计数会随之重算。

训练 `config.json` 会记录：原始/扩增后类别数、实际采样类别概率、最终类别
损失权重和副本扰动参数，供复核。
