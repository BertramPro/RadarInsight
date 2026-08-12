# RD 空间结构与局部速度对比消融方案

## Material Passport

- ID: `RD-SPATIAL-CONTRAST-ABLATION-20260807`
- Type: RD-only controlled ablation plan
- Origin Skill: experiment-agent
- Origin Mode: plan
- Verification Status: UNVERIFIED
- Status: `PLANNED`
- Single seed: `42`

## 1. Research Question

验证 Bird/Other（Other 对应 Unknown）混淆是否主要来自：

1. 末端全局平均池化丢失 RD 空间布局；
2. 单通道 RD 没有显式突出窄速度结构与局部速度背景的差异。

## 2. Reference Baseline

宽度选择参照为最新的 `audit_rd_vr360_w900_lr005_seed42_cached`：

- 输入：单通道 `31 x 900` RD；`Vr=-90~89 m/s`
- 重采样：`db_linear`
- 归一化：`global_z`
- 学习率：`0.05`
- 最佳验证轮次：30
- 验证航迹级 Macro-F1：`0.849864`

720 宽度的对应结果为 `0.842125`，因此后续结构/通道消融固定在 900 宽度。注意 900 相对 360 历史基线 `0.850375` 仍是近似持平，不应写成整体性能提升。

新实验必须使用同一轨迹切分和训练帧采样规则。直接复用 900 宽度本地缓存，供两项实验共用；缓存只含训练采样帧和验证帧，不含测试帧。

切分来源固定为 [900 宽度对照 split.json](H:\RadarInsight\artifacts\audit_rd_vr360_w900_lr005_seed42_cached\split.json)，不得在两项实验之间重新随机切分。

## 3. Fixed Protocol

| Parameter | Fixed value |
|---|---:|
| epochs | 50 |
| batch size | 128 |
| workers | 0 |
| max train frames / trajectory | 32 |
| normalization samples | 2048 |
| learning rate | 0.05 |
| weight decay | 0.0001 |
| patience | 10 |
| seed | 42 |
| velocity interval | -90 to 89 m/s |
| target size | 31 x 900 |
| resampling | db_linear |
| normalization | global_z |
| augmentation | off |
| split | random_stratified |
| test evaluation | skipped |

Every experiment has an independent output directory. No checkpoint resume and no multi-seed rerun are used in this ablation stage.

## 4. Experiments

### R1: Spatial-Preserving Classification Head

ID: `rd_ablation_R1_spatial_pool_vr360_w900_lr005_seed42`

Only the classification head changes. Keep the first three convolutional blocks, batch normalization, activations and max-pooling unchanged. Replace:

```text
AdaptiveAvgPool2d(1, 1) -> Flatten -> Linear(128, 5)
```

with:

```text
AdaptiveAvgPool2d(2, 8) -> Flatten -> Dropout(0.25) -> Linear(128*2*8, 5)
```

The `2 x 8` grid preserves coarse RD-row position and velocity-band layout while keeping the model small. No extra input channel, loss weight, sampler or augmentation is introduced.

Hypothesis: preserving the location and width of the near-zero-Doppler response will reduce both `Bird -> Other` and `Other -> Bird` errors.

### R2: RD + Local Velocity Contrast Channel

ID: `rd_ablation_D4_local_contrast_vr360_w900_lr005_seed42`

Only the input mode changes from `rd` to `rd_contrast`; the original CNN structure remains unchanged. The input has two channels:

```text
channel 1 = global_z(RD)
channel 2 = clip((RD - local_velocity_mean(RD)) / global_std, -5, 5)
```

The local velocity mean is a centered moving average along the velocity axis. Its width is approximately 4% of the target width, corresponding to about 7.2 m/s for the common interval:

- 900 columns: 37-column window.

The second channel is computed from the physical dB RD before normalization. Existing RD caches can be reused; no second MAT conversion is needed.

Hypothesis: explicitly exposing narrow velocity peaks and local spectral contrast will improve Bird/Other separation, especially on Other samples that resemble Bird.

## 5. Execution

Run R1 and R2 concurrently under the scheduler, with the same 900 cache and fixed protocol. Confirm before launch:

- both commands contain no `--resume`;
- both use seed 42 and `H:\RadarInsight\cache\rd_vr360_w900_db_linear_seed42_train32`;
- R1 has `input_mode=rd` and the spatial head variant;
- R2 has `input_mode=rd_contrast` and the original head;
- both begin at epoch 1.

## 6. Primary and Secondary Metrics

Primary metric: validation trajectory-level Macro-F1.

Secondary metrics:

- Bird F1 and recall;
- Other/Unknown F1 and recall;
- `Bird -> Other` and `Other -> Bird` counts;
- `Other -> Clutter` count;
- overall trajectory accuracy and per-class confusion matrix;
- best epoch, training duration, peak memory and GPU utilization.

The test set remains unused until the design is selected. Because this is one seed and one run per experiment, differences are exploratory and should not be reported as statistically stable effects.

## 7. Decision Rule

Prefer a candidate when it improves validation Macro-F1 without materially reducing Drone, Balloon or Clutter F1, and reduces the direct Bird/Other confusion sum:

```text
Bird -> Other + Other -> Bird
```

If the direct pair confusion decreases but `Other -> Clutter` increases, report the tradeoff instead of claiming an overall solution. If both candidates are inconclusive, proceed to the planned hard-trajectory sampler or trajectory-level temporal aggregation rather than adding more class-loss weight blindly.
