# RD 五分类实验：当前重要结论

## Material Passport

- ID: `RD-FIVECLASS-STRUCTURE-20260806`
- Type: 数据核查与实验设计记录
- Status: `ANALYZED`
- Scope: 仅使用 MAT 文件中的 `data_proc_MTD_result_db` 和配套 `Vr`
- Dataset root: `K:\23所雷达数据\CQ-08中国航天科工二院二十三所-低空监视雷达目标智能识别技术研究数据集`
- Scan date: 2026-08-06
- MAT files scanned: 106,026
- Read errors: 0
- Validation: all `Vr` arrays were monotonic increasing

## 1. Classification target

The five classes are defined as follows:

| Model class | Original target label | Target type |
|---|---:|---|
| `drone` | 1 and 2 merged | Drone |
| `bird` | 3 | Bird |
| `balloon` | 4 | Balloon |
| `clutter` | 5 | Clutter |
| `other` | 6 | Other |

The MAT filename identifies the target type and trajectory identifier. The trajectory identifier is consistent with the point-track and track files according to the user’s confirmation.

## 2. Full RD shape and velocity-coordinate inventory

The RD matrix has 31 range bins along its first dimension. The second dimension is the number of velocity samples and equals `len(Vr)`. There are five widths, not three, and 12 distinct combinations of width and velocity range.

| RD shape | `Vr` range (m/s) | Count | Share |
|---|---:|---:|---:|
| `31×142` | -171.703297 to 169.284940 | 403 | 0.38% |
| `31×142` | -148.426677 to 146.336161 | 408 | 0.38% |
| `31×226` | -240.384615 to 238.257318 | 15,606 | 14.72% |
| `31×226` | -196.540881 to 194.801581 | 16,802 | 15.85% |
| `31×226` | -100.160256 to 99.273882 | 84 | 0.08% |
| `31×226` | -90.220137 to 89.421729 | 84 | 0.08% |
| `31×360` | -100.160256 to 99.603811 | 22,960 | 21.66% |
| `31×360` | -92.489826 to 91.975994 | 16,095 | 15.18% |
| `31×570` | -100.160256 to 99.808817 | 16,792 | 15.84% |
| `31×570` | -92.489826 to 92.165300 | 11,862 | 11.19% |
| `31×900` | -100.160256 to 99.937678 | 2,847 | 2.69% |
| `31×900` | -92.489826 to 92.284293 | 2,083 | 1.96% |
| **Total** | — | **106,026** | **100.00%** |

Width totals:

| Width | Count | Share |
|---:|---:|---:|
| 142 | 811 | 0.77% |
| 226 | 32,576 | 30.73% |
| 360 | 39,055 | 36.84% |
| 570 | 28,654 | 27.03% |
| 900 | 4,930 | 4.65% |
| **Total** | **106,026** | **100.00%** |

## 3. Important interpretation

Width and physical velocity range are related but are not interchangeable:

- `RD.shape[1] == len(Vr)`;
- a larger width can mean denser velocity sampling, not necessarily a wider physical range;
- the same width can cover different physical ranges;
- for example, `31×226` occurs with approximately `-240~238 m/s`, `-196~195 m/s`, `-100~99 m/s`, and `-90~89 m/s` ranges;
- therefore column index alone is not a valid physical velocity coordinate.

The model preprocessing must use each file’s `Vr`, not just resize the matrix by column index.

## 4. Recommended R1 preprocessing

The first robust RD baseline should define a common physical velocity interval and then resample every file onto the same grid:

1. Keep the 31 range bins unchanged.
2. Define a common interval covered by all samples, approximately `-90~+89 m/s` based on the inventory.
3. Create one fixed target velocity grid with 360 points in that interval.
4. For each file, use its own `Vr` as the source x-coordinate and resample each of the 31 RD rows onto that target grid.
5. Produce one-channel tensors of shape `1×31×360`.
6. Values outside the source interval must not be extrapolated into R1; only the common physically observed interval is used.

Conceptually:

```text
31×142, 31×226, 31×360, 31×570, 31×900
        + each file's own Vr coordinate
                    ↓
       common velocity interval -90~+89 m/s
                    ↓
             fixed 31×360 RD tensor
```

For a production-quality implementation, perform anti-alias filtering or bin averaging before downsampling wider grids (`570` or `900` to `360`). Linear interpolation is acceptable for an initial baseline, but it cannot create information for narrower grids (`142` or `226`); it only estimates values between measured samples.

## 5. Information-retention control experiment

R1 intentionally tests a common interval and fixed input shape. It discards velocity content outside the common interval. To determine whether that discarded content is useful, define a follow-up control:

- use the union velocity interval (approximately `-240~+238 m/s`);
- resample to a fixed width such as 512 or 720;
- add a second channel indicating whether each target velocity position was observed in the original file;
- report results by velocity configuration as well as overall Macro-F1.

This control distinguishes genuine classification improvement from an artifact of velocity-range normalization.

## 6. Training protocol and current execution status

The intended R1 classifier is a light 2D CNN applied to individual RD frames. Frame probabilities from one trajectory are averaged to obtain the trajectory-level prediction. Splits are by trajectory ID, so frames from one trajectory cannot cross train/validation/test sets.

The initial implementation generated a leakage-free split:

- train/validation/test trajectories: 1,084 / 232 / 233;
- train/validation/test frames: 73,323 / 16,553 / 16,150;
- sampled training frames: 34,514;
- training normalization mean/std: 56.188 / 6.999.

The first training attempt failed before the first epoch because raw RD widths were passed directly to a batch. The error was:

```text
stack expects each tensor to be equal size
```

No model result should be interpreted from that run. The next implementation step is to add the `Vr`-based crop and resampling described above, then rerun the same seed and split.

## 7. Open risks / checks

- Verify the exact common interval endpoints from all files before implementation rather than relying only on rounded inventory values.
- Check whether the RD values are treated as dB or linear power before choosing interpolation versus power-domain averaging.
- Measure how much target energy lies outside `-90~+89 m/s` by class and velocity configuration.
- Report per-configuration metrics; aggregate Macro-F1 alone may hide domain shift.
- Keep the failed raw-size run as a diagnostic artifact; do not treat it as a baseline result.
