# RD Bird/Other Feature Attribution Report

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-07
- Verification Status: UNVERIFIED
- Version Label: rd_attribution_v1

## Scope

- Model: `r1_rd_cnn_vr360_lr005_seed42`
- Checkpoint: validation-best epoch 17
- Input: 31 x 360 RD, Vr = -90 to 89 m/s
- Validation trajectories: 232
- Attribution: Grad-CAM on the last convolution layer, aggregated at trajectory level
- Label mapping: the dataset/model label `other` corresponds to the user's Unknown category

## Confusion Counts Used

| Group | Trajectories | Sampled frames |
|---|---:|---:|
| Bird correct | 46 | 288 |
| Bird -> Other | 2 | 48 |
| Other correct | 9 | 216 |
| Other -> Bird | 2 | 48 |
| Other -> Clutter | 3 | 72 |

## Main Findings

1. The Bird decision is dominated by a narrow response around zero Doppler. The two central velocity deciles contain 34.3% of the Grad-CAM mass for correctly classified Bird trajectories.
2. Other -> Bird samples reproduce the Bird pattern: 32.2% of their attribution lies in the two central deciles, with a peak near -1.25 m/s.
3. Correct Other samples use a broader velocity distribution. Only 20.3% of their attribution lies in the two central deciles, and their mean absolute attributed velocity is 44.5 m/s versus 36.5 m/s for correct Bird.
4. Bird -> Other samples shift toward the correct-Other profile. Their mean absolute attributed velocity rises to 41.3 m/s, while the lowest RD-row quartile contributes 34.2%, close to correct Other at 35.9% and clearly above correct Bird at 26.0%.
5. Other -> Clutter is a separate failure mode. Its strongest distinction is row position: the highest RD-row quartile contributes 32.2%, versus 11.3% for correct Other. It should not be treated as the same problem as Bird/Other confusion.

## Interpretation

The model appears to use a narrow near-zero-Doppler structure as a Bird cue and a broader, lower-row response as an Other cue. Misclassification occurs when an Other trajectory contains the narrow central structure, or when a Bird trajectory lacks it and presents a broader response.

The architecture likely amplifies this overlap. `SmallRDCNN` ends with `AdaptiveAvgPool2d((1, 1))`, which collapses spatial layout before classification. This favors the presence and overall strength of local patterns, while discarding much of their detailed RD arrangement. Bird and Other may therefore become hard to separate when their dominant spectral structures overlap.

## Limitations

- Bird -> Other and Other -> Bird each contain only two validation trajectories. These patterns are useful hypotheses, not population-level causal conclusions.
- Grad-CAM identifies model-sensitive regions; it does not prove that those regions are physically causal.
- The Other label is heterogeneous, so its aggregate attribution can hide several physically different subclasses.

## Recommended Verification

1. Add velocity-band and RD-row occlusion tests to confirm that masking the central Doppler band changes Bird probability more than Other probability.
2. Compare the current global-average-pooling CNN with a structure that preserves coarse spatial layout, such as adaptive 4 x 8 pooling followed by a small classifier.
3. Report Bird/Other results at trajectory level and inspect the four directly confused trajectories individually.
4. If metadata permits, split Other into interpretable subgroups for analysis even if the final classifier remains five-class.
