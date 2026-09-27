# Training summary

Synthetic terrain. The surrogate is a U-Net predicting the ray-traced path gain as a residual over B0, plus a logit for whether the ray tracer has power in each cell (spec M1, M1b). Trained on the train split, best epoch chosen on validation; the test split is not touched here. Written by `twin training-summary`.

| setting | value |
|---|---|
| epochs | 50 |
| width | 32 |
| parameters | 7767170 |
| batch_size | 16 |
| learning_rate | 0.001 |
| weight_decay | 0.0001 |
| power_loss_weight | 1.0 |
| augmentation | symmetry |
| schedule | cosine over all steps |
| optimizer | AdamW |
| cudnn_deterministic | True |
| dataset | /home/adityonugrohoid/projects/sionna-twin-ops/data/dataset-v2 (manifest sha256 fb0f1b247b3b2171...) |
| maps | train 5040, validation 360 |

| provenance | values seen |
|---|---|
| commit | f6f13e3857d7d25de399641d863200e6a3e7b904 |
| platform | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| python | 3.13.12 |
| torch | 2.14.0+cu130 |
| device | cuda |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56 |

## Best epoch per seed (validation)

L1 is the mean absolute error in dB over validation cells where the ray tracer has power; power accuracy is the share of all validation cells whose power logit has the right sign. LOS cells are direct-dominated when the traced gain is less than 3 dB above B0 and reflection-dominated otherwise (ASSUMPTION, the rule of `twin fold-check`).

| seed | best epoch | L1 (dB) | NLOS L1 (dB) | LOS direct L1 (dB) | LOS reflection L1 (dB) | BCE | power accuracy | total loss | last epoch: L1 (dB) | last epoch: BCE | last epoch: power accuracy | peak RSS after load (MiB) | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 13 | 1.549 | 3.836 | 0.799 | 3.621 | 0.0472 | 98.07% | 0.2021 | 2.119 (epoch 50) | 0.2435 | 97.52% | 2798 | 1250 |
| 1 | 10 | 1.569 | 3.896 | 0.847 | 3.361 | 0.0496 | 98.00% | 0.2065 | 2.146 (epoch 50) | 0.2757 | 97.43% | 2835 | 1277 |
| 2 | 14 | 1.499 | 3.755 | 0.800 | 3.229 | 0.0467 | 98.12% | 0.1966 | 1.975 (epoch 50) | 0.1793 | 97.53% | 2800 | 1274 |

Across seeds: L1 1.539 (min 1.499, max 1.569) dB; power accuracy 0.981 (min 0.980, max 0.981); BCE 0.048 (min 0.047, max 0.050).

All 3 seeds peak at epochs 10-14 and overfit after: for seed 0, validation L1 goes from 1.549 dB at epoch 13 to 2.119 dB at epoch 50, while the training loss falls from 0.190 to 0.095. The saved weights are those of the best epoch.

## Context: earlier runs, quoted

Quoted from the run records of commit cc4199d (augmentation symmetry), not retrained or recomputed here.

| seed | best epoch | L1 (dB) | power accuracy | L1 at last epoch (dB) |
|---|---|---|---|---|
| 0 | 16 | 1.796 | 97.68% | 2.502 (epoch 50) |
| 1 | 14 | 1.841 | 97.62% | 2.457 (epoch 50) |
| 2 | 14 | 1.824 | 97.67% | 2.350 (epoch 50) |
