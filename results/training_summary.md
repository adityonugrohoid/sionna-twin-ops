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
| dataset | /home/adityonugrohoid/projects/sionna-twin-ops/data/dataset-v1 (manifest sha256 a8c98681b9a412bd...) |
| maps | train 1680, validation 360 |

| provenance | values seen |
|---|---|
| commit | cc4199d2bfd652b6261b1b15d7fb93ccf7da24f7 |
| platform | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| python | 3.13.12 |
| torch | 2.14.0+cu130 |
| device | cuda |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56 |

## Best epoch per seed (validation)

L1 is the mean absolute error in dB over validation cells where the ray tracer has power; power accuracy is the share of all validation cells whose power logit has the right sign.

| seed | best epoch | L1 (dB) | NLOS L1 (dB) | BCE | power accuracy | total loss | L1 at last epoch (dB) | seconds |
|---|---|---|---|---|---|---|---|---|
| 0 | 16 | 1.796 | 4.270 | 0.0572 | 97.68% | 0.2368 | 2.502 (epoch 50) | 441 |
| 1 | 14 | 1.841 | 4.378 | 0.0590 | 97.62% | 0.2431 | 2.457 (epoch 50) | 449 |
| 2 | 14 | 1.824 | 4.398 | 0.0588 | 97.67% | 0.2412 | 2.350 (epoch 50) | 451 |

Across seeds: L1 1.820 (min 1.796, max 1.841) dB; power accuracy 0.977 (min 0.976, max 0.977); BCE 0.058 (min 0.057, max 0.059).

All 3 seeds peak at epochs 14-16 and overfit after: for seed 0, validation L1 goes from 1.796 dB at epoch 16 to 2.502 dB at epoch 50, while the training loss falls from 0.214 to 0.102. The saved weights are those of the best epoch.

## Context: earlier runs, quoted

Quoted from the run records of commit 5b18180 (augmentation none), not retrained or recomputed here.

| seed | best epoch | L1 (dB) | power accuracy | L1 at last epoch (dB) |
|---|---|---|---|---|
| 0 | 7 | 1.958 | 97.44% | 2.541 (epoch 50) |
| 1 | 5 | 1.936 | 97.41% | 2.476 (epoch 50) |
| 2 | 7 | 1.935 | 97.46% | 2.414 (epoch 50) |
