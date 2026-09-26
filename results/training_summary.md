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
| schedule | cosine over all steps |
| optimizer | AdamW |
| cudnn_deterministic | True |
| dataset | /home/adityonugrohoid/projects/sionna-twin-ops/data/dataset-v1 (manifest sha256 a8c98681b9a412bd...) |
| maps | train 1680, validation 360 |

| provenance | values seen |
|---|---|
| commit | 5b18180b76b7fd38694878d4048a270a24ace666 |
| platform | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| python | 3.13.12 |
| torch | 2.14.0+cu130 |
| device | cuda |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56 |

## Best epoch per seed (validation)

L1 is the mean absolute error in dB over validation cells where the ray tracer has power; power accuracy is the share of all validation cells whose power logit has the right sign.

| seed | best epoch | L1 (dB) | BCE | power accuracy | total loss | seconds |
|---|---|---|---|---|---|---|
| 0 | 7 | 1.958 | 0.0640 | 97.44% | 0.2598 | 396 |
| 1 | 5 | 1.936 | 0.0666 | 97.41% | 0.2602 | 408 |
| 2 | 7 | 1.935 | 0.0649 | 97.46% | 0.2584 | 408 |

Across seeds: L1 1.943 (min 1.935, max 1.958) dB; power accuracy 0.974 (min 0.974, max 0.975); BCE 0.065 (min 0.064, max 0.067).
