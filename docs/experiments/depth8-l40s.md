# 8-block L40S comparison

Reducing depth from 12 to 8 blocks reduced seed-0 time to the pooled AUROC
target by 25.3%, from 412.5826 s to 308.3431 s. Both first crossed at step 100.
These are L40S measurements from the 2026-10-29 sweep; H200 timing is pending.

| blocks | seed | time to target | pooled AUROC at crossing | best pooled AUROC within 45 min | best mean-task AUROC within 45 min |
| --- | --- | --- | --- | --- | --- |
| 12 | 0 | 412.5826 s | 0.868183 | 0.868183 | 0.578857 |
| 8 | 0 | 308.3431 s | 0.833957 | 0.857635 | 0.631231 |
| 12 | 1 | missed within 45 min | — | 0.762457 | 0.608454 |
| 8 | 1 | missed within 45 min | — | 0.810990 | 0.584052 |

The two best-score columns can refer to different evaluations. They exclude
initialization and evaluations logged after 2700 s. The 8-block seed-1 run
crossed at step 900 after 2769.2446 s, outside the budget, so it is a miss here.

Both seed-0 models already exceeded the pooled target at initialization:
0.865028 for 12 blocks and 0.817912 for 8 blocks. Step 0 cannot establish a
record. These timings therefore compare how quickly the models complete the
first eligible evaluation; they do not establish faster learning from below
the target. The 8-block seed-0 mean-task AUROC was already 0.624655 at
initialization, compared with 0.567726 for 12 blocks.

## Measurement setup

- One NVIDIA L40S per run; one run per configuration and seed, seeds 0 and 1.
- Training batch 128; validation batch 256; sequence length 1024.
- Width 256, 8 heads, FFN width 1024, static typed trilinear attention.
- Reference Muon, LR 0.005, 3000-update warmup; AdamW LR 0.001;
  fixed LR horizon 1,000,000 updates; column SIGReg weight 0.001 after block 0.
- BF16 backbone, FP32 SIGReg, compilation enabled.
- Full frozen `rel-avito` validation, target pooled AUROC 0.81621,
  evaluation every 100 updates, 45-minute timed budget.
- Python 3.12.0, PyTorch 2.6.0+cu124, CUDA 12.4.
- Training artifact `GabrielVeloso/nanort-artifacts`, immutable revision
  `8e6289edd0aa1d61b3b4c2ec1e1beed48224cd2c`, prefix `ra-holdout-rel-avito`.

The experiment used a separate sweep checkout, commit
`d77359b7d819ac5c071810b1b81c5e34ec5b6d17`, based on
`d063c31ef9a92f9890c4dc9437cec942044eedec`. That checkout added local JSONL
logging and an independent validation batch setting. Its optional batched
Muon and residual decay were disabled for these four runs. Only depth differed
between the 12-block and 8-block configurations at a given seed.

This branch changes the default depth. Its default training batch remains
256, and its launcher shares that batch size with validation. The measurements
above came from the sweep checkout with training batch 128 and validation
batch 256; they are not timings of this branch's default invocation.

## Evaluation history

[depth8-l40s.csv](depth8-l40s.csv) contains every evaluation for these four runs,
extracted from their local `metrics.jsonl` files. It includes initialization,
late evaluations, each task's AUROC, pooled AUROC and mean-task AUROC.

`time_to_target_seconds` is the evaluator's `train/time_to_target` field,
present only at the first post-update crossing, including a late crossing.
`elapsed_seconds` is sampled slightly later, after checkpoint writing and
logging. `eligible` requires a positive step and elapsed time at most 2700 s.

The historical timer starts after the first optimizer update and its logging.
Setup, compilation warmup and initialization evaluation are untimed. Later
training, validation, logging and checkpoint writing count toward the budget.
The sweep continued to the budget after reaching the target; its final
evaluation could overrun the budget, and those rows are retained and marked
ineligible in the CSV.
