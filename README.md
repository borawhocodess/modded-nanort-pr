# modded-nanoRT

This repository hosts the *nanoRT speedrun*, in which we (collaboratively|competitively) search for the fastest way to use 1 NVIDIA H200 to train a Relational Transformer that beats the baseline on [`rel-avito`](https://relbench.stanford.edu/) from RelBench.

The code is derived from [snap-stanford/relational-transformer](https://github.com/snap-stanford/relational-transformer) with the inspiration of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

The current record beats the baseline in **7m 26s** with the following techniques:

* Replaced AdamW with [Muon](https://github.com/KellerJordan/Muon) (`muon_lr=5e-3`) for all matrix parameters, keeping AdamW for embeddings and biases.
* Added **SIGReg column-view** geometry regularization on the first block (`sigreg_layer=0`, `sigreg_lambda_col=1e-3`, global column pooling), which de-collapses the early per-feature representation. Training-only — adds zero inference cost.

The following techniques were evaluated but did not lead to improvements:

* SIGReg **row-view** (`sigreg_lambda_row`): regularizes the post-aggregation relation stream (`dr`), which is the wrong stream — it did not improve over the column-only variant and destabilized training at larger λ.


## Running the current record

```bash
git clone https://github.com/gabrielmveloso/Modded-NanoRT.git
cd Modded-NanoRT
pixi install
cd rustler && pixi run maturin develop --uv --release && cd ..

pixi run hf download gabrielveloso/nanort-artifacts \
    --repo-type dataset \
    --include "holdout-rel-avito/*" \
    --local-dir ~/scratch/nanort_artifacts

mkdir -p ~/scratch/pre
ln -sf ~/scratch/nanort_artifacts/holdout-rel-avito/train ~/scratch/pre/nanort-train-rel-avito
ln -sf ~/scratch/nanort_artifacts/holdout-rel-avito/val/rel-avito ~/scratch/pre/nanort-val-rel-avito

./run.sh
```

Note: `torch.compile` of the flex-attention masks adds ~3 minutes on the first run. This is untimed.


## Record history

The following is the historical progression of speed records for the following task:

> *Train a Relational Transformer to ≥0.81621 zero-shot AUROC on `rel-avito` using 1× NVIDIA H200, within 45 minutes of active training time.*

The 0.81621 target was selected to match the baseline calibration run over the upstream [snap-stanford/relational-transformer](https://github.com/snap-stanford/relational-transformer) code.

| #  | Record            | Date       | Description              | Links                      | Contributors     |
| -- | ----------------- | ---------- | ------------------------ | -------------------------- | ---------------- |
| 1  | 0.81621 AUROC @ 45 min | TBD   | Baseline                 | [log](records/01_baseline) |                  |
| 2  | 12m 27s           | 2026-05-05 | Muon optimizer (`muon_lr=5e-3`) | [log](records/02_muon) | Gabriel Veloso   |
| 3  | **7m 26s**        | 2026-06-24 | Muon + SIGReg col-view (`sigreg_layer=0`, `λ_col=1e-3`) | [log](records/03_sigreg) | Gabriel Veloso   |


## Rules

New records must:
* Not modify the val artifact (seeds, embeddings, 3-hop extraction radius).
* Not modify the training mix (same 5 databases, same α=0.3 mixing, no `rel-avito`).
* Not use `rel-avito` data in any form during training.
* Attain ≥0.81621 mean val AUROC at the 45-minute mark (p<0.01 over multiple seeds).
* Run on 1× H200, with active training time measured by `rt/main.py`'s `TrainTimer`.

Other than that, anything and everything is fair game!


## Evaluation details

Evaluation is zero-shot AUROC on the `rel-avito` validation seeds.

* Val set: `rel-avito` mini-database, seeds from `Val` split, capped at 50 000 seeds, extracted at `max_hops=3`.
* Seeds, hops, and embeddings are frozen for all submissions.
* Metric: mean AUROC across `rel-avito` binary-classification tasks, averaged over the val seeds.


## References

1. [Robinson et al. RelBench: A Benchmark for Deep Learning on Relational Databases. NeurIPS 2024.](https://arxiv.org/abs/2407.20060)
2. [Ranjan et al. Relational Transformer. snap-stanford, 2024.](https://github.com/snap-stanford/relational-transformer)
3. [Reimers & Gurevych. Sentence-BERT. EMNLP 2019.](https://arxiv.org/abs/1908.10084)
4. [Jordan et al. modded-nanogpt. 2024–2026.](https://github.com/KellerJordan/modded-nanogpt)
5. [Müller et al. modded-nanotabpfn. 2026.](https://github.com/borawhocodess/modded-nanotabpfn)
