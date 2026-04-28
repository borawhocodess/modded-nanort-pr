#!/usr/bin/env bash
# Reproduces the current record on 1x H200.
# Usage: ./run.sh [--holdout-db rel-avito] [--hf-repo <user/repo>] [--optimizer muon] [--muon-lr 5e-3]
set -euo pipefail

HOLDOUT_DB="${HOLDOUT_DB:-rel-avito}"
HF_REPO="${HF_REPO:-gabrielveloso/nanort-artifacts}"
OPTIMIZER="${OPTIMIZER:-muon}"
MUON_LR="${MUON_LR:-5e-3}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --holdout-db) HOLDOUT_DB="$2"; shift 2 ;;
    --hf-repo)    HF_REPO="$2";    shift 2 ;;
    --optimizer)  OPTIMIZER="$2";  shift 2 ;;
    --muon-lr)    MUON_LR="$2";    shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=== nanoRT run ==="
echo "  holdout  : $HOLDOUT_DB"
echo "  hf-repo  : $HF_REPO"
echo "  optimizer: $OPTIMIZER"
if [[ "$OPTIMIZER" == "muon" ]]; then
  echo "  muon-lr  : $MUON_LR"
fi
echo ""

# ── 1. Build Rust extension ───────────────────────────────────────────────────
cd rustler
pixi run maturin develop --uv --release
cd ..

# ── 2. Download artifacts ─────────────────────────────────────────────────────
mkdir -p ~/scratch/pre

echo "--- Downloading train artifact ---"
hf download "$HF_REPO" \
  --repo-type dataset \
  --include "holdout-${HOLDOUT_DB}/train/*" \
  --local-dir ~/scratch/nanort_artifacts

echo "--- Downloading val artifact ---"
hf download "$HF_REPO" \
  --repo-type dataset \
  --include "holdout-${HOLDOUT_DB}/val/*" \
  --local-dir ~/scratch/nanort_artifacts

# ── 3. Symlink into layout rt/data.py expects ─────────────────────────────────
TRAIN_LINK=~/scratch/pre/nanort-train-${HOLDOUT_DB}
VAL_LINK=~/scratch/pre/nanort-val-${HOLDOUT_DB}

[ -L "$TRAIN_LINK" ] && rm "$TRAIN_LINK"
[ -L "$VAL_LINK"   ] && rm "$VAL_LINK"

ln -sf ~/scratch/nanort_artifacts/holdout-${HOLDOUT_DB}/train              "$TRAIN_LINK"
ln -sf ~/scratch/nanort_artifacts/holdout-${HOLDOUT_DB}/val/${HOLDOUT_DB}  "$VAL_LINK"

# ── 4. Train ──────────────────────────────────────────────────────────────────
echo "--- Starting training (45 min budget) ---"
pixi run python scripts/nanort_train.py \
  --holdout-db "$HOLDOUT_DB" \
  --optimizer  "$OPTIMIZER" \
  --muon-lr    "$MUON_LR"
