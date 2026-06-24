#!/usr/bin/env bash
# Reproduces the current record on 1x H200: Muon + SIGReg col-view.
# Usage: ./run.sh [--holdout-db rel-avito] [--hf-repo <user/repo>] [--optimizer muon] [--muon-lr 5e-3]
#                 [--hf-ckpt-repo <user/repo>] [--hf-ckpt-name sigreg] [--stop-at-target]
set -euo pipefail

HOLDOUT_DB="${HOLDOUT_DB:-rel-avito}"
HF_REPO="${HF_REPO:-gabrielveloso/nanort-artifacts}"
OPTIMIZER="${OPTIMIZER:-muon}"
MUON_LR="${MUON_LR:-5e-3}"

# SIGReg col-view record config (see SIGREG.md / README.md):
SIGREG_LAYER="${SIGREG_LAYER:-0}"
SIGREG_LAMBDA_ROW="${SIGREG_LAMBDA_ROW:-0.0}"
SIGREG_LAMBDA_COL="${SIGREG_LAMBDA_COL:-0.001}"
SIGREG_COL_POOL="${SIGREG_COL_POOL:-1}"

# Record-reproducing run shape (overridable via env):
export NANORT_BATCH_SIZE="${NANORT_BATCH_SIZE:-256}"
export NANORT_VAL_EVAL_FREQ="${NANORT_VAL_EVAL_FREQ:-500}"

# Optional HuggingFace checkpoint upload at the AUROC target (off by default):
HF_CKPT_REPO="${HF_CKPT_REPO:-}"
HF_CKPT_NAME="${HF_CKPT_NAME:-sigreg}"
STOP_AT_TARGET="${STOP_AT_TARGET:-0}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --holdout-db)   HOLDOUT_DB="$2";        shift 2 ;;
    --hf-repo)      HF_REPO="$2";           shift 2 ;;
    --optimizer)    OPTIMIZER="$2";         shift 2 ;;
    --muon-lr)      MUON_LR="$2";           shift 2 ;;
    --hf-ckpt-repo) HF_CKPT_REPO="$2";      shift 2 ;;
    --hf-ckpt-name) HF_CKPT_NAME="$2";      shift 2 ;;
    --stop-at-target) STOP_AT_TARGET=1;     shift   ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=== nanoRT run ==="
echo "  holdout    : $HOLDOUT_DB"
echo "  hf-repo    : $HF_REPO"
echo "  optimizer  : $OPTIMIZER"
if [[ "$OPTIMIZER" == "muon" ]]; then
  echo "  muon-lr    : $MUON_LR"
fi
echo "  sigreg     : layer=$SIGREG_LAYER row=$SIGREG_LAMBDA_ROW col=$SIGREG_LAMBDA_COL pool=$SIGREG_COL_POOL"
echo "  batch_size : $NANORT_BATCH_SIZE"
if [[ -n "$HF_CKPT_REPO" ]]; then
  echo "  hf-ckpt    : $HF_CKPT_REPO -> checkpoints/$HF_CKPT_NAME.pt (stop_at_target=$STOP_AT_TARGET)"
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
HF_CKPT_ARGS=()
if [[ -n "$HF_CKPT_REPO" ]]; then
  HF_CKPT_ARGS+=(--hf-ckpt-repo "$HF_CKPT_REPO" --hf-ckpt-name "$HF_CKPT_NAME")
  [[ "$STOP_AT_TARGET" == "1" ]] && HF_CKPT_ARGS+=(--stop-at-target)
fi

pixi run python scripts/nanort_train.py \
  --holdout-db "$HOLDOUT_DB" \
  --optimizer  "$OPTIMIZER" \
  --muon-lr    "$MUON_LR" \
  --sigreg-layer "$SIGREG_LAYER" \
  --sigreg-lambda-row "$SIGREG_LAMBDA_ROW" \
  --sigreg-lambda-col "$SIGREG_LAMBDA_COL" \
  --sigreg-col-pool "$SIGREG_COL_POOL" \
  "${HF_CKPT_ARGS[@]}"
