"""NanoRT baseline training script.

This is the file competitors fork. Modify only:
  1. The BFS heuristic parameters (max_bfs_width)
  2. The transformer architecture (num_blocks, d_model, num_heads, d_ff,
     and any changes to rt/model.py)
  3. The optimizer (--optimizer muon, --muon-lr)
  4. SIGReg early-layer geometry regularization (--sigreg-layer, --sigreg-lambda-row,
     --sigreg-lambda-col). Training-only; adds zero inference cost.

Do NOT modify: val_tasks, val_eval_freq, time_budget_minutes, or the database
lists. Those define the competition protocol.

Run:
    pixi run python scripts/nanort_train.py --holdout-db rel-avito
    pixi run python scripts/nanort_train.py --holdout-db rel-avito --optimizer muon --muon-lr 5e-3
    # SIGReg on the first block, both views (paper-style λ=0.01):
    pixi run python scripts/nanort_train.py --holdout-db rel-avito --optimizer muon \
        --muon-lr 5e-3 --sigreg-layer 0 --sigreg-lambda-row 0.01 --sigreg-lambda-col 0.01
"""

import argparse
import json
import os

from rt.main import main
from rt.tasks import forecast_clf_tasks, autocomplete_clf_tasks

HOME = os.environ["HOME"]
PRE = os.path.join(HOME, "scratch", "pre")

HOLDOUT_CANDIDATES = [
    "rel-amazon",
    "rel-hm",
    "rel-stack",
    "rel-avito",
    "rel-trial",
    "rel-f1",
]

ALL_CLF_TASKS = [
    t for t in forecast_clf_tasks + autocomplete_clf_tasks if t[0] != "rel-event"
]


def clf_tasks_for_db(db_name: str):
    return [t for t in ALL_CLF_TASKS if t[0] == db_name]


def run(
    holdout_db: str,
    optimizer: str,
    muon_lr: float,
    sigreg_layer=None,
    sigreg_lambda_row: float = 0.0,
    sigreg_lambda_col: float = 0.0,
    sigreg_n_proj: int = 64,
    sigreg_lr: float | None = None,
    sigreg_min_group: int = 8,
    sigreg_col_global_pool: bool = True,
) -> None:
    assert holdout_db in HOLDOUT_CANDIDATES, f"Unknown holdout: {holdout_db}"
    assert optimizer in ("adamw", "muon"), f"Unknown optimizer: {optimizer}"

    # A negative layer (e.g. the -1 default passed by plx_nanort_train.yaml)
    # disables SIGReg, so the baseline stays byte-for-byte identical.
    if sigreg_layer is not None and sigreg_layer < 0:
        sigreg_layer = None

    train_artifact_db = f"nanort-train-{holdout_db}"
    train_artifact_path = os.path.join(PRE, train_artifact_db)
    if not os.path.isdir(train_artifact_path):
        raise FileNotFoundError(
            f"Train artifact not found at {train_artifact_path}.\n"
            f"Build it first: pixi run python scripts/build_artifact.py --holdout {holdout_db}"
        )

    train_tasks = [
        (train_artifact_db, f"{db}/{table}", target, drop)
        for db, table, target, drop in ALL_CLF_TASKS
        if db != holdout_db
    ]

    val_artifact_db = f"nanort-val-{holdout_db}"
    val_artifact_path = os.path.join(PRE, val_artifact_db)
    if not os.path.isdir(val_artifact_path):
        raise FileNotFoundError(
            f"Val artifact not found at {val_artifact_path}.\n"
            f"Build it first: pixi run python scripts/build_artifact.py --holdout {holdout_db}"
        )

    with open(os.path.join(val_artifact_path, "table_info.json")) as f:
        val_table_info = json.load(f)

    holdout_clf_tasks = clf_tasks_for_db(holdout_db)
    val_tasks = [
        (val_artifact_db, table, target, "Val", drop)
        for _, table, target, drop in holdout_clf_tasks
        if f"{table}:Val" in val_table_info
    ]
    if not val_tasks:
        raise RuntimeError(f"No val tasks found in {val_artifact_path}/table_info.json")
    print(f"Val tasks ({len(val_tasks)}): {[t[1] for t in val_tasks]}")

    seed = int(os.environ.get("NANORT_SEED", "0"))
    run_name_tag = f"{optimizer}"
    if optimizer == "muon":
        run_name_tag += f"-lr{muon_lr}"
    if sigreg_layer is not None:
        run_name_tag += f"-sig{sigreg_layer}r{sigreg_lambda_row}c{sigreg_lambda_col}"

    main(
        # misc
        project=os.environ.get("WANDB_PROJECT", "modded-nanort"),
        eval_splits=["val"],
        eval_freq=None,
        eval_pow2=False,
        max_eval_steps=40,
        load_ckpt_path=None,
        save_ckpt_dir=f"ckpts/nanort-{holdout_db}-{run_name_tag}-s{seed}",
        compile_=True,
        seed=seed,
        # data
        train_tasks=train_tasks,
        eval_tasks=[],
        batch_size=int(os.environ.get("NANORT_BATCH_SIZE", "32")),
        num_workers=8,
        max_bfs_width=256,          # ← competitors may adjust
        # optimization
        lr=1e-3,
        wd=0.1,
        lr_schedule=True,
        max_grad_norm=1.0,
        max_steps=1_000_000,
        # optimizer
        optimizer=optimizer,
        muon_lr=muon_lr,
        muon_warmup_steps=200,
        # model
        embedding_model="all-MiniLM-L12-v2",
        d_text=384,
        seq_len=1024,
        num_blocks=12,              # ← competitors may adjust
        d_model=256,                # ← competitors may adjust
        num_heads=8,                # ← competitors may adjust
        d_ff=1024,                  # ← competitors may adjust
        # SIGReg geometry regularization (← competitors may adjust)
        sigreg_layer=sigreg_layer,
        sigreg_lambda_row=sigreg_lambda_row,
        sigreg_lambda_col=sigreg_lambda_col,
        sigreg_n_proj=sigreg_n_proj,
        sigreg_lr=sigreg_lr,
        sigreg_min_group=sigreg_min_group,
        sigreg_col_global_pool=sigreg_col_global_pool,
        # NanoRT: competition hooks (do not modify)
        val_tasks=val_tasks,
        val_eval_freq=int(os.environ.get("NANORT_VAL_EVAL_FREQ", "100")),
        time_budget_minutes=45,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-db",
        required=True,
        choices=HOLDOUT_CANDIDATES,
        help="The held-out database (fixed by competition organizers)",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=["adamw", "muon"],
        help="Optimizer: 'adamw' (baseline) or 'muon'",
    )
    parser.add_argument(
        "--muon-lr",
        type=float,
        default=5e-3,
        help="Muon group learning rate (ignored when --optimizer adamw)",
    )
    parser.add_argument(
        "--sigreg-layer",
        type=int,
        default=None,
        help="Block index to apply SIGReg to (0 = first block, as in the paper). "
        "Omit to disable SIGReg.",
    )
    parser.add_argument(
        "--sigreg-lambda-row",
        type=float,
        default=0.0,
        help="SIGReg row-view (per-node) coefficient λr",
    )
    parser.add_argument(
        "--sigreg-lambda-col",
        type=float,
        default=0.0,
        help="SIGReg column-view (per-feature) coefficient λc",
    )
    parser.add_argument(
        "--sigreg-n-proj",
        type=int,
        default=64,
        help="Number of random 1-D projection directions per view",
    )
    parser.add_argument(
        "--sigreg-lr",
        type=float,
        default=None,
        help="Projector learning rate (defaults to the main AdamW lr)",
    )
    parser.add_argument(
        "--sigreg-min-group",
        type=int,
        default=8,
        help="Minimum valid cells per row/column group used by SIGReg",
    )
    parser.add_argument(
        "--sigreg-col-pool",
        type=int,
        default=1,
        choices=[0, 1],
        help="Column-view grouping: 1 = pool each (table, col) feature globally "
        "across the batch (recommended; large uniform groups), 0 = per-sequence "
        "groups (legacy).",
    )
    args = parser.parse_args()
    run(
        args.holdout_db,
        args.optimizer,
        args.muon_lr,
        sigreg_layer=args.sigreg_layer,
        sigreg_lambda_row=args.sigreg_lambda_row,
        sigreg_lambda_col=args.sigreg_lambda_col,
        sigreg_n_proj=args.sigreg_n_proj,
        sigreg_lr=args.sigreg_lr,
        sigreg_min_group=args.sigreg_min_group,
        sigreg_col_global_pool=bool(args.sigreg_col_pool),
    )
