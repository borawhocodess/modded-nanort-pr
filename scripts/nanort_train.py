"""NanoRT baseline training script.

This is the file competitors fork. Modify only:
  1. The BFS heuristic parameters (max_bfs_width)
  2. The transformer architecture (num_blocks, d_model, num_heads, d_ff,
     and any changes to rt/model.py)
  3. The optimizer (--optimizer muon, --muon-lr)

Do NOT modify: val_tasks, val_eval_freq, time_budget_minutes, or the database
lists. Those define the competition protocol.

Run:
    pixi run python scripts/nanort_train.py --holdout-db rel-avito
    pixi run python scripts/nanort_train.py --holdout-db rel-avito --optimizer muon --muon-lr 5e-3
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
    time_budget_minutes: float,
    max_steps: int,
    lr_total_steps: int | None,
    save_ckpt_dir: str | None,
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

    # A negative layer (e.g. the -1 default from plx_nanort_train.yaml) disables
    # SIGReg so the simplicial-only baseline stays byte-for-byte identical.
    if sigreg_layer is not None and sigreg_layer < 0:
        sigreg_layer = None

    train_dbs = [db for db in HOLDOUT_CANDIDATES if db != holdout_db]
    for db in train_dbs:
        db_path = os.path.join(PRE, db)
        if not os.path.isdir(db_path):
            raise FileNotFoundError(
                f"Train artifact not found at {db_path}.\n"
                f"Build it first: plx_nanort_build_upload_relations_holdout.yaml --holdout {holdout_db}"
            )

    train_tasks = [
        (db, table, target, drop)
        for db, table, target, drop in ALL_CLF_TASKS
        if db != holdout_db
    ]

    val_artifact_path = os.path.join(PRE, holdout_db)
    if not os.path.isdir(val_artifact_path):
        raise FileNotFoundError(
            f"Val artifact not found at {val_artifact_path}.\n"
            f"Build it first: plx_nanort_build_upload_relations_holdout.yaml --holdout {holdout_db}"
        )

    with open(os.path.join(val_artifact_path, "table_info.json")) as f:
        val_table_info = json.load(f)

    holdout_clf_tasks = clf_tasks_for_db(holdout_db)
    val_tasks = [
        (holdout_db, table, target, "Val", drop)
        for _, table, target, drop in holdout_clf_tasks
        if f"{table}:Val" in val_table_info
    ]
    if not val_tasks:
        raise RuntimeError(f"No val tasks found in {val_artifact_path}/table_info.json")
    print(f"Val tasks ({len(val_tasks)}): {[t[1] for t in val_tasks]}")

    seed = int(os.environ.get("NANORT_SEED", "0"))

    # Which blocks get the simplicial pair (SimplicialNbrAttn + RelAttn).
    # "all" (default) = every block; "none" = disable the simplicial path;
    # "0,4,8" = only those block indices. Fewer blocks cut step-time linearly.
    # Tolerant parsing: accepts "[0, 4, 8]", "'0,4,8'", stray spaces/quotes —
    # Polyaxon/Jinja may re-render the param in list-ish forms.
    _simplicial_env = os.environ.get("NANORT_SIMPLICIAL_LAYERS", "all")
    _clean = _simplicial_env.strip().strip("[]()\"'").strip().lower()
    if _clean in ("all", ""):
        simplicial_layers = None
    elif _clean == "none":
        simplicial_layers = []
    else:
        simplicial_layers = [
            int(tok.strip("\"' ")) for tok in _clean.split(",") if tok.strip("\"' ")
        ]
    print(
        f"simplicial_layers={'all' if simplicial_layers is None else simplicial_layers}"
        f"  (NANORT_SIMPLICIAL_LAYERS={_simplicial_env!r})",
        flush=True,
    )

    def _env_bool(name, default="1"):
        v = os.environ.get(name, default).strip().strip("\"'").lower()
        return v not in ("0", "false", "no", "off", "")

    update_relations = _env_bool("NANORT_UPDATE_RELATIONS")  # 0 = static (HGT-style)
    qk_norm = _env_bool("NANORT_QK_NORM")                    # 0 = no QK-norm
    print(
        f"update_relations={update_relations}  qk_norm={qk_norm}",
        flush=True,
    )
    print(
        f"sigreg_layer={sigreg_layer}  lambda_row={sigreg_lambda_row}  "
        f"lambda_col={sigreg_lambda_col}  n_proj={sigreg_n_proj}",
        flush=True,
    )

    run_name_tag = f"{optimizer}"
    if optimizer == "muon":
        run_name_tag += f"-lr{muon_lr}"
    if sigreg_layer is not None and (sigreg_lambda_row + sigreg_lambda_col) > 0:
        run_name_tag += f"-sig-l{sigreg_layer}-r{sigreg_lambda_row}-c{sigreg_lambda_col}"
    if save_ckpt_dir is None:
        save_ckpt_dir = f"ckpts/nanort-{holdout_db}-{run_name_tag}-s{seed}"

    main(
        # misc
        project=os.environ.get("WANDB_PROJECT", "modded-nanort"),
        eval_splits=["val"],
        eval_freq=None,
        eval_pow2=False,
        max_eval_steps=40,
        load_ckpt_path=None,
        save_ckpt_dir=save_ckpt_dir,
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
        max_steps=max_steps,
        lr_total_steps=lr_total_steps,
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
        simplicial_layers=simplicial_layers,  # ← NANORT_SIMPLICIAL_LAYERS
        update_relations=update_relations,    # ← NANORT_UPDATE_RELATIONS
        qk_norm=qk_norm,                      # ← NANORT_QK_NORM
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
        time_budget_minutes=time_budget_minutes,
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
        "--time-budget-minutes",
        type=float,
        default=45,
        help="Active training time budget before saving the final checkpoint",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1_000_000,
        help="Hard step cap (wall-clock budget normally stops first)",
    )
    parser.add_argument(
        "--lr-total-steps",
        type=int,
        default=None,
        help="OneCycleLR horizon; defaults to --max-steps if not set",
    )
    parser.add_argument(
        "--save-ckpt-dir",
        default=None,
        help="Directory where checkpoints will be written",
    )
    # SIGReg geometry regularization (← competitors may adjust)
    parser.add_argument(
        "--sigreg-layer", type=int, default=None,
        help="Block index whose cell hidden state is SIGReg-regularized; "
             "negative disables SIGReg (simplicial-only baseline)",
    )
    parser.add_argument(
        "--sigreg-lambda-row", type=float, default=0.0,
        help="SIGReg row-view loss coefficient",
    )
    parser.add_argument(
        "--sigreg-lambda-col", type=float, default=0.0,
        help="SIGReg column-view loss coefficient",
    )
    parser.add_argument(
        "--sigreg-n-proj", type=int, default=64,
        help="Number of random projections used by SIGReg",
    )
    parser.add_argument(
        "--sigreg-lr", type=float, default=None,
        help="SIGReg projector LR (defaults to the main lr)",
    )
    parser.add_argument(
        "--sigreg-min-group", type=int, default=8,
        help="Minimum group size for the SIGReg statistic",
    )
    parser.add_argument(
        "--sigreg-col-pool", type=int, choices=[0, 1], default=1,
        help="1 = pool the column view globally across the batch",
    )
    args = parser.parse_args()
    try:
        run(
            args.holdout_db,
            args.optimizer,
            args.muon_lr,
            args.time_budget_minutes,
            args.max_steps,
            args.lr_total_steps,
            args.save_ckpt_dir,
            sigreg_layer=args.sigreg_layer,
            sigreg_lambda_row=args.sigreg_lambda_row,
            sigreg_lambda_col=args.sigreg_lambda_col,
            sigreg_n_proj=args.sigreg_n_proj,
            sigreg_lr=args.sigreg_lr,
            sigreg_min_group=args.sigreg_min_group,
            sigreg_col_global_pool=bool(args.sigreg_col_pool),
        )
    except BaseException:
        # Polyaxon pod logs vanish with the pod — persist the traceback in
        # wandb (if the run got that far) and flush it to stdout.
        import traceback

        tb = traceback.format_exc()
        print(tb, flush=True)
        try:
            import wandb

            if wandb.run is not None:
                wandb.run.summary["crash_traceback"] = tb
                wandb.finish(exit_code=1)
        except Exception:
            pass
        raise
