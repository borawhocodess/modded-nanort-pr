"""NanoRT training entry point.

All configuration lives in ``rt.config.TrainConfig`` and is populated from the
``NANORT_*`` environment variables (set by ``plx_nanort_train.yaml``). There is
**no command-line flag parsing** — to change a knob, set the env var / YAML input.

Run:
    NANORT_HOLDOUT_DB=rel-avito pixi run python scripts/nanort_train.py
"""

import json
import os

from rt.config import TrainConfig
from rt.main import main
from rt.tasks import autocomplete_clf_tasks, forecast_clf_tasks

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

# Competition protocol — do NOT modify. rel-event is excluded from the pool.
ALL_CLF_TASKS = [
    t for t in forecast_clf_tasks + autocomplete_clf_tasks if t[0] != "rel-event"
]


def clf_tasks_for_db(db_name: str):
    return [t for t in ALL_CLF_TASKS if t[0] == db_name]


def resolve_tasks(cfg: TrainConfig) -> TrainConfig:
    """Fill cfg.train_tasks / cfg.val_tasks from cfg.holdout_db and validate that
    the required preprocessed artifacts are present. Returns the same cfg."""
    assert cfg.holdout_db in HOLDOUT_CANDIDATES, f"Unknown holdout: {cfg.holdout_db}"
    assert cfg.optimizer in ("adamw", "muon"), f"Unknown optimizer: {cfg.optimizer}"

    train_dbs = [db for db in HOLDOUT_CANDIDATES if db != cfg.holdout_db]
    for db in train_dbs:
        db_path = os.path.join(PRE, db)
        if not os.path.isdir(db_path):
            raise FileNotFoundError(
                f"Train artifact not found at {db_path}.\n"
                f"Build it first: plx_nanort_build_upload_relations_holdout.yaml "
                f"--holdout {cfg.holdout_db}"
            )

    cfg.train_tasks = [
        (db, table, target, drop)
        for db, table, target, drop in ALL_CLF_TASKS
        if db != cfg.holdout_db
    ]
    cfg.eval_tasks = []

    val_artifact_path = os.path.join(PRE, cfg.holdout_db)
    if not os.path.isdir(val_artifact_path):
        raise FileNotFoundError(
            f"Val artifact not found at {val_artifact_path}.\n"
            f"Build it first: plx_nanort_build_upload_relations_holdout.yaml "
            f"--holdout {cfg.holdout_db}"
        )
    with open(os.path.join(val_artifact_path, "table_info.json")) as f:
        val_table_info = json.load(f)

    cfg.val_tasks = [
        (cfg.holdout_db, table, target, "Val", drop)
        for _, table, target, drop in clf_tasks_for_db(cfg.holdout_db)
        if f"{table}:Val" in val_table_info
    ]
    if not cfg.val_tasks:
        raise RuntimeError(
            f"No val tasks found in {val_artifact_path}/table_info.json"
        )
    print(f"Val tasks ({len(cfg.val_tasks)}): {[t[1] for t in cfg.val_tasks]}", flush=True)
    return cfg


def default_ckpt_dir(cfg: TrainConfig) -> str:
    tag = cfg.optimizer
    if cfg.optimizer == "muon":
        tag += f"-lr{cfg.muon_lr}"
    if cfg.sigreg_layer is not None and (cfg.sigreg_lambda_row + cfg.sigreg_lambda_col) > 0:
        tag += f"-sig-l{cfg.sigreg_layer}-c{cfg.sigreg_lambda_col}"
    if not cfg.update_relations:
        tag += "-staticrel"
    return f"ckpts/nanort-{cfg.holdout_db}-{tag}-s{cfg.seed}"


def run(cfg: TrainConfig) -> None:
    resolve_tasks(cfg)
    if cfg.save_ckpt_dir is None:
        cfg.save_ckpt_dir = default_ckpt_dir(cfg)
    print(f"[config] {cfg.summary()}", flush=True)
    main(cfg)


if __name__ == "__main__":
    cfg = TrainConfig.from_env()
    try:
        run(cfg)
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
