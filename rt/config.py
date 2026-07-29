"""Single source of truth for training configuration.

All knobs live in ``TrainConfig``. The only runtime entry point (``scripts/
nanort_train.py``) builds one via ``TrainConfig.from_env()`` and hands it to
``rt.main.main`` — there is no command-line flag parsing anywhere.

This module deliberately has **no torch import** so the config surface can be
inspected and unit-tested without a GPU environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


# ── env parsing helpers ────────────────────────────────────────────────────
def _str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().strip("\"'")


def _int(name: str, default: int) -> int:
    return int(_str(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_str(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _str(name, "1" if default else "0").lower() not in (
        "0", "false", "no", "off", ""
    )


def _opt_int(name: str, default, *, disabled_below: int | None = None):
    """Read an int env var; return None when unset or (optionally) when the
    value is a sentinel below ``disabled_below`` (e.g. sigreg_layer=-1, or
    lr_total_steps=0 → 'use max_steps')."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    v = int(raw.strip().strip("\"'"))
    if disabled_below is not None and v < disabled_below:
        return None
    return v


def _opt_float(name: str, default, *, zero_is_none: bool = False):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    v = float(raw.strip().strip("\"'"))
    if zero_is_none and v == 0.0:
        return None
    return v


def _simplicial_layers(name: str):
    """'all' → None (every block); 'none' → []; '0,4,8' → [0,4,8]. Tolerant of
    list-ish forms Polyaxon/Jinja may emit ('[0, 4]', quotes, spaces)."""
    raw = os.environ.get(name, "all")
    clean = raw.strip().strip("[]()\"'").strip().lower()
    if clean in ("all", ""):
        return None
    if clean == "none":
        return []
    return [int(tok.strip("\"' ")) for tok in clean.split(",") if tok.strip("\"' ")]


@dataclass
class TrainConfig:
    """Every training knob. Defaults reproduce the competition run; ``from_env``
    overlays the ``NANORT_*`` environment variables set by the Polyaxon YAML."""

    # ── run identity ────────────────────────────────────────────────────────
    holdout_db: str = "rel-avito"
    seed: int = 0
    project: str = "modded-nanort"

    # ── data ────────────────────────────────────────────────────────────────
    batch_size: int = 256
    num_workers: int = 8
    max_bfs_width: int = 256
    seq_len: int = 1024
    embedding_model: str = "all-MiniLM-L12-v2"
    d_text: int = 384
    # Resolved at runtime from holdout_db (not env-configurable).
    train_tasks: list = field(default_factory=list)
    eval_tasks: list = field(default_factory=list)
    val_tasks: list = field(default_factory=list)

    # ── optimization ────────────────────────────────────────────────────────
    optimizer: str = "muon"            # "adamw" | "muon"
    lr: float = 1e-3                   # AdamW group LR
    muon_lr: float = 5e-3             # Muon group LR (ignored when optimizer="adamw")
    muon_warmup_steps: int = 200
    wd: float = 0.1
    lr_schedule: bool = True
    max_grad_norm: float = 1.0
    max_steps: int = 1_000_000
    lr_total_steps: int | None = None  # OneCycle horizon; None → max_steps

    # ── model ───────────────────────────────────────────────────────────────
    num_blocks: int = 12
    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 1024
    simplicial_layers: list | None = None   # None = all blocks; [] = disabled
    update_relations: bool = True            # False = static text-relation stream
    qk_norm: bool = True
    simplicial_layerscale_init: float = 1.0

    # ── SIGReg (geometry regularization; sigreg_layer=None disables it) ──────
    sigreg_layer: int | None = None
    sigreg_lambda_row: float = 0.0
    sigreg_lambda_col: float = 0.0
    sigreg_lr: float | None = None
    sigreg_n_proj: int = 64
    sigreg_proj_hidden_mult: int = 4
    sigreg_proj_out_div: int = 4
    sigreg_proj_hidden_layers: int = 1
    sigreg_knots: int = 17
    sigreg_t_max: float = 3.0
    sigreg_min_group: int = 8
    sigreg_col_global_pool: bool = True

    # ── evaluation / budget / checkpoint ────────────────────────────────────
    eval_splits: tuple = ("val",)
    eval_freq: int | None = None
    eval_pow2: bool = False
    max_eval_steps: int = 40
    val_eval_freq: int | None = 100
    eval_at_step1: bool = True
    time_budget_minutes: float | None = 45.0
    load_ckpt_path: str | None = None
    save_ckpt_dir: str | None = None
    compile_: bool = True

    @classmethod
    def from_env(cls) -> "TrainConfig":
        """Build a config from the ``NANORT_*`` environment. Unset vars fall
        back to the dataclass defaults above."""
        return cls(
            holdout_db=_str("NANORT_HOLDOUT_DB", cls.holdout_db),
            seed=_int("NANORT_SEED", cls.seed),
            project=_str("WANDB_PROJECT", cls.project),
            batch_size=_int("NANORT_BATCH_SIZE", cls.batch_size),
            optimizer=_str("NANORT_OPTIMIZER", cls.optimizer),
            lr=_float("NANORT_LR", cls.lr),
            muon_lr=_float("NANORT_MUON_LR", cls.muon_lr),
            muon_warmup_steps=_int("NANORT_MUON_WARMUP_STEPS", cls.muon_warmup_steps),
            max_steps=_int("NANORT_MAX_STEPS", cls.max_steps),
            lr_total_steps=_opt_int("NANORT_LR_TOTAL_STEPS", None, disabled_below=1),
            simplicial_layers=_simplicial_layers("NANORT_SIMPLICIAL_LAYERS"),
            update_relations=_bool("NANORT_UPDATE_RELATIONS", cls.update_relations),
            qk_norm=_bool("NANORT_QK_NORM", cls.qk_norm),
            simplicial_layerscale_init=_float(
                "NANORT_SIMPLICIAL_LAYERSCALE_INIT", cls.simplicial_layerscale_init
            ),
            sigreg_layer=_opt_int("NANORT_SIGREG_LAYER", None, disabled_below=0),
            sigreg_lambda_row=_float("NANORT_SIGREG_LAMBDA_ROW", cls.sigreg_lambda_row),
            sigreg_lambda_col=_float("NANORT_SIGREG_LAMBDA_COL", cls.sigreg_lambda_col),
            sigreg_lr=_opt_float("NANORT_SIGREG_LR", None, zero_is_none=True),
            sigreg_n_proj=_int("NANORT_SIGREG_N_PROJ", cls.sigreg_n_proj),
            sigreg_min_group=_int("NANORT_SIGREG_MIN_GROUP", cls.sigreg_min_group),
            sigreg_col_global_pool=_bool("NANORT_SIGREG_COL_POOL", cls.sigreg_col_global_pool),
            val_eval_freq=_int("NANORT_VAL_EVAL_FREQ", cls.val_eval_freq),
            eval_at_step1=_bool("NANORT_EVAL_AT_STEP1", cls.eval_at_step1),
            time_budget_minutes=_float("NANORT_TIME_BUDGET_MINUTES", cls.time_budget_minutes),
            save_ckpt_dir=os.environ.get("NANORT_SAVE_CKPT_DIR") or None,
        )

    def summary(self) -> str:
        """One-line human-readable dump of the non-task fields (for logs)."""
        skip = {"train_tasks", "eval_tasks", "val_tasks"}
        return "  ".join(
            f"{f.name}={getattr(self, f.name)}"
            for f in fields(self)
            if f.name not in skip
        )
