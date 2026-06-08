"""Export the latest SIGReg follow-up sweep from W&B.

This targets the current col-only sweep:

  * baseline: sigreg off, seeds 0/1/2
  * layer 0, lambda_row=0, lambda_col in {0.002, 0.003, 0.005, 0.01}

For each (layer, lambda_row, lambda_col, seed), the newest matching W&B run is
selected. This avoids mixing duplicate/older runs from previous sweeps.

Usage:
    python3 scripts/export_latest_sigreg_sweep.py --output sigreg_latest_history.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import wandb


DEFAULT_ENTITY = "franciscokunumi-universidade-federal-de-minas-gerais"
DEFAULT_PROJECT = "nanort-sigreg-sweep"
DEFAULT_LAMBDAS = (0.002, 0.003, 0.005, 0.01)
DEFAULT_SEEDS = (0, 1, 2)


def _float_config(config: dict, key: str, default: float = 0.0) -> float:
    value = config.get(key, default)
    if value is None:
        return default
    return float(value)


def _int_config(config: dict, key: str, default: int | None = None) -> int | None:
    value = config.get(key, default)
    if value is None:
        return default
    return int(value)


def _created_at(run) -> datetime:
    value = getattr(run, "created_at", None)
    if value is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    # W&B commonly returns an ISO string ending in Z.
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _norm_layer(config: dict) -> int:
    layer = config.get("sigreg_layer")
    if layer is None:
        return -1
    layer = int(layer)
    return -1 if layer < 0 else layer


def _matches_target(
    run,
    lambdas: set[float],
    seeds: set[int],
    min_group: int | None,
) -> tuple[int, float, float, int] | None:
    cfg = run.config
    layer = _norm_layer(cfg)
    lr_row = _float_config(cfg, "sigreg_lambda_row", 0.0)
    lr_col = _float_config(cfg, "sigreg_lambda_col", 0.0)
    seed = _int_config(cfg, "seed", None)
    run_min_group = _int_config(cfg, "sigreg_min_group", None)

    if seed not in seeds:
        return None
    if min_group is not None and run_min_group != min_group:
        return None

    is_baseline = layer == -1 and lr_row == 0.0 and lr_col == 0.0
    is_col_sweep = layer == 0 and lr_row == 0.0 and lr_col in lambdas
    if not (is_baseline or is_col_sweep):
        return None

    return layer, lr_row, lr_col, seed


def _fetch_history(run, full: bool) -> pd.DataFrame:
    if full:
        return pd.DataFrame(list(run.scan_history(page_size=1000)))
    return run.history(pandas=True, samples=100_000)


def export(args: argparse.Namespace) -> pd.DataFrame:
    api = wandb.Api()
    lambdas = {float(x) for x in args.lambdas}
    seeds = {int(x) for x in args.seeds}
    path = f"{args.entity}/{args.project}"

    selected = {}
    all_matches = []
    for run in api.runs(path, per_page=200):
        key = _matches_target(run, lambdas, seeds, args.min_group)
        if key is None:
            continue
        all_matches.append((key, run))
        current = selected.get(key)
        if current is None or _created_at(run) > _created_at(current):
            selected[key] = run

    if not selected:
        raise RuntimeError(f"No matching runs found in {path}")

    print(f"Matched {len(all_matches)} candidate runs; selected {len(selected)} latest runs")
    for key, run in sorted(selected.items()):
        print(f"  {key}: {run.name} state={run.state} created_at={run.created_at}")

    frames = []
    for key, run in sorted(selected.items()):
        hist = _fetch_history(run, full=args.full_history)
        if hist.empty:
            print(f"  skip empty history: {run.name}")
            continue
        layer, lr_row, lr_col, seed = key
        cfg = run.config
        hist["run"] = run.name
        hist["run_id"] = run.id
        hist["state"] = run.state
        hist["created_at"] = str(run.created_at)
        hist["layer"] = layer if layer >= 0 else pd.NA
        hist["lr_row"] = lr_row
        hist["lr_col"] = lr_col
        hist["seed"] = seed
        hist["sigreg_min_group"] = _int_config(cfg, "sigreg_min_group", pd.NA)
        hist["sigreg_n_proj"] = _int_config(cfg, "sigreg_n_proj", pd.NA)
        frames.append(hist)

    if not frames:
        raise RuntimeError("Matched runs exist, but all histories were empty")

    df = pd.concat(frames, ignore_index=True)
    meta_cols = [
        "run",
        "run_id",
        "state",
        "created_at",
        "layer",
        "lr_row",
        "lr_col",
        "seed",
        "sigreg_min_group",
        "sigreg_n_proj",
        "_step",
        "_runtime",
    ]
    meta_cols = [col for col in meta_cols if col in df.columns]
    metric_cols = [col for col in df.columns if col not in meta_cols]
    df = df[meta_cols + metric_cols]
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--output", type=Path, default=Path("sigreg_latest_history.csv"))
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDAS),
        help="lambda_col values for the col-only follow-up sweep",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Expected seed values",
    )
    parser.add_argument(
        "--min-group",
        type=int,
        default=8,
        help="Only include runs with this sigreg_min_group; use -1 to disable filter",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Use scan_history instead of sampled run.history",
    )
    args = parser.parse_args()
    if args.min_group < 0:
        args.min_group = None
    return args


def main() -> None:
    args = parse_args()
    df = export(args)
    df.to_csv(args.output, index=False)
    print(
        f"Saved {args.output} — {len(df):,} rows, "
        f"{df['run'].nunique()} runs, columns={list(df.columns)}"
    )


if __name__ == "__main__":
    main()
