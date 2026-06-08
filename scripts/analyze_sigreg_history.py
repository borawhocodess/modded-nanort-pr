"""Summarize and plot dense SIGReg W&B history CSV exports.

The CSV is expected to contain one row per logged metric event, not one row per
training step. This script collapses it into run/config summaries and a small
set of readable plots.

Usage:
    python3 scripts/analyze_sigreg_history.py sigreg_history.csv
    python3 scripts/analyze_sigreg_history.py sigreg_history.csv --dedupe latest
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/nanort-mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_AUROC = 0.81621
BUDGET_SECONDS = 45 * 60


def _clean_layer(value: float) -> int:
    if pd.isna(value):
        return -1
    return int(value)


def _config_label(layer: int, lr_row: float, lr_col: float) -> str:
    if layer < 0:
        return "baseline"
    return f"L{layer} row={lr_row:g} col={lr_col:g}"


def _metric_last(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def _metric_max(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.max()) if len(s) else np.nan


def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"run", "layer", "lr_row", "lr_col", "seed", "_step"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["layer_clean"] = df["layer"].map(_clean_layer)
    df["config"] = [
        _config_label(layer, row, col)
        for layer, row, col in zip(df["layer_clean"], df["lr_row"], df["lr_col"])
    ]
    df["weighted_sigreg_row"] = df["lr_row"] * df.get("sigreg/row", np.nan)
    df["weighted_sigreg_col"] = df["lr_col"] * df.get("sigreg/col", np.nan)
    df["weighted_sigreg_total"] = (
        df["weighted_sigreg_row"].fillna(0.0) + df["weighted_sigreg_col"].fillna(0.0)
    )
    df["weighted_sup_loss"] = (
        1.0 - df["lr_row"].fillna(0.0) - df["lr_col"].fillna(0.0)
    ) * df.get("loss", np.nan)
    return df


def summarize_runs(df: pd.DataFrame, target: float) -> pd.DataFrame:
    rows = []
    for run, g in df.groupby("run", sort=False):
        g = g.sort_values("_step")
        first = g.iloc[0]
        val = g[g["val/auroc_zero_shot"].notna()].copy()
        hit_rows = val[val["val/auroc_zero_shot"] >= target]
        ttt = g["train/time_to_target"].dropna()
        ttt_hit = ttt[ttt >= 0]
        logged_hit_time = float(ttt_hit.min()) if len(ttt_hit) else np.nan

        if len(hit_rows):
            first_hit = hit_rows.iloc[0]
            inferred_hit_time = first_hit.get("train/elapsed_seconds", np.nan)
            if pd.isna(inferred_hit_time):
                inferred_hit_time = first_hit.get("_runtime", np.nan)
            first_hit_step = int(first_hit["_step"])
        else:
            inferred_hit_time = np.nan
            first_hit_step = np.nan

        if not pd.isna(logged_hit_time):
            time_to_target = logged_hit_time
        else:
            time_to_target = inferred_hit_time

        reached = (not pd.isna(time_to_target)) or (
            len(hit_rows) > 0
        )
        has_final_censor = bool((ttt < 0).any())
        max_auroc = _metric_max(g.get("val/auroc_zero_shot", pd.Series(dtype=float)))
        if len(val):
            best_row = val.loc[val["val/auroc_zero_shot"].idxmax()]
            best_step = int(best_row["_step"])
            best_elapsed = best_row.get("train/elapsed_seconds", np.nan)
        else:
            best_step = np.nan
            best_elapsed = np.nan

        rows.append(
            {
                "run": run,
                "state": first.get("state", ""),
                "layer": int(first["layer_clean"]),
                "lr_row": float(first["lr_row"]),
                "lr_col": float(first["lr_col"]),
                "seed": int(first["seed"]),
                "config": first["config"],
                "rows": len(g),
                "max_step": int(g["_step"].max()),
                "max_elapsed": _metric_max(
                    g.get("train/elapsed_seconds", pd.Series(dtype=float))
                ),
                "best_auroc": max_auroc,
                "best_auroc_step": best_step,
                "best_auroc_elapsed": best_elapsed,
                "last_auroc": _metric_last(
                    g.get("val/auroc_zero_shot", pd.Series(dtype=float))
                ),
                "first_hit_step": first_hit_step,
                "time_to_target": time_to_target,
                "logged_time_to_target": logged_hit_time,
                "has_final_censor": has_final_censor,
                "target_reached": bool(reached),
                "loss_last": _metric_last(g.get("loss", pd.Series(dtype=float))),
                "loss_total_last": _metric_last(
                    g.get("loss_total", pd.Series(dtype=float))
                ),
                "weighted_sup_loss_last": _metric_last(
                    g.get("weighted_sup_loss", pd.Series(dtype=float))
                ),
                "weighted_sigreg_row_last": _metric_last(
                    g.get("weighted_sigreg_row", pd.Series(dtype=float))
                ),
                "weighted_sigreg_col_last": _metric_last(
                    g.get("weighted_sigreg_col", pd.Series(dtype=float))
                ),
                "sigreg_row_last": _metric_last(
                    g.get("sigreg/row", pd.Series(dtype=float))
                ),
                "sigreg_col_last": _metric_last(
                    g.get("sigreg/col", pd.Series(dtype=float))
                ),
                "erank_last": _metric_last(
                    g.get("sigreg/effective_rank", pd.Series(dtype=float))
                ),
                "mean_dim_std_last": _metric_last(
                    g.get("sigreg/mean_dim_std", pd.Series(dtype=float))
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["censored_time"] = np.where(
        out["target_reached"], out["time_to_target"], BUDGET_SECONDS
    )
    return out.sort_values(["layer", "lr_row", "lr_col", "seed", "run"])


def dedupe_runs(run_summary: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "none":
        return run_summary.copy()

    def pick(group: pd.DataFrame) -> pd.Series:
        if mode == "latest":
            return group.sort_values(["max_step", "max_elapsed", "run"]).iloc[-1]
        if mode == "longest":
            return group.sort_values(["max_elapsed", "max_step", "run"]).iloc[-1]
        if mode == "best-auroc":
            return group.sort_values(["best_auroc", "max_step", "run"]).iloc[-1]
        raise ValueError(f"Unknown dedupe mode: {mode}")

    keys = ["layer", "lr_row", "lr_col", "seed"]
    return run_summary.groupby(keys, dropna=False, as_index=False).apply(
        pick, include_groups=False
    ).reset_index(drop=True)


def summarize_configs(run_summary: pd.DataFrame) -> pd.DataFrame:
    def hit_mean(series: pd.Series) -> float:
        s = series.dropna()
        return float(s.mean()) if len(s) else np.nan

    def hit_median(series: pd.Series) -> float:
        s = series.dropna()
        return float(s.median()) if len(s) else np.nan

    cfg = (
        run_summary.groupby(["layer", "lr_row", "lr_col", "config"], sort=True)
        .agg(
            runs=("run", "count"),
            seeds=("seed", lambda x: ",".join(map(str, sorted(x.tolist())))),
            unique_seeds=("seed", "nunique"),
            reached=("target_reached", "sum"),
            success_rate=("target_reached", "mean"),
            best_auroc_mean=("best_auroc", "mean"),
            best_auroc_std=("best_auroc", "std"),
            best_auroc_median=("best_auroc", "median"),
            hit_time_mean=("time_to_target", hit_mean),
            hit_time_median=("time_to_target", hit_median),
            censored_time_mean=("censored_time", "mean"),
            max_step_median=("max_step", "median"),
            weighted_row_last_mean=("weighted_sigreg_row_last", "mean"),
            weighted_col_last_mean=("weighted_sigreg_col_last", "mean"),
            erank_last_mean=("erank_last", "mean"),
            mean_dim_std_last_mean=("mean_dim_std_last", "mean"),
        )
        .reset_index()
    )
    cfg["failures"] = cfg["runs"] - cfg["reached"]
    return cfg.sort_values(
        ["success_rate", "censored_time_mean", "best_auroc_mean"],
        ascending=[False, True, False],
    )


def duplicate_table(run_summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["layer", "lr_row", "lr_col", "seed"]
    counts = (
        run_summary.groupby(keys, dropna=False)
        .agg(
            n=("run", "count"),
            runs=("run", lambda x: ", ".join(x)),
            config=("config", "first"),
        )
        .reset_index()
    )
    return counts[counts["n"] > 1].sort_values(keys)


def _save_table_md(df: pd.DataFrame, path: Path, floatfmt: str = ".4f") -> None:
    path.write_text(df.to_markdown(index=False, floatfmt=floatfmt) + "\n")


def plot_outcomes(cfg: pd.DataFrame, outdir: Path) -> None:
    labels = cfg["config"].tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(max(12, len(labels) * 1.8), 4.5))

    axes[0].bar(x, cfg["success_rate"], color="#4C78A8")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("success rate")
    axes[0].set_title("Target reached")

    axes[1].bar(x, cfg["censored_time_mean"], color="#F58518")
    axes[1].axhline(BUDGET_SECONDS, color="black", linewidth=1, linestyle="--")
    axes[1].set_ylabel("seconds")
    axes[1].set_title("Mean time, failures censored at budget")

    axes[2].bar(x, cfg["best_auroc_mean"], color="#54A24B")
    axes[2].axhline(TARGET_AUROC, color="black", linewidth=1, linestyle="--")
    axes[2].set_ylabel("AUROC")
    axes[2].set_title("Best AUROC")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / "outcomes_by_config.png", dpi=180)
    plt.close(fig)


def plot_time_to_target(run_summary: pd.DataFrame, cfg: pd.DataFrame, outdir: Path) -> None:
    labels = cfg["config"].tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.8), 5.0))

    means = cfg["censored_time_mean"].to_numpy()
    bars = ax.bar(x, means, color="#F58518", alpha=0.82, label="mean over seeds")
    ax.axhline(
        BUDGET_SECONDS,
        color="black",
        linewidth=1.2,
        linestyle="--",
        label=f"budget ({BUDGET_SECONDS:.0f}s)",
    )

    for i, config in enumerate(labels):
        rows = run_summary[run_summary["config"] == config].sort_values("seed")
        offsets = np.linspace(-0.18, 0.18, max(len(rows), 1))
        for offset, (_, row) in zip(offsets, rows.iterrows()):
            reached = bool(row["target_reached"])
            y = row["censored_time"]
            ax.scatter(
                i + offset,
                y,
                marker="o" if reached else "x",
                s=55,
                color="#4C78A8" if reached else "#D62728",
                zorder=3,
            )
            ax.text(
                i + offset,
                y + 45,
                str(int(row["seed"])),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )
        reached_count = int(rows["target_reached"].sum())
        ax.text(
            i,
            means[i] + 120,
            f"{reached_count}/{len(rows)}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 25, BUDGET_SECONDS + 90),
            f"{value:.0f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("seconds")
    ax.set_title("Mean time to target across seeds (failures censored at budget)")
    ax.set_ylim(0, BUDGET_SECONDS * 1.16)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "time_to_target_by_config.png", dpi=180)
    plt.close(fig)


def plot_auroc_curves(df: pd.DataFrame, run_summary: pd.DataFrame, outdir: Path) -> None:
    configs = run_summary["config"].drop_duplicates().tolist()
    n = len(configs)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, config in zip(axes, configs):
        gcfg = df[df["config"] == config]
        val = gcfg[gcfg["val/auroc_zero_shot"].notna()]
        for run, gr in val.groupby("run"):
            gr = gr.sort_values("_step")
            ax.plot(
                gr["_step"],
                gr["val/auroc_zero_shot"],
                marker="o",
                linewidth=0.8,
                alpha=0.35,
                label=run,
            )

        mean = (
            val.groupby("_step")["val/auroc_zero_shot"]
            .agg(["mean", "count", "std"])
            .reset_index()
            .sort_values("_step")
        )
        if len(mean):
            ax.plot(
                mean["_step"],
                mean["mean"],
                color="black",
                linewidth=2.2,
                marker="o",
                label="mean",
            )
        ax.axhline(TARGET_AUROC, color="#D62728", linestyle="--", linewidth=1)
        ax.set_title(config)
        ax.set_xlabel("step")
        ax.set_ylabel("val AUROC")
        ax.grid(alpha=0.25)

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(outdir / "auroc_curves_by_config.png", dpi=180)
    plt.close(fig)


def plot_sigreg_terms(df: pd.DataFrame, outdir: Path) -> None:
    sig = df[df["layer_clean"] >= 0].copy()
    metrics = [
        ("weighted_sigreg_row", "lambda_row * sigreg/row"),
        ("weighted_sigreg_col", "lambda_col * sigreg/col"),
        ("weighted_sup_loss", "(1 - lambdas) * loss"),
        ("sigreg/effective_rank", "effective rank"),
        ("sigreg/mean_dim_std", "mean dim std"),
    ]
    metrics = [
        (metric, title)
        for metric, title in metrics
        if metric in sig.columns and sig[metric].notna().any()
    ]
    if not metrics:
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.text(0.5, 0.5, "No SIGReg term columns found", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(outdir / "sigreg_terms_by_step.png", dpi=180)
        plt.close(fig)
        return
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3.0 * len(metrics)))
    axes = np.atleast_1d(axes)

    for ax, (metric, title) in zip(axes, metrics):
        if metric not in sig:
            ax.axis("off")
            continue
        rows = sig[sig[metric].notna()]
        for config, gr in rows.groupby("config"):
            mean = (
                gr.groupby("_step")[metric]
                .mean()
                .reset_index()
                .sort_values("_step")
            )
            ax.plot(mean["_step"], mean[metric], linewidth=1.6, label=config)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)

    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "sigreg_terms_by_step.png", dpi=180)
    plt.close(fig)


def plot_group_stats(df: pd.DataFrame, outdir: Path) -> None:
    metrics = [
        ("sigreg/col/groups_valid", "col valid groups"),
        ("sigreg/col/groups_valid_frac", "col valid group fraction"),
        ("sigreg/col/group_size_valid_mean", "col mean valid group size"),
        ("sigreg/col/cells_covered_frac", "col covered cell fraction"),
        ("sigreg/row/groups_valid", "row valid groups"),
        ("sigreg/row/groups_valid_frac", "row valid group fraction"),
        ("sigreg/row/group_size_valid_mean", "row mean valid group size"),
        ("sigreg/row/cells_covered_frac", "row covered cell fraction"),
    ]
    metrics = [(metric, title) for metric, title in metrics if metric in df.columns]
    if not metrics:
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.text(0.5, 0.5, "No SIGReg group-stat columns found", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(outdir / "group_stats_by_step.png", dpi=180)
        plt.close(fig)
        return

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 2.8 * len(metrics)))
    axes = np.atleast_1d(axes)
    for ax, (metric, title) in zip(axes, metrics):
        rows = df[df[metric].notna()]
        for config, gr in rows.groupby("config"):
            mean = (
                gr.groupby("_step")[metric]
                .mean()
                .reset_index()
                .sort_values("_step")
            )
            ax.plot(mean["_step"], mean[metric], linewidth=1.6, label=config)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "group_stats_by_step.png", dpi=180)
    plt.close(fig)


def plot_heatmaps(cfg: pd.DataFrame, outdir: Path) -> None:
    layer0 = cfg[cfg["layer"] == 0]
    if layer0.empty:
        return

    rows = sorted(layer0["lr_row"].unique())
    cols = sorted(layer0["lr_col"].unique())

    def matrix(metric: str) -> np.ndarray:
        mat = np.full((len(rows), len(cols)), np.nan)
        for i, lr_row in enumerate(rows):
            for j, lr_col in enumerate(cols):
                hit = layer0[(layer0["lr_row"] == lr_row) & (layer0["lr_col"] == lr_col)]
                if len(hit):
                    mat[i, j] = hit.iloc[0][metric]
        return mat

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    specs = [
        ("best_auroc_mean", "mean best AUROC", TARGET_AUROC),
        ("success_rate", "success rate", None),
    ]
    for ax, (metric, title, threshold) in zip(axes, specs):
        mat = matrix(metric)
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels([f"{x:g}" for x in cols])
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels([f"{x:g}" for x in rows])
        ax.set_xlabel("lambda_col")
        ax.set_ylabel("lambda_row")
        ax.set_title(title)
        for i in range(len(rows)):
            for j in range(len(cols)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if threshold is not None:
            im.set_clim(min(np.nanmin(mat), threshold), max(np.nanmax(mat), threshold))

    fig.tight_layout()
    fig.savefig(outdir / "layer0_lambda_heatmaps.png", dpi=180)
    plt.close(fig)


def write_index(outdir: Path) -> None:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SIGReg history report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; }
    img { max-width: 100%; border: 1px solid #ddd; margin: 16px 0 32px; }
    a { display: block; margin: 6px 0; }
  </style>
</head>
<body>
  <h1>SIGReg history report</h1>
  <h2>Tables</h2>
  <a href="config_summary.csv">config_summary.csv</a>
  <a href="run_summary.csv">run_summary.csv</a>
  <a href="duplicates.csv">duplicates.csv</a>
  <h2>Plots</h2>
  <h3>Outcomes</h3>
  <img src="outcomes_by_config.png">
  <h3>Mean Time To Target</h3>
  <img src="time_to_target_by_config.png">
  <h3>AUROC Curves</h3>
  <img src="auroc_curves_by_config.png">
  <h3>SIGReg Terms</h3>
  <img src="sigreg_terms_by_step.png">
  <h3>Group Stats</h3>
  <img src="group_stats_by_step.png">
  <h3>Layer 0 Lambda Heatmaps</h3>
  <img src="layer0_lambda_heatmaps.png">
</body>
</html>
"""
    (outdir / "index.html").write_text(html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to sigreg_history.csv")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("sigreg_report"),
        help="Directory for tables and plots",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=TARGET_AUROC,
        help="AUROC target used for time-to-target/success summaries",
    )
    parser.add_argument(
        "--dedupe",
        choices=["none", "latest", "longest", "best-auroc"],
        default="none",
        help=(
            "How to collapse duplicate runs with the same layer/lambda/seed. "
            "Default keeps duplicates and reports them."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write summary CSV/Markdown files; skip PNG generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    history = load_history(args.csv)
    run_summary_all = summarize_runs(history, args.target)
    duplicates = duplicate_table(run_summary_all)
    run_summary = dedupe_runs(run_summary_all, args.dedupe)
    config_summary = summarize_configs(run_summary)

    run_summary_all.to_csv(args.outdir / "run_summary_all.csv", index=False)
    run_summary.to_csv(args.outdir / "run_summary.csv", index=False)
    config_summary.to_csv(args.outdir / "config_summary.csv", index=False)
    duplicates.to_csv(args.outdir / "duplicates.csv", index=False)
    _save_table_md(config_summary, args.outdir / "config_summary.md")

    plot_history = history[history["run"].isin(run_summary["run"])]
    if not args.no_plots:
        plot_outcomes(config_summary, args.outdir)
        plot_time_to_target(run_summary, config_summary, args.outdir)
        plot_auroc_curves(plot_history, run_summary, args.outdir)
        plot_sigreg_terms(plot_history, args.outdir)
        plot_group_stats(plot_history, args.outdir)
        plot_heatmaps(config_summary, args.outdir)
    write_index(args.outdir)

    print(f"Loaded {len(history):,} history rows from {args.csv}")
    print(f"Runs: {run_summary_all['run'].nunique()} total")
    if args.dedupe != "none":
        print(f"Runs after --dedupe {args.dedupe}: {run_summary['run'].nunique()}")
    if len(duplicates):
        print("\nDuplicate layer/lambda/seed groups:")
        print(duplicates.to_string(index=False))
    print("\nConfig summary:")
    cols = [
        "config",
        "runs",
        "seeds",
        "reached",
        "success_rate",
        "best_auroc_mean",
        "best_auroc_std",
        "hit_time_mean",
        "censored_time_mean",
        "weighted_row_last_mean",
        "weighted_col_last_mean",
        "erank_last_mean",
    ]
    print(config_summary[cols].to_string(index=False))
    print(f"\nWrote report to: {args.outdir / 'index.html'}")


if __name__ == "__main__":
    main()
