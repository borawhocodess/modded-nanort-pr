"""Decomposition of the zero-shot val AUROC into interpretable parts.

``rt.main`` reports ``val/auroc_zero_shot``: a single AUROC over the concatenated
predictions of *all* rel-avito val tasks. That is not the metric the record
targets — README ("Evaluation details") defines it as the **mean AUROC across the
binary-classification tasks** — and the two diverge as soon as the tasks differ in
base rate, which rel-avito's two val tasks do (``user-visits`` and ``user-clicks``,
both on ``num_click``). This module computes what is needed to tell them apart.

Like ``rt.config``, it deliberately imports **no torch**, so it can be unit-tested
without a GPU environment (``tests/test_val_metrics.py``).
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score


def eval_is_due(
    step: int,
    *,
    eval_every: int,
    eval_at_step1: bool,
    dense_until: int = 0,
    dense_freq: int = 5,
) -> bool:
    """Whether `step` gets a val eval.

    Step 0 is the zero-gradient baseline and is only ever reached by an explicit
    call, so it always answers True. Below `dense_until` the period is
    `dense_freq`, above it `eval_every` — the zero-shot AUROC peaks within the
    first few dozen steps, which the default eval_every=100 steps straight over.

    With ``dense_until=0`` this reproduces the original schedule exactly: every
    `eval_every` steps, plus step 1 when `eval_at_step1`. Note that step 1 is due
    whenever the *period* lands on it, regardless of `eval_at_step1` — that flag
    only ever adds step 1, it never removes it (the original condition was a
    conjunction, and a unit test pins the equivalence).
    """
    if step == 0:
        return True
    period = max(1, dense_freq) if step <= dense_until else eval_every
    return step % period == 0 or (step == 1 and eval_at_step1)


def safe_auroc(labels_bin, preds) -> float:
    """AUROC, or nan when the group holds a single class."""
    if len(set(labels_bin)) < 2:
        return float("nan")
    return roc_auc_score(labels_bin, preds)


def auroc_between_ceiling(counts: dict) -> float:
    """Pooled AUROC of the best score that is CONSTANT within each task.

    ``counts`` maps task name -> (n_pos, n_neg). Such a score carries *zero*
    within-task information — it only says which task a target came from — and
    ranking the tasks by descending base rate is the ordering that maximises the
    pooled AUROC it can reach. So this is the ceiling that pure base-rate
    separation buys for free. Pairs inside one task are ties, worth 0.5 each.

    For rel-avito (``user-visits`` and ``user-clicks``, both ``num_click``, with
    different base rates) this sits in the 0.75-0.90 band — it brackets the
    0.81621 target. Log it next to the pooled AUROC, or the pooled number cannot
    be interpreted.
    """
    order = sorted(
        counts.values(),
        key=lambda c: c[0] / max(c[0] + c[1], 1),
        reverse=True,
    )
    n_pos = sum(p for p, _ in order)
    n_neg = sum(n for _, n in order)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Positives of a higher-ranked task beat negatives of every lower-ranked one.
    num = sum(
        p_i * sum(n_j for _, n_j in order[i + 1:]) for i, (p_i, _) in enumerate(order)
    )
    num += 0.5 * sum(p * n for p, n in order)  # within-task ties
    return num / (n_pos * n_neg)


def pooled_auroc_parts(preds, labels, task_ids) -> dict:
    """Exact split of the pooled AUROC into within-task and cross-task terms.

    Pooled AUROC is a Mann-Whitney average over every (positive, negative) pair,
    so it factors exactly over the task each element came from:

        pooled = Σ_g,h  w_gh · A_gh,        w_gh = n⁺_g·n⁻_h / (N⁺·N⁻)

    with ``A_gh`` = P(a positive of task g scores above a negative of task h),
    ties counting 0.5. The ``g = h`` terms are the per-task AUROCs. The ``g ≠ h``
    terms compare a positive of one task against a negative of *another* — a
    comparison the benchmark never asks for, and precisely what a score that only
    identifies the task exploits. Hence the identity, asserted by the unit test:

        pooled = (1 - cross_weight)·pooled_within + cross_weight·pooled_cross

    ``pooled_cross_weight`` is the fraction of the reported metric that rests on
    cross-task pairs. ``pooled_within`` weights the per-task AUROCs by n⁺_g·n⁻_g,
    so it differs from the unweighted ``auroc_mean_per_task``.
    """
    tasks = np.unique(task_ids)
    pos = {t: np.sort(preds[(task_ids == t) & (labels == 1)]) for t in tasks}
    neg = {t: np.sort(preds[(task_ids == t) & (labels == 0)]) for t in tasks}
    n_pos = sum(v.size for v in pos.values())
    n_neg = sum(v.size for v in neg.values())
    nan = float("nan")
    if n_pos == 0 or n_neg == 0:
        return {
            "val/auroc_pooled_within": nan,
            "val/auroc_pooled_cross": nan,
            "val/pooled_cross_weight": nan,
        }

    w_within = u_within = w_cross = u_cross = 0.0
    for g in tasks:
        p = pos[g]
        if p.size == 0:
            continue
        for h in tasks:
            n = neg[h]
            if n.size == 0:
                continue
            lo = np.searchsorted(n, p, side="left")   # negatives strictly below
            hi = np.searchsorted(n, p, side="right")  # ... and the tied ones
            u = float(lo.sum()) + 0.5 * float((hi - lo).sum())
            w = float(p.size * n.size)
            if g == h:
                w_within, u_within = w_within + w, u_within + u
            else:
                w_cross, u_cross = w_cross + w, u_cross + u

    return {
        "val/auroc_pooled_within": u_within / w_within if w_within else nan,
        "val/auroc_pooled_cross": u_cross / w_cross if w_cross else nan,
        "val/pooled_cross_weight": w_cross / (n_pos * n_neg),
    }


def decompose_val_auroc(preds, labels_bin, task_ids, task_names) -> dict:
    """Split the pooled val AUROC into its per-task and between-task parts.

    ``val/auroc_zero_shot`` pools every val task into a single AUROC, but the
    metric the record targets is the **mean across tasks**. The two diverge as
    soon as the tasks have different base rates, and rel-avito's two val tasks do:
    a predictor that merely tells ``user-visits`` targets from ``user-clicks``
    targets already scores ``auroc_between`` in the pooled number while being
    uninformative within either task. The target cell carries its column-name
    embedding (see the cell-stream init in ``rt/model.py``) and the two tasks have
    different column indices, so that separation is available from the input
    alone, with no training at all.

    Emits, per eval:
      ``auroc/<task>``, ``base_rate/<task>``, ``n_pos/<task>``, ``n_neg/<task>``
      ``auroc_mean_per_task``  — the README metric
      ``auroc_between``        — base-rate-only ceiling (see above)
      ``auroc_pooled_{within,cross}`` + ``pooled_cross_weight`` — exact split of
                                 the pooled number into same-task and cross-task
                                 pairs (see ``pooled_auroc_parts``)
      ``pred_unique`` / ``pred_tie_frac`` — the whole net runs in bf16, so scores
                                 carry ~3 decimal digits and ties are resolved as
                                 0.5. This is the measurement noise floor.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels_bin)
    task_ids = np.asarray(task_ids)
    out, per_task, counts = {}, [], {}

    for tid in np.unique(task_ids):
        sel = task_ids == tid
        name = task_names.get(int(tid), f"col{int(tid)}")
        y, p = labels[sel], preds[sel]
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        counts[name] = (n_pos, n_neg)

        auroc_t = safe_auroc(y.tolist(), p)
        out[f"val/auroc/{name}"] = auroc_t
        out[f"val/base_rate/{name}"] = n_pos / max(n_pos + n_neg, 1)
        out[f"val/n_pos/{name}"] = float(n_pos)
        out[f"val/n_neg/{name}"] = float(n_neg)
        if not math.isnan(auroc_t):
            per_task.append(auroc_t)

    out["val/auroc_mean_per_task"] = (
        sum(per_task) / len(per_task) if per_task else float("nan")
    )
    out["val/auroc_between"] = auroc_between_ceiling(counts)
    out.update(pooled_auroc_parts(preds, labels, task_ids))
    out["val/n_tasks"] = float(len(counts))
    # >0 means the col_name_idxs -> task map missed: the per-task split is still
    # correct (it groups by the observed id), the names are just not resolved.
    out["val/n_unmapped_tasks"] = float(
        sum(1 for tid in np.unique(task_ids) if int(tid) not in task_names)
    )
    n_unique = int(np.unique(preds).size)
    out["val/pred_unique"] = float(n_unique)
    out["val/pred_tie_frac"] = 1.0 - n_unique / max(preds.size, 1)
    return out
