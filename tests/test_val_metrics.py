"""Unit tests for rt.val_metrics — the val-AUROC decomposition.

Runs without torch and without a GPU. When scikit-learn is missing (e.g. on a
laptop with no training env) a numpy midrank AUROC is installed in its place, so
the decomposition algebra is still exercised; with scikit-learn present the same
assertions are checked against the real ``roc_auc_score``.

    python3 -m pytest tests/test_val_metrics.py      # or: python3 tests/test_val_metrics.py
"""

import math
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _midrank_auroc(labels_bin, preds):
    """AUROC via midranks — same tie convention as sklearn (ties count 0.5)."""
    y = np.asarray(labels_bin, dtype=float)
    p = np.asarray(preds, dtype=float)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


try:
    from sklearn.metrics import roc_auc_score  # noqa: F401

    HAVE_SKLEARN = True
except ModuleNotFoundError:  # stub it so the algebra is still testable
    HAVE_SKLEARN = False
    metrics = types.ModuleType("sklearn.metrics")
    metrics.roc_auc_score = _midrank_auroc
    sklearn = types.ModuleType("sklearn")
    sklearn.metrics = metrics
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules["sklearn.metrics"] = metrics

from rt.val_metrics import (  # noqa: E402
    auroc_between_ceiling,
    decompose_val_auroc,
    eval_is_due,
    pooled_auroc_parts,
)

# rel-avito val seed counts, from ~/scratch/pre/rel-avito/table_info.json
N_VISITS, N_CLICKS = 29979, 21183


def test_between_ceiling_matches_a_real_constant_per_task_score():
    """The closed form must equal the pooled AUROC actually attained by a score
    that is constant inside each task."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        n_tasks = int(rng.integers(2, 5))
        rates = rng.uniform(0.02, 0.98, size=n_tasks)
        counts, y, p = {}, [], []
        for t in range(n_tasks):
            n = int(rng.integers(50, 400))
            n_pos = min(max(int(round(n * rates[t])), 0), n)
            counts[f"t{t}"] = (n_pos, n - n_pos)
            y += [1] * n_pos + [0] * (n - n_pos)
            p += [float(rates[t])] * n  # constant within task
        assert abs(auroc_between_ceiling(counts) - _midrank_auroc(y, p)) < 1e-9


def test_between_ceiling_brackets_the_target_on_avito_shapes():
    """A score with no within-task information reaches the 0.81621 target as soon
    as the two rel-avito tasks are far enough apart in base rate."""
    def ceiling(p_visits, p_clicks):
        return auroc_between_ceiling({
            "user-visits/num_click": (
                round(N_VISITS * p_visits), round(N_VISITS * (1 - p_visits))
            ),
            "user-clicks/num_click": (
                round(N_CLICKS * p_clicks), round(N_CLICKS * (1 - p_clicks))
            ),
        })

    assert abs(ceiling(0.5, 0.5) - 0.5) < 1e-6      # equal base rates -> nothing free
    assert ceiling(0.7, 0.05) > 0.81621             # target reachable for free
    assert ceiling(0.8, 0.10) > 0.81621
    assert ceiling(0.6, 0.30) < 0.81621             # ... but not unconditionally


def test_between_ceiling_degenerate():
    assert math.isnan(auroc_between_ceiling({}))
    assert math.isnan(auroc_between_ceiling({"a": (10, 0)}))
    assert math.isnan(auroc_between_ceiling({"a": (0, 10)}))
    assert auroc_between_ceiling({"a": (5, 5)}) == 0.5  # one task -> all ties


def _two_task_fixture(seed=1, offset=10.0, signal=1.5):
    rng = np.random.default_rng(seed)
    n_a, n_b = 4000, 3000
    y_a = (rng.random(n_a) < 0.8).astype(int)
    y_b = (rng.random(n_b) < 0.1).astype(int)
    s_a = rng.normal(0, 1, n_a) + signal * y_a + offset
    s_b = rng.normal(0, 1, n_b) + signal * y_b
    return (
        np.concatenate([s_a, s_b]),
        np.concatenate([y_a, y_b]),
        np.concatenate([np.full(n_a, 9), np.full(n_b, 3246415)]),
        {9: "user-visits/num_click", 3246415: "user-clicks/num_click"},
        (y_a, s_a, y_b, s_b),
    )


def test_pooled_splits_exactly_into_within_and_cross():
    """pooled = (1-cw)*within + cw*cross must hold to floating-point precision —
    it is the Mann-Whitney sum regrouped, not an approximation."""
    p, y, tids, _names, _ = _two_task_fixture()
    parts = pooled_auroc_parts(p, y, tids)
    cw = parts["val/pooled_cross_weight"]
    recon = (
        (1 - cw) * parts["val/auroc_pooled_within"]
        + cw * parts["val/auroc_pooled_cross"]
    )
    assert abs(recon - _midrank_auroc(y, p)) < 1e-9


def test_per_task_split_and_weighting():
    p, y, tids, names, (y_a, s_a, y_b, s_b) = _two_task_fixture()
    out = decompose_val_auroc(p, y.tolist(), tids, names)

    a_a, a_b = _midrank_auroc(y_a, s_a), _midrank_auroc(y_b, s_b)
    assert abs(out["val/auroc/user-visits/num_click"] - a_a) < 1e-9
    assert abs(out["val/auroc/user-clicks/num_click"] - a_b) < 1e-9
    assert abs(out["val/auroc_mean_per_task"] - (a_a + a_b) / 2) < 1e-9
    assert out["val/n_tasks"] == 2.0
    assert out["val/n_unmapped_tasks"] == 0.0
    assert out["val/n_pos/user-visits/num_click"] == float(y_a.sum())

    # pooled_within weights per-task AUROCs by n+_g * n-_g
    w_a = y_a.sum() * (1 - y_a).sum()
    w_b = y_b.sum() * (1 - y_b).sum()
    expected = (w_a * a_a + w_b * a_b) / (w_a + w_b)
    assert abs(out["val/auroc_pooled_within"] - expected) < 1e-9

    # a between-task offset inflates pooled above the honest per-task mean
    assert _midrank_auroc(y, p) > out["val/auroc_mean_per_task"] + 0.05


def test_zero_within_task_signal_still_scores_high_pooled():
    """The failure mode the decomposition exists to expose: a score that only
    identifies the task scores ~0.84 pooled while being exactly 0.5 per task."""
    _p, y, tids, names, (y_a, _s_a, y_b, _s_b) = _two_task_fixture()
    n_a = y_a.size
    p0 = np.concatenate([np.full(n_a, 10.0), np.zeros(y_b.size)])
    out = decompose_val_auroc(p0, y.tolist(), tids, names)

    assert out["val/auroc_mean_per_task"] == 0.5
    assert out["val/auroc_pooled_within"] == 0.5
    assert abs(_midrank_auroc(y, p0) - out["val/auroc_between"]) < 1e-9
    assert out["val/auroc_between"] > 0.8
    # only two distinct scores -> essentially everything is a tie
    assert out["val/pred_tie_frac"] > 0.999
    assert out["val/pred_unique"] == 2.0


def test_unmapped_task_id_and_single_class_task():
    p, y, tids, _names, (y_a, s_a, y_b, s_b) = _two_task_fixture()

    out = decompose_val_auroc(p, y.tolist(), tids, {9: "known"})
    assert out["val/n_unmapped_tasks"] == 1.0
    assert "val/auroc/col3246415" in out  # still split, just under the raw id

    # a task with one class only -> nan, and excluded from the mean
    y1 = np.concatenate([np.ones(50, dtype=int), y_b[:100]])
    p1 = np.concatenate([s_a[:50], s_b[:100]])
    t1 = np.concatenate([np.full(50, 9), np.full(100, 3246415)])
    out1 = decompose_val_auroc(
        p1, y1.tolist(), t1, {9: "one-class", 3246415: "both"}
    )
    assert math.isnan(out1["val/auroc/one-class"])
    assert abs(
        out1["val/auroc_mean_per_task"] - _midrank_auroc(y_b[:100], s_b[:100])
    ) < 1e-9


def test_eval_schedule_is_backwards_compatible():
    """dense_until=0 must reproduce the pre-existing schedule exactly, so the
    record runs stay comparable."""
    def old(step, eval_every, at1):  # the original one-liner from main.py
        return not (step % eval_every != 0 and not (step == 1 and at1))

    for at1 in (True, False):
        for every in (1, 5, 100):
            for step in range(1, 400):
                got = eval_is_due(
                    step, eval_every=every, eval_at_step1=at1, dense_until=0
                )
                assert got == old(step, every, at1), (step, every, at1, got)


def test_eval_schedule_dense_window():
    kw = dict(eval_every=100, eval_at_step1=True, dense_until=50, dense_freq=5)
    due = [s for s in range(0, 251) if eval_is_due(s, **kw)]
    # step 0 baseline, step 1, every 5 up to 50, then back to every 100
    assert due == [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200]
    # the step-25 peak the default schedule misses is inside the dense window
    assert 25 in due
    assert 25 not in [
        s for s in range(1, 251)
        if eval_is_due(s, eval_every=100, eval_at_step1=True, dense_until=0)
    ]


def test_step0_always_due_even_when_step1_disabled():
    assert eval_is_due(0, eval_every=100, eval_at_step1=False)
    assert not eval_is_due(1, eval_every=100, eval_at_step1=False)


if __name__ == "__main__":
    print(f"sklearn: {'real' if HAVE_SKLEARN else 'stubbed with midrank AUROC'}")
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                failed += 1
                print(f"  FAIL  {name}: {exc}")
    print("ALL PASSED" if not failed else f"{failed} FAILED")
    raise SystemExit(1 if failed else 0)
