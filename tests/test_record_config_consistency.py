"""Keep every public record launcher aligned with ``TrainConfig`` defaults."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from rt.config import TrainConfig

ROOT = Path(__file__).resolve().parents[1]


def _run_sh_defaults() -> dict[str, str]:
    text = (ROOT / "run.sh").read_text()
    return dict(
        re.findall(
            r'^export (NANORT_[A-Z0-9_]+)="\$\{\1:-([^}]*)\}"$',
            text,
            flags=re.MULTILINE,
        )
    )


def _polyaxon_input_defaults() -> dict[str, str]:
    """Parse the simple ``inputs: - name/value`` prefix without a YAML dependency."""
    text = (ROOT / "plx_nanort_train.yaml").read_text()
    inputs_text = text.split("\nrun:\n", maxsplit=1)[0]
    defaults: dict[str, str] = {}
    current_name: str | None = None
    for line in inputs_text.splitlines():
        name_match = re.match(r"  - name: ([a-z0-9_]+)$", line)
        if name_match:
            current_name = name_match.group(1)
            continue
        value_match = re.match(r'    value: "?([^"\n]+)"?$', line)
        if value_match and current_name is not None:
            defaults[current_name] = value_match.group(1)
    return defaults


class RecordConfigConsistencyTest(unittest.TestCase):
    def test_shell_launcher_matches_record_defaults(self):
        cfg = TrainConfig()
        expected = {
            "NANORT_BATCH_SIZE": str(cfg.batch_size),
            "NANORT_OPTIMIZER": cfg.optimizer,
            "NANORT_LR": str(cfg.lr),
            "NANORT_MUON_LR": str(cfg.muon_lr),
            "NANORT_MUON_WARMUP_STEPS": str(cfg.muon_warmup_steps),
            "NANORT_SIMPLICIAL_LAYERS": "all",
            "NANORT_UPDATE_RELATIONS": "0",
            "NANORT_QK_NORM": "1",
            "NANORT_SIMPLICIAL_LAYERSCALE_INIT": str(cfg.simplicial_layerscale_init),
            "NANORT_SIGREG_LAYER": str(cfg.sigreg_layer),
            "NANORT_SIGREG_LAMBDA_ROW": str(cfg.sigreg_lambda_row),
            "NANORT_SIGREG_LAMBDA_COL": str(cfg.sigreg_lambda_col),
            "NANORT_SIGREG_N_PROJ": str(cfg.sigreg_n_proj),
            "NANORT_SIGREG_MIN_GROUP": str(cfg.sigreg_min_group),
            "NANORT_SIGREG_COL_POOL": "1",
            "NANORT_EVAL_AT_STEP0": "1",
            "NANORT_EVAL_AT_STEP1": "0",
            "NANORT_VAL_EVAL_FREQ": str(cfg.val_eval_freq),
            "NANORT_VAL_EVAL_DENSE_UNTIL": str(cfg.val_eval_dense_until),
            "NANORT_TIME_BUDGET_MINUTES": str(int(cfg.time_budget_minutes)),
        }
        defaults = _run_sh_defaults()
        self.assertEqual({key: defaults.get(key) for key in expected}, expected)

    def test_polyaxon_launcher_matches_record_defaults(self):
        cfg = TrainConfig()
        expected = {
            "batch_size": str(cfg.batch_size),
            "optimizer": cfg.optimizer,
            "lr": str(cfg.lr),
            "muon_lr": str(cfg.muon_lr),
            "muon_warmup_steps": str(cfg.muon_warmup_steps),
            "simplicial_layers": "all",
            "update_relations": "0",
            "qk_norm": "1",
            "simplicial_layerscale_init": str(cfg.simplicial_layerscale_init),
            "sigreg_layer": str(cfg.sigreg_layer),
            "sigreg_lambda_row": str(cfg.sigreg_lambda_row),
            "sigreg_lambda_col": str(cfg.sigreg_lambda_col),
            "sigreg_n_proj": str(cfg.sigreg_n_proj),
            "sigreg_min_group": str(cfg.sigreg_min_group),
            "sigreg_col_pool": "1",
            "eval_at_step0": "1",
            "eval_at_step1": "0",
            "val_eval_freq": str(cfg.val_eval_freq),
            "val_eval_dense_until": str(cfg.val_eval_dense_until),
            "time_budget_minutes": str(int(cfg.time_budget_minutes)),
        }
        defaults = _polyaxon_input_defaults()
        self.assertEqual({key: defaults.get(key) for key in expected}, expected)


if __name__ == "__main__":
    unittest.main()
