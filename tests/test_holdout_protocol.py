"""CPU-only checks for the physical and logical holdout boundary."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rt.config import TrainConfig
from scripts import nanort_train


class HoldoutProtocolTest(unittest.TestCase):
    def _artifact_tree(self, root: Path) -> None:
        for db in nanort_train.HOLDOUT_CANDIDATES:
            (root / db).mkdir()
        (root / "rel-avito" / "table_info.json").write_text(
            json.dumps({"user-visits:Val": {}, "user-clicks:Val": {}})
        )

    def test_resolver_uses_exactly_five_train_databases_and_one_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._artifact_tree(root)
            with patch.object(nanort_train, "PRE_ROOT", root):
                cfg = nanort_train.resolve_tasks(TrainConfig())

        self.assertEqual(
            {task[0] for task in cfg.train_tasks},
            set(nanort_train.HOLDOUT_CANDIDATES) - {"rel-avito"},
        )
        self.assertEqual({task[0] for task in cfg.val_tasks}, {"rel-avito"})
        self.assertEqual({task[3] for task in cfg.val_tasks}, {"Val"})

    def test_resolver_rejects_holdout_symlinked_to_training_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._artifact_tree(root)
            holdout_path = root / "rel-avito"
            (holdout_path / "table_info.json").unlink()
            holdout_path.rmdir()
            (root / "rel-amazon" / "table_info.json").write_text(
                json.dumps({"user-visits:Val": {}, "user-clicks:Val": {}})
            )
            holdout_path.symlink_to(root / "rel-amazon", target_is_directory=True)

            with (
                patch.object(nanort_train, "PRE_ROOT", root),
                self.assertRaisesRegex(RuntimeError, "Holdout isolation failed"),
            ):
                nanort_train.resolve_tasks(TrainConfig())

    def test_resolver_rejects_duplicate_training_artifact_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._artifact_tree(root)
            duplicate_path = root / "rel-hm"
            duplicate_path.rmdir()
            duplicate_path.symlink_to(root / "rel-amazon", target_is_directory=True)

            with (
                patch.object(nanort_train, "PRE_ROOT", root),
                self.assertRaisesRegex(RuntimeError, "same physical artifact"),
            ):
                nanort_train.resolve_tasks(TrainConfig())


if __name__ == "__main__":
    unittest.main()
