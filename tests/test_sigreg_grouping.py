"""Focused tests for SIGReg grouping keys (GPU not required)."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # local documentation-only environments
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class SigregGroupingTest(unittest.TestCase):
    def test_table_column_pairing_has_no_fixed_stride_collision(self):
        from rt.sigreg import _pair_nonnegative_ids

        # These collide under the old table * 1_000_003 + column encoding.
        table = torch.tensor([1, 0, 7, 7])
        column = torch.tensor([0, 1_000_003, 3_246_415, 3_246_416])
        keys = _pair_nonnegative_ids(table, column)
        self.assertEqual(keys.unique().numel(), 4)

    def test_identical_pairs_share_a_key(self):
        from rt.sigreg import _pair_nonnegative_ids

        table = torch.tensor([9, 9, 10])
        column = torch.tensor([3_246_415, 3_246_415, 3_246_415])
        keys = _pair_nonnegative_ids(table, column)
        self.assertEqual(keys[0].item(), keys[1].item())
        self.assertNotEqual(keys[0].item(), keys[2].item())


if __name__ == "__main__":
    unittest.main()
