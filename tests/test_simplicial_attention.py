"""Focused unit tests for sparse simplicial-attention diagnostics."""

from __future__ import annotations

import math
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class SimplicialAttentionTest(unittest.TestCase):
    def test_entropy_averages_active_segments_and_heads(self):
        from rt.simplicial_attention import _mean_segment_entropy

        # Two active segments and two heads. Segment 0 has two edges; segment 1
        # has one deterministic edge. Inactive buffer slots must not dilute the
        # reported mean.
        attention = torch.tensor(
            [[0.5, 0.25], [0.5, 0.75], [1.0, 1.0]], dtype=torch.float32
        )
        segments = torch.tensor([0, 0, 1], dtype=torch.int64)
        got = _mean_segment_entropy(attention, segments, n_segs=8)
        binary_entropy = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
        expected = (math.log(2.0) + binary_entropy) / 4.0
        self.assertAlmostEqual(got.item(), expected, places=6)


if __name__ == "__main__":
    unittest.main()
