"""SIGReg — Sketched Isotropic Gaussian Regularization for nanoRT.

Adapts the SIGReg loss from LeJEPA (Balestriero & LeCun, 2025,
arXiv:2511.08544) and "Optimizing Pre-Training of Tabular Foundation Models by
Shaping Geometry" to the *relational* hidden states of nanoRT.

nanoRT's hidden state at a block is a flat sequence of cell embeddings
``x ∈ (B, S, E)`` — not the clean ``(B, R, C, E)`` grid of nanoTabPFN. We
recover the paper's two complementary views from the relational structure that
the model already computes:

  * **row view**  — group cells by ``node_idx`` (the C features of one datapoint)
  * **column view** — group cells by ``(table_name_idx, col_name_idx)`` (the R
    datapoints of one feature; this is exactly what nanoRT's ``col`` attention
    operates over)

Each view gets its own projection head (2-layer MLP with masked BatchNorm) and
the sketched Epps–Pulley statistic is evaluated *per group* and averaged over
groups, mirroring §2.3–2.4 of the paper.

Everything here runs in **fp32** (the rest of the model is bf16): the
characteristic-function quadrature and BatchNorm are numerically sensitive.
The module is **training-only** — it is never invoked at eval, so BatchNorm
always uses batch statistics and needs no running buffers.
"""

import torch
import torch.nn.functional as F
from torch import nn


# ── Epps–Pulley quadrature reference (LeJEPA `EppsPulley`) ─────────────────────
#
# For a 1-D sample {x_n}, the statistic compares the empirical characteristic
# function φ̂(t) = mean_n exp(i t x_n) to that of N(0,1), φ_N(t) = exp(-t²/2),
# integrated against a Gaussian window w(t) = exp(-t²/2):
#
#     T = N · ∫ |φ̂(t) − φ_N(t)|² w(t) dt
#       = N · ∫ [(C(t) − e^{-t²/2})² + S(t)²] e^{-t²/2} dt
#
# with C(t)=mean_n cos(t x_n), S(t)=mean_n sin(t x_n). LeJEPA exploits evenness
# of the integrand to integrate on [0, t_max] with a doubled trapezoidal rule.


class MaskedBatchNorm1d(nn.Module):
    """BatchNorm over the *valid* cells only (training-only, no running stats).

    Standard ``nn.BatchNorm1d`` would fold zero-valued padding cells into the
    batch statistics. Here we compute per-channel mean/var over the non-padding
    cells across the whole (B, S) batch — matching BatchNorm's cross-sample
    behaviour while ignoring padding.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # x: (B, S, C) fp32 ; valid: (B, S) bool
        m = valid[..., None].to(x.dtype)            # (B, S, 1)
        n = m.sum().clamp(min=1.0)                  # scalar: #valid cells
        mean = (x * m).sum(dim=(0, 1)) / n          # (C,)
        var = (((x - mean) ** 2) * m).sum(dim=(0, 1)) / n
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class Projector(nn.Module):
    """VICReg/LeJEPA-style projection head: E → mult·E → … → E/div.

    One or more ``Linear → MaskedBatchNorm → ReLU`` hidden layers followed by a
    final linear projection. Defaults reproduce the paper's 1-hidden-layer
    ``E → 4E → E/4`` projector.
    """

    def __init__(
        self,
        d_in: int,
        hidden_mult: int = 4,
        out_div: int = 4,
        hidden_layers: int = 1,
    ):
        super().__init__()
        d_hidden = d_in * hidden_mult
        d_out = max(1, d_in // out_div)
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        prev = d_in
        for _ in range(hidden_layers):
            self.lins.append(nn.Linear(prev, d_hidden, bias=False))
            self.bns.append(MaskedBatchNorm1d(d_hidden))
            prev = d_hidden
        self.out = nn.Linear(prev, d_out, bias=True)
        self.d_out = d_out

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        for lin, bn in zip(self.lins, self.bns):
            x = F.relu(bn(lin(x), valid))
        return self.out(x)


def _dense_group_ids(keys: torch.Tensor) -> torch.Tensor:
    """Per-row dense relabelling of integer keys into ``[0, S)``.

    ``keys`` is (B, S) of arbitrary integer group keys (e.g. node ids, or packed
    ``(table, col)`` keys). Returns (B, S) int64 where, within each row, distinct
    keys map to distinct contiguous ids in ``[0, n_unique_row)``. Fully
    vectorised (sort + cumsum + scatter) so it is torch.compile-friendly with
    static shapes.
    """
    sorted_keys, sort_idx = keys.sort(dim=1)
    is_new = torch.ones_like(sorted_keys, dtype=torch.bool)
    is_new[:, 1:] = sorted_keys[:, 1:] != sorted_keys[:, :-1]
    dense_sorted = is_new.to(torch.int64).cumsum(dim=1) - 1
    gid = torch.empty_like(keys, dtype=torch.int64)
    gid.scatter_(1, sort_idx, dense_sorted)
    return gid


def _dense_group_ids_global(keys: torch.Tensor) -> torch.Tensor:
    """Global dense relabelling across the whole ``(B, S)`` batch into ``[0, B*S)``.

    Like ``_dense_group_ids`` but pools cells with the same key across *all* rows
    of the batch into a single group, instead of one group per ``(row, key)``.
    Used by the column view so each ``(table, col)`` feature forms one large,
    uniform population (mirroring the paper's clean-grid column groups) rather
    than ~B small per-sequence groups. Fully vectorised (sort + cumsum + scatter)
    so it is torch.compile-friendly with static shapes (sort is over ``B*S``).
    """
    flat = keys.reshape(-1)
    sorted_keys, sort_idx = flat.sort()
    is_new = torch.ones_like(sorted_keys, dtype=torch.bool)
    is_new[1:] = sorted_keys[1:] != sorted_keys[:-1]
    dense_sorted = is_new.to(torch.int64).cumsum(0) - 1
    gid = torch.empty_like(flat, dtype=torch.int64)
    gid.scatter_(0, sort_idx, dense_sorted)
    return gid


def _grouped_epps_pulley(
    z: torch.Tensor,        # (B, S, D) fp32 projected embeddings
    flat_idx: torch.Tensor, # (B*S,) int64 group slot of each cell in [0, n_slots)
    n_slots: int,           # static size of the group buffer (= B*S)
    valid: torch.Tensor,    # (B, S) bool valid (non-padding) cells
    dirs: torch.Tensor,     # (D, K) fp32 unit-norm projection directions
    t: torch.Tensor,        # (T,) fp32 quadrature nodes
    phi: torch.Tensor,      # (T,) fp32 = exp(-t²/2) = φ_N(t) = window
    weights: torch.Tensor,  # (T,) fp32 trapezoidal weights × window
    min_group: float,       # groups with fewer valid cells are ignored
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sketched Epps–Pulley statistic evaluated per group, pooled over groups.

    ``flat_idx`` maps each of the B·S cells to a group slot in ``[0, n_slots)``;
    the caller chooses the grouping — per-``(row, node)`` for the row view, or
    pooled globally per ``(table, col)`` across the whole batch for the column
    view. Padding cells contribute zero to both the per-group sums and counts (so
    they cannot pollute a group even if a key collision puts them in a real one).

    The per-group Epps–Pulley statistic is scaled by the group size ``N_g`` (as
    in LeJEPA), then aggregated as an ``N_g``-weighted mean of the per-group,
    per-slice deviation — i.e. divided by the total valid cells, not the group
    count. This makes the loss **scale-invariant to the grouping**: pooling a
    feature across the batch (≈ N_g·B larger groups) no longer inflates the loss
    magnitude by ~B, so ``λ`` stays comparable across pooled/per-sequence groups
    and batch sizes.
    """
    B, S, D = z.shape
    K = dirs.shape[1]
    T = t.shape[0]
    F_ = K * T

    vmask = valid.to(z.dtype)                            # (B, S)
    p = z @ dirs                                         # (B, S, K)
    x_t = p.unsqueeze(-1) * t                            # (B, S, K, T)
    w_cells = vmask[..., None, None]                     # zero out padding cells
    cos = (torch.cos(x_t) * w_cells).reshape(B * S, F_)
    sin = (torch.sin(x_t) * w_cells).reshape(B * S, F_)

    sum_cos = torch.zeros(n_slots, F_, dtype=z.dtype, device=z.device).index_add_(
        0, flat_idx, cos
    )
    sum_sin = torch.zeros(n_slots, F_, dtype=z.dtype, device=z.device).index_add_(
        0, flat_idx, sin
    )
    cnt = torch.zeros(n_slots, dtype=z.dtype, device=z.device).index_add_(
        0, flat_idx, vmask.reshape(B * S)
    )                                                    # (n_slots,) valid count/group

    denom = cnt.clamp(min=1.0).unsqueeze(-1)             # (n_slots, 1)
    cos_mean = (sum_cos / denom).reshape(n_slots, K, T)
    sin_mean = (sum_sin / denom).reshape(n_slots, K, T)
    err = (cos_mean - phi) ** 2 + sin_mean ** 2          # (n_slots, K, T)
    stat = (err @ weights) * cnt.unsqueeze(-1)           # (n_slots, K) — scaled by N_g
    stat = stat.mean(dim=-1)                             # (n_slots,) mean over slices

    group_present = (cnt > 0).to(z.dtype)
    group_ok = (cnt >= min_group).to(z.dtype)            # (n_slots,)
    num_present = group_present.sum()
    num_groups = group_ok.sum()
    num_groups_safe = num_groups.clamp(min=1.0)
    valid_cells = (cnt * group_ok).sum()
    valid_cells_safe = valid_cells.clamp(min=1.0)
    total_cells = cnt.sum().clamp(min=1.0)

    # N_g-weighted mean of the per-group statistic (scale-invariant to grouping).
    loss = (stat * group_ok).sum() / valid_cells_safe
    stats = {
        "groups_present": num_present,
        "groups_valid": num_groups,
        "groups_valid_frac": num_groups / num_present.clamp(min=1.0),
        "group_size_present_mean": cnt.sum() / num_present.clamp(min=1.0),
        "group_size_valid_mean": valid_cells / num_groups_safe,
        "group_size_valid_max": (cnt * group_ok).max(),
        "cells_covered_frac": valid_cells / total_cells,
    }
    return loss, stats


def _zero_stats(device, dtype=torch.float32) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), device=device, dtype=dtype)
    return {
        "groups_present": zero,
        "groups_valid": zero,
        "groups_valid_frac": zero,
        "group_size_present_mean": zero,
        "group_size_valid_mean": zero,
        "group_size_valid_max": zero,
        "cells_covered_frac": zero,
    }


class SIGReg(nn.Module):
    """Row- and column-wise SIGReg for nanoRT relational hidden states."""

    # packed col key = table * COL_STRIDE + col ; stride safely exceeds any
    # per-table column count.
    COL_STRIDE = 1_000_003

    def __init__(
        self,
        d_model: int,
        n_proj: int = 64,
        proj_hidden_mult: int = 4,
        proj_out_div: int = 4,
        proj_hidden_layers: int = 1,
        knots: int = 17,
        t_max: float = 3.0,
        min_group: int = 8,
        col_global_pool: bool = True,
    ):
        super().__init__()
        self.row_proj = Projector(
            d_model, proj_hidden_mult, proj_out_div, proj_hidden_layers
        )
        self.col_proj = Projector(
            d_model, proj_hidden_mult, proj_out_div, proj_hidden_layers
        )
        self.n_proj = n_proj
        self.min_group = float(min_group)
        # Column view: pool each (table, col) feature across the whole batch into
        # one large population (recommended), vs. one group per sequence (legacy).
        self.col_global_pool = col_global_pool

        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        w = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        w[0] = dt
        w[-1] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)              # φ_N(t) and window
        self.register_buffer("weights", w * window)

    def _sample_dirs(self, d: int, device) -> torch.Tensor:
        a = torch.randn(d, self.n_proj, device=device, dtype=torch.float32)
        return a / (a.norm(dim=0, keepdim=True) + 1e-8)

    def forward(
        self,
        x: torch.Tensor,            # (B, S, E) hidden state at the chosen block
        node_idxs: torch.Tensor,    # (B, S) int
        table_name_idxs: torch.Tensor,
        col_name_idxs: torch.Tensor,
        is_padding: torch.Tensor,   # (B, S) bool
        use_row: bool = True,
        use_col: bool = True,
    ):
        x = x.float()
        valid = ~is_padding
        B, S = node_idxs.shape
        n_slots = B * S
        zero = torch.zeros((), device=x.device, dtype=torch.float32)
        stats: dict[str, torch.Tensor] = {}

        # row view: group cells by datapoint (node), independently per sequence.
        # Nodes are sequence-local, so the row view always pools per (row, node).
        if use_row:
            row_gid = _dense_group_ids(node_idxs.to(torch.int64))   # (B, S) in [0, S)
            row_base = (
                torch.arange(B, device=x.device, dtype=torch.int64) * S
            ).unsqueeze(1)
            row_flat = (row_base + row_gid).reshape(n_slots)
            z_row = self.row_proj(x, valid)
            row_loss, row_stats = _grouped_epps_pulley(
                z_row, row_flat, n_slots, valid,
                self._sample_dirs(z_row.shape[-1], x.device),
                self.t, self.phi, self.weights, self.min_group,
            )
        else:
            row_loss = zero
            row_stats = _zero_stats(x.device)
        for key, value in row_stats.items():
            stats[f"row/{key}"] = value

        # column view: group cells by feature = (table, column). With
        # col_global_pool, the same feature is pooled across the *whole* batch
        # into one large, uniform population (mirroring the paper's clean-grid
        # column groups); otherwise it falls back to per-sequence grouping.
        if use_col:
            col_keys = (
                table_name_idxs.to(torch.int64) * self.COL_STRIDE
                + col_name_idxs.to(torch.int64)
            )
            if self.col_global_pool:
                col_flat = _dense_group_ids_global(col_keys)        # (B*S,) in [0, B*S)
            else:
                col_gid = _dense_group_ids(col_keys)
                col_base = (
                    torch.arange(B, device=x.device, dtype=torch.int64) * S
                ).unsqueeze(1)
                col_flat = (col_base + col_gid).reshape(n_slots)
            z_col = self.col_proj(x, valid)
            col_loss, col_stats = _grouped_epps_pulley(
                z_col, col_flat, n_slots, valid,
                self._sample_dirs(z_col.shape[-1], x.device),
                self.t, self.phi, self.weights, self.min_group,
            )
        else:
            col_loss = zero
            col_stats = _zero_stats(x.device)
        for key, value in col_stats.items():
            stats[f"col/{key}"] = value

        return row_loss, col_loss, stats
