"""Differentiable losses. The ACF estimator mirrors acf.py exactly:
global mean + biased variance of the point set the ACF is evaluated on,
correlation = mean of centred pair products / variance.

`acf_on="grid"`: ACF of the *predicted full field* (all grid columns /
column pairs) is pushed toward the CPT experimental targets — this is
what shapes the reconstruction away from the data columns.
`acf_on="obs"`: ACF evaluated only on predictions at the CPT locations.
"""
import torch

from .config import ExperimentConfig
from .data import Bundle


def _acf(z: torch.Tensor, pairs, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    ia, ib = pairs
    return ((z[ia] - mu) * (z[ib] - mu)).mean() / var


def total_loss(z_grid: torch.Tensor, b: Bundle, cfg: ExperimentConfig):
    w = cfg.weights
    parts = {}

    # 1) data loss at the 123 CPT points
    z_at_obs = z_grid[b.obs_grid_idx]
    parts["data"] = w.data * torch.mean((z_at_obs - b.z_obs) ** 2)

    # reference stats for the ACF estimator
    if cfg.acf_on == "grid":
        zs, v_pairs, h_pairs = z_grid, b.v_pairs_grid, b.h_pairs_grid
    else:
        zs, v_pairs, h_pairs = z_at_obs, None, None  # obs mode uses grid indexing below
    mu = zs.mean()
    var = zs.var(unbiased=False).clamp_min(1e-8)

    # 2) vertical experimental ACF loss
    if cfg.acf_on == "grid":
        acf_v = torch.stack([_acf(z_grid, p, mu, var) for p in b.v_pairs_grid])
        acf_h = torch.stack([_acf(z_grid, p, mu, var) for p in b.h_pairs_grid])
    else:
        acf_v = torch.stack([_acf(z_grid, p, mu, var) for p in b.v_pairs_obs])
        acf_h = torch.stack([_acf(z_grid, p, mu, var) for p in b.h_pairs_obs])
    parts["acf_v"] = w.acf_vertical * torch.mean((acf_v - b.v_target) ** 2)

    # 3) horizontal weighted ACF loss (reliability weights 1.0 / 0.1)
    parts["acf_h"] = w.acf_horizontal * torch.sum(
        b.h_weight * (acf_h - b.h_target) ** 2) / b.h_weight.sum()

    # 4) copy4: pair-level horizontal product constraints
    if cfg.use_pair_loss:
        prod = (z_grid[b.pp_i] - mu) * (z_grid[b.pp_j] - mu) / var
        parts["pair"] = w.pair_product * torch.sum(
            b.pp_weight * (prod - b.pp_target) ** 2) / b.pp_weight.sum()

    parts["total"] = sum(parts.values())
    return parts
