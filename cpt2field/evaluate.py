"""Metrics with the same definitions as results_summary_from_html.csv:
rmse / mae / r2 on the full 1681-point field, obs_rmse on the 123 CPT points.
"""
import numpy as np
import torch

from .data import Bundle


@torch.no_grad()
def evaluate(model, b: Bundle) -> dict:
    z = model(b.xy_grid)
    err = (z - b.z_true).cpu().numpy()
    zt = b.z_true.cpu().numpy()
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    r2 = float(1 - np.sum(err ** 2) / np.sum((zt - zt.mean()) ** 2))
    obs_err = (z[b.obs_grid_idx] - b.z_obs).cpu().numpy()
    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "obs_rmse": float(np.sqrt(np.mean(obs_err ** 2))),
    }
