"""Training loop for one experiment (method x assumed scale)."""
import numpy as np
import torch

from .config import ExperimentConfig
from .data import Bundle
from .losses import total_loss
from .model import FieldMLP


def train(b: Bundle, cfg: ExperimentConfig, verbose_every: int = 0):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = FieldMLP(cfg)
    # initialise output head at observation statistics
    with torch.no_grad():
        model.out_mu.fill_(b.z_obs.mean().item())
        model.out_sigma.fill_(b.z_obs.std().item())
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history = []
    for ep in range(cfg.epochs):
        opt.zero_grad()
        z = model(b.xy_grid)
        if ep < cfg.warmup_epochs:  # phase 1: anchor amplitude on obs only
            parts = {"data": torch.mean((z[b.obs_grid_idx] - b.z_obs) ** 2)}
            parts["total"] = parts["data"]
        else:
            parts = total_loss(z, b, cfg)
        parts["total"].backward()
        opt.step()
        sched.step()
        if verbose_every and (ep % verbose_every == 0 or ep == cfg.epochs - 1):
            msg = {k: float(v.detach()) for k, v in parts.items()}
            history.append({"epoch": ep, **msg})
            print(f"[{cfg.name}] ep {ep:5d}  " +
                  "  ".join(f"{k}={v:.4g}" for k, v in msg.items()))
    return model, history
