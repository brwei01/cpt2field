"""Coordinate MLP: (x, y) -> z_hat.

copy2: inputs are domain-normalized coordinates (x/Lx, y/Ly).
copy4 ("scale_features"): additionally feeds scale-normalized coordinates
(x/a_h, y/a_v), injecting the assumed anisotropic fluctuation scales as
input features. NOTE: the exact feature set of the original notebook is
unknown (not shared); this is the documented assumption of this repo.
"""
import torch
import torch.nn as nn

from .config import LX, LY, ExperimentConfig


class FieldMLP(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        d_in = 4 if cfg.use_scale_features else 2
        layers, d = [], d_in
        for h in cfg.hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        # output affine head initialised to obs stats is handled in train.py
        self.out_mu = nn.Parameter(torch.zeros(1))
        self.out_sigma = nn.Parameter(torch.ones(1))

    def features(self, xy: torch.Tensor) -> torch.Tensor:
        f = [xy[:, 0:1] / LX, xy[:, 1:2] / LY]
        if self.cfg.use_scale_features:
            f += [xy[:, 0:1] / self.cfg.a_h, xy[:, 1:2] / self.cfg.a_v]
        return torch.cat(f, dim=1)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return (self.out_mu + self.out_sigma * self.net(self.features(xy))).squeeze(-1)
