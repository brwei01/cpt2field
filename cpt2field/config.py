"""Central configuration: paths, experiment settings, loss weights."""
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

GRID_STEP = 5.0          # m, grid spacing
LX = LY = 200.0          # domain size, m
V_OVER_H = 80.0 / 150.0  # fixed anisotropy ratio a_v / a_h


@dataclass
class LossWeights:
    """Lambda values, mirroring loss_config.csv."""
    data: float = 1.0
    acf_vertical: float = 5.0
    acf_horizontal: float = 1.0
    pair_product: float = 0.5   # copy4 only


@dataclass
class ExperimentConfig:
    method: str = "copy4"        # "copy2" | "copy4"
    a_h: float = 150.0           # assumed horizontal fluctuation scale, m
    hidden: tuple = (64, 64, 64)
    lr: float = 3e-3
    epochs: int = 4000
    warmup_epochs: int = 800   # data-loss-only phase to anchor amplitude
    seed: int = 0
    acf_on: str = "grid"         # where differentiable ACF is evaluated: "grid" | "obs"
    weights: LossWeights = field(default_factory=LossWeights)

    @property
    def a_v(self) -> float:
        return self.a_h * V_OVER_H

    @property
    def use_pair_loss(self) -> bool:
        return self.method == "copy4"

    @property
    def use_scale_features(self) -> bool:
        # Both methods feed scale-normalized coords; otherwise copy2 would be
        # insensitive to the assumed scale, contradicting results_summary.
        return True

    @property
    def name(self) -> str:
        return f"{self.method}_ah{self.a_h:g}"
