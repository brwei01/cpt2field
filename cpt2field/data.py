"""Load the data package into one bundle of numpy arrays / torch tensors,
with all pair indices pre-built so losses are pure tensor ops."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch

from .config import DATA_DIR, GRID_STEP


@dataclass
class Bundle:
    # frames (kept for reference / evaluation)
    field: pd.DataFrame
    obs: pd.DataFrame
    # grid tensors
    xy_grid: torch.Tensor          # (1681, 2) coordinates in metres
    z_true: torch.Tensor           # (1681,) evaluation ONLY, never in losses
    nx: int
    ny: int
    # observation tensors
    obs_grid_idx: torch.Tensor     # (123,) index of each obs point in the grid
    z_obs: torch.Tensor            # (123,)
    # vertical ACF targets: per-lag index pairs on the GRID (column-wise)
    v_lags: np.ndarray
    v_target: torch.Tensor         # (n_lags,)
    v_pairs_grid: list             # list of (idx_a, idx_b) LongTensors, full grid
    v_pairs_obs: list              # same but restricted to CPT columns
    # horizontal ACF targets
    h_lags: np.ndarray
    h_target: torch.Tensor         # (2,)
    h_weight: torch.Tensor         # (2,) reliability weights
    h_pairs_grid: list
    h_pairs_obs: list
    # copy4 pair-product targets (indices into the grid)
    pp_i: torch.Tensor
    pp_j: torch.Tensor
    pp_target: torch.Tensor
    pp_weight: torch.Tensor


def _column_pairs(ids: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """ids: grid indices of one vertical column ordered by y; lag = k steps."""
    return ids[:-k], ids[k:]


def load(data_dir=DATA_DIR, device="cpu") -> Bundle:
    field = pd.read_csv(data_dir / "field_data.csv")
    obs = pd.read_csv(data_dir / "cpt_observations.csv")
    vt = pd.read_csv(data_dir / "acf_vertical_targets.csv")
    ht = pd.read_csv(data_dir / "acf_horizontal_targets.csv")
    pp = pd.read_csv(data_dir / "horizontal_pair_product_targets.csv")

    nx = field.grid_i_x.max() + 1
    ny = field.grid_j_y.max() + 1
    field = field.sort_values("point_id").reset_index(drop=True)

    t = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt, device=device)
    li = lambda a: torch.tensor(np.asarray(a), dtype=torch.long, device=device)

    # ---- vertical pairs ----
    v_lags = vt.lag_h_m.values
    grid_cols = [field.loc[field.grid_i_x == i].sort_values("y_m")["point_id"].values
                 for i in range(nx)]
    obs_cols = [obs.loc[obs.cpt_column == c].sort_values("y_m")["point_id"].values
                for c in sorted(obs.cpt_column.unique())]
    v_pairs_grid, v_pairs_obs = [], []
    for lag in v_lags:
        k = int(round(lag / GRID_STEP))
        pg = [_column_pairs(c, k) for c in grid_cols if k < len(c)]
        po = [_column_pairs(c, k) for c in obs_cols if k < len(c)]
        v_pairs_grid.append((li(np.concatenate([p[0] for p in pg])),
                             li(np.concatenate([p[1] for p in pg]))))
        v_pairs_obs.append((li(np.concatenate([p[0] for p in po])),
                            li(np.concatenate([p[1] for p in po]))))

    # ---- horizontal pairs ----
    h_lags = ht.lag_h_m.values
    h_pairs_grid, h_pairs_obs = [], []
    xs_grid = np.sort(field.x_m.unique())
    xs_obs = np.sort(obs.x_m.unique())
    for lag in h_lags:
        ga, gb, oa, ob = [], [], [], []
        for xi in xs_grid:
            xj = xi + lag
            if xj in xs_grid:
                a = field.loc[field.x_m == xi].sort_values("y_m")["point_id"].values
                b = field.loc[field.x_m == xj].sort_values("y_m")["point_id"].values
                ga.append(a); gb.append(b)
        for xi in xs_obs:
            xj = xi + lag
            if xj in xs_obs:
                a = obs.loc[obs.x_m == xi].sort_values("y_m")["point_id"].values
                b = obs.loc[obs.x_m == xj].sort_values("y_m")["point_id"].values
                oa.append(a); ob.append(b)
        h_pairs_grid.append((li(np.concatenate(ga)), li(np.concatenate(gb))))
        h_pairs_obs.append((li(np.concatenate(oa)), li(np.concatenate(ob))))

    return Bundle(
        field=field, obs=obs,
        xy_grid=t(field[["x_m", "y_m"]].values),
        z_true=t(field["z_true_mpa"].values),
        nx=int(nx), ny=int(ny),
        obs_grid_idx=li(obs["point_id"].values),
        z_obs=t(obs["z_obs_mpa"].values),
        v_lags=v_lags,
        v_target=t(vt["acf_cpt_observed"].values),
        v_pairs_grid=v_pairs_grid, v_pairs_obs=v_pairs_obs,
        h_lags=h_lags,
        h_target=t(ht["acf_cpt_target_used"].values),
        h_weight=t(ht["reliability_weight"].values),
        h_pairs_grid=h_pairs_grid, h_pairs_obs=h_pairs_obs,
        pp_i=li(pp["point_id_i"].values),
        pp_j=li(pp["point_id_j"].values),
        pp_target=t(pp["target_pair_corr"].values),
        pp_weight=t(pp["pair_weight"].values),
    )
