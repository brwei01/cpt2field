"""Experimental ACF estimators (numpy), reverse-engineered and verified
to reproduce the target CSVs *exactly* (abs err < 1e-10).

Estimator convention (confirmed against acf_*_targets.csv):
  - mu  = global mean of ALL 123 CPT observations
  - var = global BIASED variance (ddof=0) of ALL 123 CPT observations
  - vertical pairs: within the same CPT column, |dy| = lag
  - horizontal pairs: same y, different columns, |dx| = lag
  - rho(lag) = sum[(z_a - mu)(z_b - mu)] / (n_pairs * var)
  - pair-level target: (z_i - mu)(z_j - mu) / var
"""
import numpy as np
import pandas as pd


def global_stats(z: np.ndarray) -> tuple[float, float]:
    return float(z.mean()), float(z.var())  # biased variance


def vertical_acf(obs: pd.DataFrame, lags: np.ndarray) -> pd.DataFrame:
    """obs: cpt_observations.csv frame. Returns acf + pair count per lag."""
    mu, var = global_stats(obs["z_obs_mpa"].values)
    step = np.diff(np.sort(obs["y_m"].unique())).min()
    rows = []
    for lag in lags:
        k = int(round(lag / step))
        num, cnt = 0.0, 0
        for c in obs["cpt_column"].unique():
            z = obs.loc[obs.cpt_column == c].sort_values("y_m")["z_obs_mpa"].values
            if k < len(z):
                num += ((z[:-k] - mu) * (z[k:] - mu)).sum()
                cnt += len(z) - k
        rows.append((lag, num / (cnt * var), cnt))
    return pd.DataFrame(rows, columns=["lag_h_m", "acf", "count_pairs"])


def horizontal_acf(obs: pd.DataFrame, lags: np.ndarray) -> pd.DataFrame:
    mu, var = global_stats(obs["z_obs_mpa"].values)
    xs = np.sort(obs["x_m"].unique())
    rows = []
    for lag in lags:
        num, cnt = 0.0, 0
        for i, xi in enumerate(xs):
            for xj in xs[i + 1:]:
                if abs((xj - xi) - lag) < 1e-9:
                    a = obs.loc[obs.x_m == xi].sort_values("y_m")["z_obs_mpa"].values
                    b = obs.loc[obs.x_m == xj].sort_values("y_m")["z_obs_mpa"].values
                    num += ((a - mu) * (b - mu)).sum()
                    cnt += len(a)
        rows.append((lag, num / (cnt * var), cnt))
    return pd.DataFrame(rows, columns=["lag_h_m", "acf", "count_pairs"])


def pair_products(obs: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """Standardized products for given obs_i/obs_j index pairs."""
    mu, var = global_stats(obs["z_obs_mpa"].values)
    zi = obs.set_index("obs_id").loc[pairs["obs_i"], "z_obs_mpa"].values
    zj = obs.set_index("obs_id").loc[pairs["obs_j"], "z_obs_mpa"].values
    return (zi - mu) * (zj - mu) / var
