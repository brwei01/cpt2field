"""Visualization suite: truth + CPT layout, reconstruction grids,
error maps, and ACF curves (reconstruction vs targets vs true field).
Requires fields saved by scripts/_run_batch.py in outputs/field_*.npy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cpt2field.config import DATA_DIR
from cpt2field import acf

OUT = Path(__file__).resolve().parent.parent / "outputs"
field = pd.read_csv(DATA_DIR / "field_data.csv").sort_values("point_id")
obs = pd.read_csv(DATA_DIR / "cpt_observations.csv")
vt = pd.read_csv(DATA_DIR / "acf_vertical_targets.csv")
ht = pd.read_csv(DATA_DIR / "acf_horizontal_targets.csv")

NX = field.grid_i_x.max() + 1
NY = field.grid_j_y.max() + 1
Z_TRUE = field.z_true_mpa.values.reshape(NY, NX)  # point_id row-major in y? verify below
# point_id ordering: id = j*NX + i (x fastest) per field_data construction
Z_TRUE = field.z_true_mpa.values.reshape(NY, NX)
EXT = [0, 200, 0, 200]
SCALES = [10, 50, 100, 150]
METHODS = ["copy2", "copy4"]

vmin, vmax = Z_TRUE.min(), Z_TRUE.max()


def grid(z_flat):
    return z_flat.reshape(NY, NX)


def load_field(method, ah):
    return np.load(OUT / f"field_{method}_ah{ah}.npy")


# ---- Fig 1: truth + CPT layout ----
fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(Z_TRUE, origin="lower", extent=EXT, cmap="viridis", vmin=vmin, vmax=vmax)
ax.scatter(obs.x_m, obs.y_m, s=6, c="red", marker="s", label="CPT obs (123)")
ax.set(xlabel="x (m)", ylabel="y (m)", title="True field + CPT columns")
ax.legend(loc="upper right", fontsize=8)
fig.colorbar(im, ax=ax, label="qt (MPa)")
fig.tight_layout()
fig.savefig(OUT / "fig1_truth_layout.png", dpi=150)

# ---- Fig 2: reconstructions 2x4 ----
fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), sharex=True, sharey=True)
for r, method in enumerate(METHODS):
    for c, ah in enumerate(SCALES):
        z = grid(load_field(method, ah))
        ax = axes[r, c]
        im = ax.imshow(z, origin="lower", extent=EXT, cmap="viridis", vmin=vmin, vmax=vmax)
        rmse = float(np.sqrt(np.mean((z - Z_TRUE) ** 2)))
        ax.set_title(f"{method}  a_h={ah}  rmse={rmse:.3f}", fontsize=10)
        if c == 0:
            ax.set_ylabel("y (m)")
fig.colorbar(im, ax=axes, shrink=0.8, label="qt (MPa)")
fig.suptitle("Reconstructed fields (rows: method, cols: assumed a_h)")
fig.savefig(OUT / "fig2_reconstructions.png", dpi=150)

# ---- Fig 3: error maps 2x4 ----
emax = max(np.abs(grid(load_field(m, a)) - Z_TRUE).max() for m in METHODS for a in SCALES)
fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), sharex=True, sharey=True)
for r, method in enumerate(METHODS):
    for c, ah in enumerate(SCALES):
        err = grid(load_field(method, ah)) - Z_TRUE
        ax = axes[r, c]
        im = ax.imshow(err, origin="lower", extent=EXT, cmap="RdBu_r", vmin=-emax, vmax=emax)
        ax.vlines([25, 100, 175], 0, 200, colors="k", lw=0.4, alpha=0.5)
        ax.set_title(f"{method}  a_h={ah}", fontsize=10)
        if c == 0:
            ax.set_ylabel("y (m)")
fig.colorbar(im, ax=axes, shrink=0.8, label="error (MPa)")
fig.suptitle("Error maps (pred - true); black lines = CPT columns")
fig.savefig(OUT / "fig3_error_maps.png", dpi=150)

# ---- Fig 4: ACF curves ----
def field_frame(z_flat):
    f = field.copy()
    f["z_obs_mpa"] = z_flat  # reuse estimator: treat grid columns as CPT columns
    return f


def grid_vertical_acf(z_flat, lags):
    """Vertical ACF of a full grid field with the verified estimator."""
    df = pd.DataFrame({"cpt_column": field.grid_i_x.values,
                       "y_m": field.y_m.values, "z_obs_mpa": z_flat})
    return acf.vertical_acf(df, lags)


lags = vt.lag_h_m.values
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, method in zip(axes, METHODS):
    ax.plot(lags, vt.acf_cpt_observed, "k-o", ms=3, lw=1.5, label="target (CPT experimental)")
    ax.plot(lags, vt.acf_true_full_field, "k--", lw=1, label="true full field")
    for ah, col in zip(SCALES, plt.cm.plasma(np.linspace(0.15, 0.85, 4))):
        a = grid_vertical_acf(load_field(method, ah), lags)
        ax.plot(lags, a.acf, color=col, lw=1.2, label=f"recon a_h={ah}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set(xlabel="vertical lag (m)", ylabel="ACF", title=f"{method}: vertical ACF")
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig4_acf_vertical.png", dpi=150)

# horizontal ACF bar comparison
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
width = 0.18
for ax, method in zip(axes, METHODS):
    xpos = np.arange(2)
    ax.bar(xpos - 1.5 * width, ht.acf_cpt_target_used, width, color="k", alpha=0.75,
           label="target (weights 1.0/0.1)")
    ax.bar(xpos - 0.5 * width, ht.acf_true_full_field, width, color="gray", label="true field")
    for k, (ah, col) in enumerate(zip([10, 150], ["#7f77dd", "#d85a30"])):
        z = load_field(method, ah)
        df = pd.DataFrame({"x_m": field.x_m.values, "y_m": field.y_m.values,
                           "z_obs_mpa": z})
        h = acf.horizontal_acf(df, ht.lag_h_m.values)
        ax.bar(xpos + (0.5 + k) * width, h.acf, width, color=col, label=f"recon a_h={ah}")
    ax.set_xticks(xpos, ["75 m", "150 m"])
    ax.axhline(0, color="gray", lw=0.5)
    ax.set(title=f"{method}: horizontal ACF", xlabel="lag")
axes[0].set_ylabel("ACF")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig5_acf_horizontal.png", dpi=150)
print("saved:", sorted(p.name for p in OUT.glob("fig*.png")))
