"""RA tasks 1-3: recompute all experimental ACF targets from raw CPT
observations and check exact agreement with the target CSVs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from cpt2field.config import DATA_DIR
from cpt2field import acf

obs = pd.read_csv(DATA_DIR / "cpt_observations.csv")
vt = pd.read_csv(DATA_DIR / "acf_vertical_targets.csv")
ht = pd.read_csv(DATA_DIR / "acf_horizontal_targets.csv")
pp = pd.read_csv(DATA_DIR / "horizontal_pair_product_targets.csv")

v = acf.vertical_acf(obs, vt.lag_h_m.values)
ok_v = np.allclose(v.acf, vt.acf_cpt_observed) and (v.count_pairs.values == vt.count_cpt_pairs.values).all()
print(f"Task 1  vertical ACF   (40 lags): {'PASS' if ok_v else 'FAIL'}  max|err|={np.abs(v.acf - vt.acf_cpt_observed).max():.2e}")

h = acf.horizontal_acf(obs, ht.lag_h_m.values)
ok_h = np.allclose(h.acf, ht.acf_cpt_raw) and (h.count_pairs.values == ht.count_cpt_pairs.values).all()
print(f"Task 2  horizontal ACF  (2 lags): {'PASS' if ok_h else 'FAIL'}  max|err|={np.abs(h.acf - ht.acf_cpt_raw).max():.2e}")
print(f"        weighted-loss inputs: lags={list(ht.lag_h_m)}, weights={list(ht.reliability_weight)}, |acf|>1 flag={list(ht.acf_abs_gt_1)}")

prod = acf.pair_products(obs, pp)
ok_p = np.allclose(prod, pp.target_pair_corr)
print(f"Task 3  pair products (123 pairs): {'PASS' if ok_p else 'FAIL'}  max|err|={np.abs(prod - pp.target_pair_corr).max():.2e}")

sys.exit(0 if (ok_v and ok_h and ok_p) else 1)
