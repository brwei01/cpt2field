"""RA tasks 4-5: run 2 methods x 4 assumed scales, write our metrics
side-by-side with results_summary_from_html.csv."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from cpt2field.config import DATA_DIR, ExperimentConfig
from cpt2field.data import load
from cpt2field.train import train
from cpt2field.evaluate import evaluate

b = load()
scales = pd.read_csv(DATA_DIR / "scale_config.csv")
ref = pd.read_csv(DATA_DIR / "results_summary_from_html.csv")
rows = []
for method in ["copy2", "copy4"]:
    for _, sc in scales.iterrows():
        cfg = ExperimentConfig(method=method, a_h=sc.a_h_m)
        t0 = time.time()
        model, _ = train(b, cfg)
        m = evaluate(model, b)
        rows.append({"method": method, "a_h_m": sc.a_h_m, "a_v_m": round(sc.a_v_m, 1),
                     **{k: round(v, 4) for k, v in m.items()},
                     "sec": round(time.time() - t0, 1)})
        print(rows[-1])
out = pd.DataFrame(rows)
out.to_csv("outputs/our_results.csv", index=False)
ref_m = ref.assign(method=ref.method.str.split("_").str[0])
cmp = out.merge(ref_m, on=["method", "a_h_m"], suffixes=("_ours", "_ref"))
cols = ["method", "a_h_m", "rmse_ours", "rmse_ref", "r2_ours", "r2_ref",
        "obs_rmse_ours", "obs_rmse_ref"]
cmp[cols].to_csv("outputs/comparison.csv", index=False)
print("\n", cmp[cols].to_string(index=False))
