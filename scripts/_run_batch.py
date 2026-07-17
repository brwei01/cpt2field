import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from cpt2field.config import DATA_DIR, ExperimentConfig
from cpt2field.data import load
from cpt2field.train import train
from cpt2field.evaluate import evaluate

method = sys.argv[1]
b = load()
scales = pd.read_csv(DATA_DIR / "scale_config.csv")
rows = []
for _, sc in scales.iterrows():
    cfg = ExperimentConfig(method=method, a_h=sc.a_h_m)
    t0 = time.time()
    model, _ = train(b, cfg)
    m = evaluate(model, b)
    rows.append({"method": method, "a_h_m": sc.a_h_m, "a_v_m": round(sc.a_v_m, 1),
                 **{k: round(v, 4) for k, v in m.items()},
                 "sec": round(time.time() - t0, 1)})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"outputs/our_results_{method}.csv", index=False)
