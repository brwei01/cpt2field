"""Train a single experiment. Usage:
   python scripts/02_train.py [copy2|copy4] [a_h] [epochs]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpt2field.config import ExperimentConfig
from cpt2field.data import load
from cpt2field.train import train
from cpt2field.evaluate import evaluate

method = sys.argv[1] if len(sys.argv) > 1 else "copy4"
a_h = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 4000

cfg = ExperimentConfig(method=method, a_h=a_h, epochs=epochs)
b = load()
model, _ = train(b, cfg, verbose_every=max(1, epochs // 8))
print(f"[{cfg.name}] metrics:", evaluate(model, b))
