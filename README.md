# cpt2field

Reconstruct a 200 m x 200 m soil property field (41x41 grid) from 3 CPT
profiles (123 observations), using an MLP constrained by experimental
autocorrelation (ACF) losses — replacing the conditional-random-field
generator of CSM-style methods with a coordinate network + ACF-based losses.

## Layout
```
cpt2field/
  config.py     paths, ExperimentConfig, loss lambdas (mirrors loss_config.csv)
  acf.py        numpy experimental-ACF estimators, verified EXACT vs targets
  data.py       loads CSVs, pre-builds all pair indices as tensors
  model.py      FieldMLP; scale-normalized input features (x/a_h, y/a_v)
  losses.py     4 differentiable losses: data / vertical ACF / weighted
                horizontal ACF / copy4 pair-product
  train.py      Adam + cosine schedule, with data-only warmup phase
  evaluate.py   rmse / mae / r2 (full field), obs_rmse (123 CPT points)
scripts/
  01_verify_targets.py   RA tasks 1-3: exact re-derivation of all targets
  02_train.py            single run: python scripts/02_train.py copy4 150 4000
  _run_batch.py          one method x 4 scales -> outputs/our_results_*.csv
outputs/                 our_results.csv, comparison.csv
```

## Verified estimator (tasks 1-3, max err < 1e-12)
mu and biased variance over ALL 123 obs; vertical pairs within columns,
horizontal pairs across columns at equal y; rho = mean centred product / var;
pair target = (z_i-mu)(z_j-mu)/var; reliability weights 1.0 (75 m) / 0.1 (150 m).

## Documented assumptions (original notebooks not shared)
1. Differentiable ACF losses are evaluated on the FULL predicted grid
   (acf_on="grid"), pushing the reconstruction's correlation structure
   toward the CPT experimental targets.
2. BOTH methods use scale-normalized input features; copy4 additionally
   uses the pair-product loss. (If only copy4 had scale features, copy2
   would be scale-invariant — contradicting results_summary.)
3. Warmup: normalized ACF losses are amplitude-invariant, so the first
   800 epochs use data loss only to anchor the field amplitude; joint
   training without warmup collapses to a near-flat field (documented bug).

## Reproduction status vs results_summary_from_html.csv
- copy2 @ a_h=10/50: closely reproduced (rmse 0.66/0.45 vs ref 0.60/0.44).
- copy4 @ a_h=100/150: closely reproduced (0.32/0.24 vs ref 0.35/0.41).
- Open gaps: ref copy2 degrades at large a_h (rmse > 1) while ours improves;
  ref copy4 stays strong at wrongly-small scales (0.27 @ a_h=10) which our
  simple (x/a_h, y/a_v) features do not achieve. The exact "scale_features"
  construction of the original copy4 remains the main unknown.
