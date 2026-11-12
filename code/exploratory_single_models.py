"""Exploratory (not prespecified) comparison: each MEFF expert vs the Dense-ShuffleGCANet model with the same seed.
Test macro-F1 on the 1,191 test images; 95% CI of the difference from 2,000 paired image-level bootstrap resamples.
Usage: python exploratory_single_models.py --runs_dir runs --out_dir results"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--runs_dir", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--n_boot", type=int, default=2000)
a = ap.parse_args()
rng = np.random.default_rng(42)                       # one generator, seeds processed in order 42, 43, 44
rows = []
for s in (42, 43, 44):
    e = pd.read_csv(os.path.join(a.runs_dir, f"meff_s{s}", "test_predictions.csv"))
    b = pd.read_csv(os.path.join(a.runs_dir, f"dsg_s{s}", "test_predictions.csv"))
    assert (e.relpath.values == b.relpath.values).all()
    y, pe, pb = e["true"].values, e["pred"].values, b["pred"].values
    f = lambda idx, p: f1_score(y[idx], p[idx], average="macro", labels=[0, 1, 2, 3], zero_division=0)
    full = np.arange(len(y)); fe, fb = f(full, pe), f(full, pb)
    d = []
    for _ in range(a.n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        d.append(f(idx, pe) - f(idx, pb))
    lo, hi = np.percentile(d, [2.5, 97.5])
    rows.append(dict(seed=s, expert_macro_f1=fe, base_macro_f1=fb, delta=fe - fb, ci95_low=lo, ci95_high=hi))
df = pd.DataFrame(rows); os.makedirs(a.out_dir, exist_ok=True)
df.to_csv(os.path.join(a.out_dir, "exploratory_single_meff_vs_dsg.csv"), index=False)
print(df.to_string(index=False))
