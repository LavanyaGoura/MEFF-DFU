"""Post-hoc sensitivity analysis (not prespecified): repeat the main comparisons after removing the test images
that have a perceptual-hash near-duplicate in the training set (split/groups.csv; 39 of 1,191 test images).
Same statistics as the paper: paired image-level bootstrap (2,000 resamples, numpy default_rng(42)), exact McNemar, Holm.
Usage: python sensitivity_near_duplicates.py --runs_dir runs --split_dir split --out_dir results"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

ap = argparse.ArgumentParser()
for k in ("--runs_dir", "--split_dir", "--out_dir"): ap.add_argument(k, required=True)
ap.add_argument("--n_boot", type=int, default=2000)
a = ap.parse_args()
L = [0, 1, 2, 3]
g = pd.read_csv(os.path.join(a.split_dir, "groups.csv")); gid = dict(zip(g.relpath, g.group_id))
tr = pd.read_csv(os.path.join(a.split_dir, "train.csv")); train_groups = set(tr.relpath.map(gid))

def load(v, s):
    return pd.read_csv(os.path.join(a.runs_dir, f"{v}_s{s}", "test_predictions.csv"))
def hard_vote(v):
    ds = [load(v, s) for s in (42, 43, 44)]
    P = np.stack([d.pred.values for d in ds], 1)
    prob = np.mean([d[[f"prob_{c}" for c in ("NONE", "INFECTION", "ISCHAEMIA", "BOTH")]].values for d in ds], 0)
    out = []
    for i in range(len(P)):                          # same rule as meff_main.py: majority; on a tie, highest mean softmax among the tied classes
        cnt = np.bincount(P[i], minlength=4); top = np.flatnonzero(cnt == cnt.max())
        out.append(int(top[np.argmax(prob[i, top])]))
    return ds[0], np.array(out)

ref, _ = hard_vote("meff")
y = ref["true"].values
flag = ref.relpath.map(gid).isin(train_groups).values
pred = {f"{v.upper()}_hard": hard_vote(v)[1] for v in ("meff", "dsg", "dsg_se", "dsg_hc")}
pred["MEFF_s44"] = load("meff", 44).pred.values
for sd in (42, 43, 44): pred[f"DSG_s{sd}"] = load("dsg", sd).pred.values
for v in ("meff", "dsg", "dsg_se", "dsg_hc"):
    assert (load(v, 42).relpath.values == ref.relpath.values).all()
mf1 = lambda yy, pp: f1_score(yy, pp, labels=L, average="macro", zero_division=0)

rows = []
for name, sub in (("all 1,191 test images", np.ones(len(y), bool)), ("without the 39 near-duplicate images", ~flag), ("only the 39 near-duplicate images", flag)):
    for m, p in pred.items():
        rows.append(dict(subset=name, model=m, n=int(sub.sum()), accuracy=(p[sub] == y[sub]).mean(), macro_f1=mf1(y[sub], p[sub])))
perf = pd.DataFrame(rows)

keep = np.where(~flag)[0]; yk = y[keep]
rng = np.random.default_rng(42)
samples = [rng.choice(len(keep), len(keep), replace=True) for _ in range(a.n_boot)]
comps = [("primary", "MEFF_hard", "DSG_hard")] + [(f"MEFF vs single DSG seed {sd}", "MEFF_hard", f"DSG_s{sd}") for sd in (42, 43, 44)] + [("ensemble vs best expert (seed 44, validation-selected)", "MEFF_hard", "MEFF_s44"),
         ("ablation A1: SE without handcrafted", "DSG_SE_hard", "DSG_hard"), ("ablation A2: handcrafted without SE", "DSG_HC_hard", "DSG_hard"),
         ("ablation A3: handcrafted with SE", "MEFF_hard", "DSG_SE_hard"), ("ablation A4: SE with handcrafted", "MEFF_hard", "DSG_HC_hard")]
out = []
for fam, A, B in comps:
    pa, pb = pred[A][keep], pred[B][keep]
    d = [mf1(yk[i], pa[i]) - mf1(yk[i], pb[i]) for i in samples]
    da = [(pa[i] == yk[i]).mean() - (pb[i] == yk[i]).mean() for i in samples]
    n10 = int(((pa == yk) & (pb != yk)).sum()); n01 = int(((pa != yk) & (pb == yk)).sum())
    out.append(dict(comparison=fam, A=A, B=B, n=len(keep), delta_macro_f1=mf1(yk, pa) - mf1(yk, pb),
                    ci95_low=np.percentile(d, 2.5), ci95_high=np.percentile(d, 97.5),
                    delta_accuracy=(pa == yk).mean() - (pb == yk).mean(), acc_ci95_low=np.percentile(da, 2.5), acc_ci95_high=np.percentile(da, 97.5),
                    n10=n10, n01=n01, mcnemar_p=binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0))
comp = pd.DataFrame(out)
# Holm within the two families used in the paper: the five main comparisons and the ablation (A1-A4)
for fam in (comp.comparison.str.startswith("ablation"), ~comp.comparison.str.startswith("ablation")):
    p = comp.loc[fam, "mcnemar_p"].values; o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * p[i])); adj[i] = run
    comp.loc[fam, "mcnemar_p_holm"] = adj
os.makedirs(a.out_dir, exist_ok=True)
perf.to_csv(os.path.join(a.out_dir, "sensitivity_near_duplicates_performance.csv"), index=False)
comp.to_csv(os.path.join(a.out_dir, "sensitivity_near_duplicates_comparisons.csv"), index=False)
pd.set_option("display.width", 220)
print(perf.assign(accuracy=lambda d: (d.accuracy * 100).round(2), macro_f1=lambda d: (d.macro_f1 * 100).round(2)).to_string(index=False))
c = comp.copy()
for k in ("delta_macro_f1", "ci95_low", "ci95_high", "delta_accuracy", "acc_ci95_low", "acc_ci95_high"): c[k] = (c[k] * 100).round(2)
print(c[["comparison", "A", "B", "n", "delta_macro_f1", "ci95_low", "ci95_high", "delta_accuracy", "acc_ci95_low", "acc_ci95_high", "n10", "n01", "mcnemar_p", "mcnemar_p_holm"]].round(4).to_string(index=False))
