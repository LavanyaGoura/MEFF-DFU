"""MEFF - stratified random 60:20:20 split at image level (no near-duplicate grouping).
train_test_split(test 0.20, stratify=label, random_state=42), then validation = 0.25 of the remainder.
Rows ordered by (label, filename) before splitting, so the split is reproducible.
group_id = one id per image (so evaluation resamples single images).
A near-duplicate audit (pHash groups, T = 8 bits, from groups.csv) is REPORTED, not used for splitting.
Usage: python make_split.py --groups groups.csv --out_dir split
"""
import argparse, hashlib, json, os, platform
import numpy as np, pandas as pd, sklearn
from sklearn.model_selection import train_test_split
CLASSES = ["NONE", "INFECTION", "ISCHAEMIA", "BOTH"]
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
ap = argparse.ArgumentParser(); ap.add_argument("--groups", required=True); ap.add_argument("--out_dir", required=True); a = ap.parse_args()
g = pd.read_csv(a.groups).sort_values(["label", "filename"]).reset_index(drop=True)
assert len(g) == 5955 and g.md5.nunique() == 5955
trval, te = train_test_split(g, test_size=0.20, stratify=g.label, random_state=42)
tr, va = train_test_split(trval, test_size=0.25, stratify=trval.label, random_state=42)
img_id = {r: i for i, r in enumerate(g.relpath)}
sets = {"train": tr, "validation": va, "test": te}
cols = ["relpath", "filename", "class_name", "label", "md5", "group_id"]
for n, d in sets.items():
    d = d.copy(); d["phash_group"] = d["group_id"]; d["group_id"] = d.relpath.map(img_id)
    d.sort_values(["label", "filename"])[cols].to_csv(os.path.join(a.out_dir, f"{n}.csv"), index=False)
    sets[n] = d
for k in ["relpath", "md5", "group_id"]:
    s = {n: set(d[k]) for n, d in sets.items()}
    assert not (s["train"] & s["validation"]) and not (s["train"] & s["test"]) and not (s["validation"] & s["test"]), k
tab = pd.DataFrame({n: d.class_name.value_counts().reindex(CLASSES) for n, d in sets.items()})
tab["total"] = tab.sum(axis=1); tab.loc["TOTAL"] = tab.sum()
pct = (tab[["train", "validation", "test"]].div(tab["total"], axis=0) * 100).round(2)
# near-duplicate audit (information only)
trg = set(sets["train"].phash_group)
audit = {}
for n in ["validation", "test"]:
    d = sets[n]; hit = d[d.phash_group.isin(trg)]
    audit[n] = dict(images_with_near_duplicate_in_train=int(len(hit)), per_class=hit.class_name.value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict())
with open(os.path.join(a.out_dir, "split_check.txt"), "w") as f:
    f.write("Stratified random 60:20:20 split, image level, seed 42 (no near-duplicate grouping)\n\nCounts\n" + tab.to_string() +
            "\n\nPercent of class\n" + pct.to_string() + "\n\nShared relpath / md5 between subsets: 0\n\n"
            "Near-duplicate audit (pHash, T = 8 bits; reported only, not used for splitting)\n" + json.dumps(audit, indent=2) + "\n")
man = dict(method="image-level stratified random split: train_test_split test 0.20 then 0.25 of remainder, random_state 42",
           group_id="one id per image (no grouping)", versions=dict(python=platform.python_version(), sklearn=sklearn.__version__,
           numpy=np.__version__, pandas=pd.__version__), near_duplicate_audit=audit,
           files_md5={f: md5(os.path.join(a.out_dir, f)) for f in ["train.csv", "validation.csv", "test.csv"]})
json.dump(man, open(os.path.join(a.out_dir, "SPLIT_MANIFEST.json"), "w"), indent=2)
print(open(os.path.join(a.out_dir, "split_check.txt")).read())
