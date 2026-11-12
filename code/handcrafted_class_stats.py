"""Class-mean 42-D handcrafted descriptors over the test images.
Usage: python handcrafted_class_stats.py --data_dir <DFUC2021_5955> --split_dir split --code_dir code --out_dir results"""
import argparse, sys, os, numpy as np, pandas as pd
from PIL import Image
ap = argparse.ArgumentParser()
for k in ("--data_dir", "--split_dir", "--code_dir", "--out_dir"): ap.add_argument(k, required=True)
a = ap.parse_args()
sys.path.insert(0, a.code_dir)
import meff_main as mv
DATA = a.data_dir
te = pd.read_csv(os.path.join(a.split_dir, "test.csv"))
names = [f"LBP{i}" for i in range(10)] + [f"{c}{i}" for c in "RGB" for i in range(8)] + [f"EO{i}" for i in range(8)]
rows = []; extra = []
for cls in ["NONE", "INFECTION", "ISCHAEMIA", "BOTH"]:
    d = te[te.class_name == cls]
    ims = [np.array(Image.open(os.path.join(DATA, r)).convert("RGB")) for r in d.relpath]
    F = np.stack([mv.handcrafted_42(im) for im in ims])
    mu, sd = F.mean(0), F.std(0)
    for n, m, s in zip(names, mu, sd): rows.append(dict(class_name=cls, n=len(d), feature=n, mean=m, sd=s))
    # summary quantities (per image, then class mean +- sd)
    centres = (np.arange(8) + 0.5) * 32
    mR = F[:, 10:18] @ centres; mG = F[:, 18:26] @ centres; mB = F[:, 26:34] @ centres
    redness = np.array([(im[..., 0].astype(float) - im[..., 1]).mean() for im in ims])
    meanRGB = np.array([im.reshape(-1, 3).mean(0) for im in ims])
    flat = F[:, 9]; unif = F[:, :9].sum(1)
    hv = F[:, 34] + F[:, 41] + F[:, 37] + F[:, 38]   # near-0/180 deg and near-90 deg bins
    extra.append(dict(class_name=cls, n=len(d),
        R_mean=meanRGB[:, 0].mean(), R_sd=meanRGB[:, 0].std(), G_mean=meanRGB[:, 1].mean(), G_sd=meanRGB[:, 1].std(),
        B_mean=meanRGB[:, 2].mean(), B_sd=meanRGB[:, 2].std(), RminusG=redness.mean(), RminusG_sd=redness.std(),
        lbp_nonuniform=flat.mean(), lbp_nonuniform_sd=flat.std(),
        lbp_flat0=F[:, 0].mean(), lbp_edge4=F[:, 4].mean(), lbp_spot8=F[:, 8].mean(),
        lbp8=F[:,8].mean(), lbp8_sd=F[:,8].std(), lbp4=F[:,4].mean(), lbp4_sd=F[:,4].std(),
        eo_max=F[:, 34:42].mean(0).max(), eo_min=F[:, 34:42].mean(0).min(),
        eo_axis=hv.mean(), eo_axis_sd=hv.std(),
        **{f"EO{i}": F[:, 34 + i].mean() for i in range(8)},
        **{f"R{i}": F[:, 10 + i].mean() for i in range(8)},
        **{f"LBP{i}": F[:, i].mean() for i in range(10)}))
out = a.out_dir; os.makedirs(out, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(out, "handcrafted_class_means.csv"), index=False)
pd.DataFrame(extra).to_csv(os.path.join(out, "handcrafted_class_summary.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
print(pd.DataFrame(extra).T.round(3))
