#!/usr/bin/env python3
"""
MEFF - every data/result figure of the manuscript, drawn at PRINT SIZE.

Each figure is drawn at the width it has on the page (full text width = 6.25 in), so the fonts printed in the
journal are the fonts set here: all text >= 8 pt (Springer: 8-12 pt), axis labels and panel titles 8.5-9 pt.
Insert the figures at 100 % (Word: width = the width printed by this script); do not scale them.

  python make_paper_figures.py --runs_dir runs --data_dir DFUC2021_deduplicated_5955 --split_dir split \
         --ckpt_dir checkpoints --meff_py code/meff_main.py --out_dir figs_print [--results_dir results] [--font_dir fonts]

  runs_dir   meff_s42, meff_s43, meff_s44 (history.csv, test_predictions.csv from the result zips)
  ckpt_dir   best_s42.pth, best_s43.pth, best_s44.pth (only for feature maps and Grad-CAM)
  results_dir  optional: output of `meff_main.py eval`; if given, the figures' numbers are checked against it
  font_dir   optional: folder with LiberationSans-Regular.ttf / -Bold.ttf if the system has no Liberation Sans

Outputs (each as PNG at --dpi, default 600, and as vector PDF with embedded fonts):
  fig_samples, fig_split_distribution, fig_augmentation_examples, fig_pred_examples, fig_curves, fig_learning_curves (2 x 2:
  training accuracy, validation accuracy, training loss, validation loss), fig_perclass,
  fig_prob_examples, fig_confusion, fig_roc_pr, fig_hc_<CLASS> x4, fig_featuremaps_overview,
  fig_gradcam_correct, fig_gradcam_errors; figure_sizes.csv (width/height in inches) and figure_notes.txt.

Image selection rules (the same as in notebooks/bootstrap_analysis.ipynb):
  samples        4 random TRAINING images per class, numpy seed 7 (relpaths sorted)
  augmentation   1 random training image per class, numpy + torch seed 7, 5 augmented copies
  pred examples  4 random TEST images per class, numpy seed 42
  prob, Grad-CAM correct, feature maps   correct test image with the median ensemble probability of its class
  Grad-CAM errors                         misclassified test image with the median probability of the wrong class
  handcrafted    test image whose 42-D descriptor is closest to the class mean
--only <name ...> redraws only the named figures (e.g. --only curves roc_pr).
"""
import argparse, importlib.util, os, sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from PIL import Image

# ---------------------------------------------------------------------------------------------------- style
W = 6.25                     # full text width of the manuscript (inches)
INK, INK2, GRID, MUTED = "#1f1f1e", "#5f5e58", "#e4e3de", "#c3c2b7"
EXP_COL = ["#2a78d6", "#eb6834", "#1baf7a"]     # Expert 1, 2, 3 (validated categorical slots)
EXP_MK = ["o", "s", "^"]
EXP_LS = ["-", "--", "-."]
MEFF_COL = "#0d366b"                              # the ensemble: darkest, thickest, always drawn last
CLS_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#c98500"]
CLS_LS = ["-", "--", "-.", ":"]
OK_COL, BAD_COL = "#008300", "#c62828"
BLUES = LinearSegmentedColormap.from_list("blue_ramp", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"],   # Arial-metric; DejaVu only for missing glyphs (check marks)
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "legend.handlelength": 2.2, "legend.borderaxespad": 0.3,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 1.4, "axes.titlelocation": "left", "axes.titlepad": 4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
})
NOTES, SIZES = [], []
A = None   # parsed arguments
mv = None  # meff_main module
CLASSES = ["NONE", "INFECTION", "ISCHAEMIA", "BOTH"]; NC = 4
PC = [f"prob_{c}" for c in CLASSES]
SEEDS = [42, 43, 44]
EXPN = {s: f"Expert {i + 1} (seed {s})" for i, s in enumerate(SEEDS)}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(A.out_dir, f"{name}.{ext}"), dpi=A.dpi)   # no bbox_inches: keep the designed size
    w, h = fig.get_size_inches(); SIZES.append((name, round(w, 3), round(h, 3)))
    plt.close(fig); print(f"wrote {name}  ({w:.2f} x {h:.2f} in)")


def img(rel, size=None):
    im = Image.open(os.path.join(A.data_dir, rel)).convert("RGB")
    return np.array(im.resize((size, size), Image.BILINEAR) if size else im)


def grid(nr, nc, width=W, left=0.25, right=0.02, top=0.3, bottom=0.05, ws=0.05, hs=0.2, aspect=1.0):
    """Figure of nr x nc image tiles laid out in inches (exact width, square tiles)."""
    tw = (width - left - right - (nc - 1) * ws) / nc; th = tw * aspect
    fh = top + nr * th + (nr - 1) * hs + bottom
    fig = plt.figure(figsize=(width, fh)); ax = np.empty((nr, nc), object)
    for i in range(nr):
        for j in range(nc):
            x = left + j * (tw + ws); y = fh - top - (i + 1) * th - i * hs
            a = fig.add_axes([x / width, y / fh, tw / width, th / fh]); a.set_xticks([]); a.set_yticks([]); a.grid(False)
            for s in a.spines.values():
                s.set_visible(True); s.set_color(MUTED); s.set_linewidth(0.5)
            ax[i, j] = a
    return fig, ax, tw, th, fh


def row_label(a, text, pad=0.06):
    a.text(-pad, 0.5, text, transform=a.transAxes, rotation=90, ha="right", va="center", fontsize=8.5)


def norm01(a, robust=False):
    if robust and min(a.shape) > 4:   # zero-padding makes the 1-cell border extreme: scale on the interior
        lo, hi = np.percentile(a[1:-1, 1:-1], [1, 99])
        return np.clip((a - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(a)
    a = a - a.min(); return a / a.max() if a.max() > 0 else a


def up(m, size=224):
    return np.array(Image.fromarray((255 * norm01(m)).astype(np.uint8)).resize((size, size), Image.BILINEAR)) / 255.0


def hbar(fig, x0, y0, w, h, cmap, label):
    """Small horizontal 0-1 colour bar (inches)."""
    fw, fh = fig.get_size_inches()
    ca = fig.add_axes([x0 / fw, y0 / fh, w / fw, h / fh])
    ca.imshow(np.linspace(0, 1, 256)[None], aspect="auto", cmap=cmap, extent=(0, 1, 0, 1))
    ca.set_yticks([]); ca.set_xticks([0, 0.5, 1]); ca.set_xticklabels(["0 (low)", "0.5", "1 (high)"], fontsize=8)
    ca.tick_params(length=2, pad=1.5); ca.grid(False)
    for s in ca.spines.values():
        s.set_visible(True); s.set_linewidth(0.5); s.set_color(INK2)
    ca.text(-0.03, 0.5, label, transform=ca.transAxes, ha="right", va="center", fontsize=8)


# ---------------------------------------------------------------------------------------------------- data figures
def fig_samples():
    tr = pd.read_csv(os.path.join(A.split_dir, "train.csv")).sort_values("relpath")
    rng = np.random.default_rng(7)
    fig, ax, tw, th, fh = grid(1, NC, left=0.02, top=0.04, bottom=0.24, ws=0.1)
    for k, c in enumerate(CLASSES):
        pick = rng.choice(tr[tr.class_name == c].relpath.values, 4, replace=False)
        NOTES.append(f"samples {c}: " + ", ".join(pick))
        tile = Image.new("RGB", (452, 452), "white")
        for j, r in enumerate(pick):
            tile.paste(Image.open(os.path.join(A.data_dir, r)).convert("RGB").resize((224, 224)), ((j % 2) * 228, (j // 2) * 228))
        a = ax[0, k]; a.imshow(tile); [s.set_visible(False) for s in a.spines.values()]
        a.text(0.5, -0.05, f"({'ABCD'[k]}) {c}", transform=a.transAxes, ha="center", va="top", fontsize=9)
    save(fig, "fig_samples")


def fig_split():
    cnt = {n: pd.read_csv(os.path.join(A.split_dir, f"{n}.csv")).class_name.value_counts().reindex(CLASSES)
           for n in ["train", "validation", "test"]}
    lab = {"train": "Training", "validation": "Validation", "test": "Test"}
    fig, ax = plt.subplots(figsize=(W, 2.35), layout="constrained"); w = 0.27; x = np.arange(NC)
    for i, n in enumerate(cnt):
        xs = x + (i - 1) * (w + 0.015)
        ax.bar(xs, cnt[n].values, w, color=EXP_COL[i], label=f"{lab[n]} (n = {int(cnt[n].sum()):,})")
        for xi, vi in zip(xs, cnt[n].values):
            ax.text(xi, vi + 30, f"{vi:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=8.5); ax.set_ylabel("Number of images")
    ax.set_ylim(0, 1.25 * max(c.max() for c in cnt.values())); ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}")); ax.grid(axis="x", visible=False); ax.tick_params(axis="x", length=0)
    ax.legend(ncol=3, loc="upper right", fontsize=8)
    NOTES.append("split_distribution: " + "; ".join(f"{n} " + "/".join(str(v) for v in c.values) for n, c in cnt.items()))
    save(fig, "fig_split_distribution")


def fig_augmentation(n_aug=5):
    import torch, torchvision.transforms as T
    # identical to meff_main.py train_tf, without ToTensor/Normalize
    aug = T.Compose([T.RandomResizedCrop(224, scale=(0.8, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                     T.RandomRotation(15), T.ColorJitter(0.2, 0.2, 0.2, 0.02)])
    tr = pd.read_csv(os.path.join(A.split_dir, "train.csv"))
    rng = np.random.default_rng(7); torch.manual_seed(7)
    fig, ax, tw, th, fh = grid(NC, n_aug + 1, left=0.24, top=0.2, bottom=0.02, ws=0.04, hs=0.05)
    for i, c in enumerate(CLASSES):
        rel = rng.choice(tr[tr.class_name == c].relpath.values)
        im = Image.open(os.path.join(A.data_dir, rel)).convert("RGB")
        ax[i, 0].imshow(im); row_label(ax[i, 0], c)
        for j in range(n_aug):
            ax[i, j + 1].imshow(aug(im))
        NOTES.append(f"augmentation {c}: {rel}")
    for j in range(n_aug + 1):
        ax[0, j].set_title("Original" if j == 0 else f"Augmented {j}", loc="center", fontsize=8.5)
    for i in range(NC):   # separate the original column
        ax[i, 0].spines[:].set_color(INK); ax[i, 0].spines[:].set_linewidth(1.2)
    save(fig, "fig_augmentation_examples")


# ---------------------------------------------------------------------------------------------------- result figures
def load_runs():
    runs = {}
    for s in SEEDS:
        rd = os.path.join(A.runs_dir, f"meff_s{s}")
        runs[s] = dict(hist=pd.read_csv(os.path.join(rd, "history.csv")), te=pd.read_csv(os.path.join(rd, "test_predictions.csv")))
    te = [runs[s]["te"] for s in SEEDS]
    for t in te[1:]:
        assert (t.relpath.values == te[0].relpath.values).all() and (t.true.values == te[0].true.values).all()
    hv, mp, _ = mv.hard_vote([t.pred.values for t in te], [t[PC].values for t in te])
    ens = te[0][["relpath", "true"]].copy(); ens["pred"] = hv
    for i, c in enumerate(CLASSES):
        ens[f"p_{c}"] = mp[:, i]
    y = ens.true.values   # per-class metrics of each expert and of the ensemble, computed here from the predictions
    res = pd.DataFrame([dict(model=f"MEFF_s{s}", **mv.full_metrics(y, runs[s]["te"].pred.values, runs[s]["te"][PC].values)) for s in SEEDS]
                       + [dict(model="MEFF_hard", **mv.full_metrics(y, hv, mp))])
    if A.results_dir:   # optional cross-check with the official evaluation
        cm_off = pd.read_csv(os.path.join(A.results_dir, "confusion_MEFF_hard.csv"), index_col=0).values
        assert (cm_off == mv.confusion(y, hv)).all(), "hard vote differs from the official eval"
        off = pd.read_csv(os.path.join(A.results_dir, "results.csv")).set_index("model")
        k = [c for c in res.columns if c != "model"]
        assert np.allclose(off.loc[res.model, k].values.astype(float), res[k].values.astype(float), atol=1e-9), "metrics differ from results.csv"
        print("checked against", A.results_dir, ": confusion matrix and all metrics identical")
    return runs, ens, mp, res


def fig_curves(runs):
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.6), layout="constrained")
    for i, s in enumerate(SEEDS):
        h = runs[s]["hist"]; c = EXP_COL[i]; e = h.epoch.values
        b = int(h.val_macro_f1.idxmax()); be = int(h.epoch[b])
        ax[0].plot(e, 100 * h.train_acc, color=c, lw=1.3); ax[0].plot(e, 100 * h.val_acc, color=c, lw=1.3, ls="--")
        ax[1].plot(e, h.train_loss, color=c, lw=1.3); ax[1].plot(e, h.val_loss, color=c, lw=1.3, ls="--")
        ax[2].plot(e, 100 * h.val_macro_f1, color=c, lw=1.3)
        ax[2].plot(be, 100 * h.val_macro_f1[b], EXP_MK[i], ms=[9, 7.5, 7.5][i], mfc="white", mec=c, mew=1.6, zorder=5)   # hollow: seeds 42 and 44 peak at the same epoch
        NOTES.append(f"curves: meff_s{s}: {len(h)} epochs, best val macro-F1 {h.val_macro_f1[b]:.4f} at epoch {be}")
    ax[0].set_title("(a) Accuracy"); ax[0].set_ylabel("Accuracy (%)")
    ax[1].set_title("(b) Focal loss"); ax[1].set_ylabel("Loss")
    ax[2].set_title("(c) Validation macro-F1"); ax[2].set_ylabel("Macro-F1 (%)")
    lo = np.floor(min(100 * runs[s]["hist"].val_macro_f1.min() for s in SEEDS) / 5) * 5
    ax[2].set_ylim(max(lo, 50), 95)
    for a in ax:
        a.set_xlabel("Epoch"); a.set_xlim(0, max(len(runs[s]["hist"]) for s in SEEDS) + 1)
    hs = [Line2D([], [], color=EXP_COL[i], lw=1.6, marker=EXP_MK[i], ms=5.5, mfc="white", mec=EXP_COL[i], mew=1.3, label=EXPN[s]) for i, s in enumerate(SEEDS)]
    hs += [Line2D([], [], color=INK2, lw=1.3, label="training"), Line2D([], [], color=INK2, lw=1.3, ls="--", label="validation")]
    fig.legend(handles=hs, loc="outside upper center", ncol=5, fontsize=8, columnspacing=1.2, handlelength=2.0)
    save(fig, "fig_curves")


def fig_learning_curves(runs):
    """Four separate graphs: training accuracy, validation accuracy, training loss, validation loss (three experts).
    Marker on the validation graphs = epoch of the saved checkpoint (best validation macro-F1)."""
    fig, ax = plt.subplots(2, 2, figsize=(W, 4.75), layout="constrained")
    panels = [(ax[0, 0], "train_acc", 100, "(a) Training accuracy", "Accuracy (%)", False),
              (ax[0, 1], "val_acc", 100, "(b) Validation accuracy", "Accuracy (%)", True),
              (ax[1, 0], "train_loss", 1, "(c) Training loss", "Focal loss", False),
              (ax[1, 1], "val_loss", 1, "(d) Validation loss", "Focal loss", True)]
    for i, s in enumerate(SEEDS):
        h = runs[s]["hist"]; e = h.epoch.values; b = int(h.val_macro_f1.idxmax())
        for a, col, k, t, yl, mark in panels:
            a.plot(e, k * h[col], color=EXP_COL[i], ls=EXP_LS[i], lw=1.3)
            if mark:
                a.plot(e[b], k * h[col][b], EXP_MK[i], ms=[9, 7.5, 7.5][i], mfc="white", mec=EXP_COL[i], mew=1.6, zorder=5)
        NOTES.append(f"learning curves: meff_s{s}: checkpoint epoch {int(e[b])}: train acc {h.train_acc[b]:.4f}, val acc {h.val_acc[b]:.4f}, "
                     f"train loss {h.train_loss[b]:.4f}, val loss {h.val_loss[b]:.4f}")
    xmax = max(len(runs[s]["hist"]) for s in SEEDS) + 1
    for a, col, k, t, yl, mark in panels:
        a.set_title(t); a.set_ylabel(yl); a.set_xlabel("Epoch"); a.set_xlim(0, xmax)
        if "loss" in col: a.set_ylim(bottom=0)
    hs = [Line2D([], [], color=EXP_COL[i], ls=EXP_LS[i], lw=1.6, label=EXPN[s]) for i, s in enumerate(SEEDS)]
    hs += [Line2D([], [], ls="none", marker="o", ms=7, mfc="white", mec=INK2, mew=1.4, label="saved checkpoint (best validation macro-F1)")]
    fig.legend(handles=hs, loc="outside upper center", ncol=2, fontsize=8, columnspacing=1.5, handlelength=2.6)
    save(fig, "fig_learning_curves")


def fig_perclass(res):
    """Dot plot: one row per class, one marker per model; x-axis zoomed to the data range (stated in the caption)."""
    models = [(f"MEFF_s{s}", EXPN[s], EXP_COL[i], EXP_MK[i], 5.5) for i, s in enumerate(SEEDS)] + \
             [("MEFF_hard", "MEFF (hard vote)", MEFF_COL, "D", 6.5)]
    R = res.set_index("model")
    vals = np.array([[100 * R.loc[m, f"{c}_{k}"] for c in CLASSES for k in ("precision", "recall", "f1")] for m, *_ in models])
    x0 = np.floor(vals.min() / 5) * 5
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.45), sharey=True, layout="constrained")
    off = np.linspace(-0.27, 0.27, len(models))
    for j, (k, t) in enumerate([("precision", "(a) Precision"), ("recall", "(b) Recall"), ("f1", "(c) F1 score")]):
        a = ax[j]
        for ci in range(NC):
            a.axhspan(ci - 0.45, ci + 0.45, color="#f4f3ef" if ci % 2 == 0 else "white", zorder=0, lw=0)
        for (m, lab, col, mk, ms), o in zip(models, off):
            v = [100 * R.loc[m, f"{c}_{k}"] for c in CLASSES]
            a.plot(v, np.arange(NC) + o, mk, ms=ms, mfc=col, mec="white" if m == "MEFF_hard" else col, mew=0.8,
                   ls="none", label=lab, zorder=4 if m == "MEFF_hard" else 3)
        a.set_title(t); a.set_xlim(x0, 100.5); a.set_xticks(np.arange(x0, 101, 5))
        a.set_xlabel("Test set (%)"); a.grid(axis="y", visible=False)
    ax[0].set_yticks(range(NC)); ax[0].set_yticklabels(CLASSES, fontsize=8.5); ax[0].set_ylim(NC - 0.5, -0.5)
    ax[0].tick_params(axis="y", length=0)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4, fontsize=8, columnspacing=1.2, handletextpad=0.3)
    NOTES.append(f"perclass: x-axis starts at {x0:g}% (lowest value {vals.min():.2f}%)")
    save(fig, "fig_perclass")


def fig_pred_examples(ens, k=4):
    fig, ax, tw, th, fh = grid(NC, k, width=5.0, left=0.24, top=0.2, bottom=0.03, ws=0.06, hs=0.22)
    rng = np.random.default_rng(42)
    for i, c in enumerate(CLASSES):
        pool = ens[ens.true == i].sort_values("relpath")
        pick = pool.iloc[np.sort(rng.choice(len(pool), min(k, len(pool)), replace=False))]
        NOTES.append(f"pred_examples {c}: " + ", ".join(pick.relpath))
        for j in range(k):
            a = ax[i, j]; r = pick.iloc[j]; ok = r.pred == r.true
            a.imshow(img(r.relpath))
            a.spines[:].set_visible(True); a.spines[:].set_color(OK_COL if ok else BAD_COL); a.spines[:].set_linewidth(2.2)
            a.set_title(f"{'✓' if ok else '✗'} {CLASSES[r.pred]} ({r[f'p_{CLASSES[r.pred]}']:.2f})", loc="center", fontsize=8,
                        color=OK_COL if ok else BAD_COL, pad=3)
        row_label(ax[i, 0], f"({'abcd'[i]}) true: {c}")
    save(fig, "fig_pred_examples")


def median_pick(d, col):
    d = d.assign(_p=d.apply(lambda r: r[f"p_{CLASSES[int(r[col])]}"], axis=1)).sort_values(["_p", "relpath"]).reset_index(drop=True)
    return d.iloc[(len(d) - 1) // 2]


def picks(ens):
    correct, errors = [], []
    for i in range(NC):
        ok = ens[(ens.true == i) & (ens.pred == i)]; bad = ens[(ens.true == i) & (ens.pred != i)]
        if len(ok): correct.append(median_pick(ok, "true"))
        if len(bad): errors.append(median_pick(bad, "pred"))
    NOTES.append("correct examples (median rule): " + ", ".join(r.relpath for r in correct))
    NOTES.append("error examples (median rule): " + ", ".join(f"{r.relpath}->{CLASSES[int(r.pred)]}" for r in errors))
    return correct, errors


def fig_prob_examples(correct):
    left, ws, right = 0.78, 0.14, 0.06
    tw = (W - left - right - 3 * ws) / NC; bh = 1.05
    top, gap, bottom = 0.22, 0.12, 0.42
    fh = top + tw + gap + bh + bottom
    fig = plt.figure(figsize=(W, fh))
    for i, r in enumerate(correct):
        x = left + i * (tw + ws)
        a = fig.add_axes([x / W, (fh - top - tw) / fh, tw / W, tw / fh]); a.imshow(img(r.relpath)); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"({'abcd'[i]}) true: {CLASSES[i]}", loc="center", fontsize=8.5)
        b = fig.add_axes([x / W, bottom / fh, tw / W, bh / fh])
        p = [r[f"p_{k}"] for k in CLASSES]
        b.barh(range(NC), p, color=[EXP_COL[0] if k == i else MUTED for k in range(NC)], height=0.65)
        for k in range(NC):
            b.text(p[k] + 0.03, k, f"{p[k]:.2f}", va="center", fontsize=8)
        b.set_yticks(range(NC)); b.set_yticklabels(CLASSES if i == 0 else [], fontsize=8); b.tick_params(axis="y", length=0)
        b.invert_yaxis(); b.set_xlim(0, 1.3); b.set_xticks([0, 0.5, 1]); b.set_xticklabels(["0", "0.5", "1"])
        b.grid(axis="y", visible=False)
        NOTES.append(f"prob_examples {CLASSES[i]}: {r.relpath}")
    fig.text((left + (W - left - right) / 2) / W, 0.04 / fh,
             "Mean softmax probability of the three experts", ha="center", va="bottom", fontsize=8.5)
    save(fig, "fig_prob_examples")


def fig_confusion(runs, ens):
    cms = [(EXPN[s], mv.confusion(runs[s]["te"].true.values, runs[s]["te"].pred.values)) for s in SEEDS]
    cms.append(("MEFF (hard vote)", mv.confusion(ens.true.values, ens.pred.values)))
    fig, ax = plt.subplots(2, 2, figsize=(W, 5.75), layout="constrained")
    for k, (a, (name, cm)) in enumerate(zip(ax.flat, cms)):
        cm = np.asarray(cm, float); pct = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
        a.imshow(pct, cmap=BLUES, vmin=0, vmax=100)
        for i in range(NC):
            for j in range(NC):
                a.text(j, i, f"{pct[i, j]:.1f}%\n({int(cm[i, j])})", ha="center", va="center", fontsize=8,
                       color="white" if pct[i, j] > 55 else INK, fontweight="bold" if i == j else "normal")
        a.set_xticks(range(NC)); a.set_yticks(range(NC)); a.grid(False); a.tick_params(length=0)
        a.set_xticklabels(CLASSES, fontsize=8, rotation=25, ha="right", rotation_mode="anchor"); a.set_yticklabels(CLASSES, fontsize=8)
        a.set_xlabel("Predicted class"); a.set_ylabel("True class")
        a.set_title(f"({'abcd'[k]}) {name}", fontweight="bold" if "MEFF" in name else "normal")
        for s in a.spines.values():
            s.set_visible(False)
    save(fig, "fig_confusion")


def fig_roc_pr(runs, ens, mp):
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
    y = ens.true.values; Y1 = np.eye(NC)[y]; g = np.linspace(0, 1, 1001)
    fig, ax = plt.subplots(2, 2, figsize=(W, 5.6), layout="constrained")
    tprs = []
    for k in range(NC):
        f_, t_, _ = roc_curve(Y1[:, k], mp[:, k]); tprs.append(np.interp(g, f_, t_))
        ax[0, 0].plot(f_, t_, color=CLS_COL[k], ls=CLS_LS[k], lw=1.4, label=f"{CLASSES[k]} ({roc_auc_score(Y1[:, k], mp[:, k]):.3f})")
        p_, r_, _ = precision_recall_curve(Y1[:, k], mp[:, k])
        ax[0, 1].plot(r_, p_, color=CLS_COL[k], ls=CLS_LS[k], lw=1.4,
                      label=f"{CLASSES[k]} ({average_precision_score(Y1[:, k], mp[:, k]):.3f}; {100 * Y1[:, k].mean():.1f}%)")
    mt = np.mean(tprs, 0); mt[0] = 0
    auc_m = roc_auc_score(y, mp, multi_class="ovr", average="macro")
    ax[0, 0].plot(g, mt, color=INK, lw=2.0, label=f"Macro-average ({auc_m:.3f})")
    for k, (s, col, ls) in enumerate(zip(SEEDS, EXP_COL, EXP_LS)):
        P = runs[s]["te"][PC].values
        m_ = np.mean([np.interp(g, *roc_curve(Y1[:, c], P[:, c])[:2]) for c in range(NC)], 0); m_[0] = 0
        lab = f"{EXPN[s]} ({roc_auc_score(y, P, multi_class='ovr', average='macro'):.3f})"
        for a in ax[1]:
            a.plot(g, m_, color=col, ls=ls, lw=1.4, label=lab)
    for a in ax[1]:
        a.plot(g, mt, color=MEFF_COL, lw=2.0, label=f"MEFF, mean softmax ({auc_m:.3f})")
    for a in (ax[0, 0], ax[1, 0]):
        a.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls=":")
    zx, zy = (0, 0.25), (0.75, 1.0)
    ax[1, 0].add_patch(plt.Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, ec=INK2, lw=0.8, ls="--"))
    spec = [(ax[0, 0], "(a) ROC per class, MEFF", "False positive rate", "True positive rate", "lower right", "Class (AUC)"),
            (ax[0, 1], "(b) Precision-recall per class, MEFF", "Recall", "Precision", "lower left", "Class (AP; prevalence)"),
            (ax[1, 0], "(c) Macro-average ROC", "False positive rate", "True positive rate", "lower right", "Model (macro-AUC)"),
            (ax[1, 1], "(d) Zoom of the dashed box in (c)", "False positive rate", "True positive rate", "lower right", "Model (macro-AUC)")]
    for a, t, xl, yl, loc, lt in spec:
        a.set_title(t); a.set_xlabel(xl); a.set_ylabel(yl)
        a.legend(loc=loc, fontsize=8, title=lt, title_fontsize=8, alignment="left", handlelength=2.4)
        if a is not ax[1, 1]:
            a.set_xlim(-0.01, 1.01); a.set_ylim(-0.01, 1.01); a.set_xticks([0, 0.25, 0.5, 0.75, 1]); a.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax[1, 1].set_xlim(*zx); ax[1, 1].set_ylim(*zy)
    save(fig, "fig_roc_pr")


def canny_map(rgb):
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    eq = cv2.equalizeHist(cv2.GaussianBlur(gray, (5, 5), 0)); med = float(np.median(eq))
    return cv2.Canny(eq, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))   # same rule as handcrafted_42


def werr(m, s):
    return np.vstack([np.minimum(s, m), s])   # SD whiskers; lower whisker clipped at 0


def fig_handcrafted():
    te = pd.read_csv(os.path.join(A.split_dir, "test.csv"))
    ek = dict(lw=0.7, capsize=1.8, ecolor=INK2)
    for i, c in enumerate(CLASSES):
        d = te[te.label == i].sort_values("relpath")
        F = np.stack([mv.handcrafted_42(img(r)) for r in d.relpath]); mu, sd = F.mean(0), F.std(0)
        rep = d.relpath.iloc[int(np.argmin(((F - mu) ** 2).sum(1))) ]
        NOTES.append(f"handcrafted {c}: {len(d)} test images; patch shown = {rep} (closest to the class mean)")
        rgb = img(rep)
        fig = plt.figure(figsize=(W, 4.15), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.04, h_pad=0.06, wspace=0.06, hspace=0.08)
        gs = fig.add_gridspec(2, 6, height_ratios=[1, 1])
        a = fig.add_subplot(gs[0, 0:2]); a.imshow(rgb); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.set_title("(a) Test patch closest to class mean", fontsize=8.5)
        a = fig.add_subplot(gs[0, 2:4]); a.imshow(255 - canny_map(rgb), cmap="gray"); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.set_title("(b) Canny edge map", fontsize=8.5)
        for s in a.spines.values():
            s.set_visible(True); s.set_color(MUTED)
        a = fig.add_subplot(gs[0, 4:6])
        a.bar(range(10), mu[:10], yerr=werr(mu[:10], sd[:10]), color=EXP_COL[0], width=0.72, error_kw=ek)
        a.set_xticks(range(10)); a.set_xlabel("LBP code (P = 8, R = 1)"); a.set_ylabel("Fraction of pixels")
        a.set_title("(c) Uniform LBP histogram", fontsize=8.5); a.grid(axis="x", visible=False)
        a = fig.add_subplot(gs[1, 0:2])
        a.bar(range(8), mu[34:42], yerr=werr(mu[34:42], sd[34:42]), color=EXP_COL[0], width=0.72, error_kw=ek)
        a.set_xticks(range(8)); a.set_xticklabels([f"{22.5 * k:g}" for k in range(8)], fontsize=8, rotation=45, ha="right", rotation_mode="anchor")
        a.set_xlabel("Edge orientation, bin start (°)"); a.set_ylabel("Fraction of edge pixels")
        a.set_title("(d) Edge-orientation histogram", fontsize=8.5); a.grid(axis="x", visible=False)
        a = fig.add_subplot(gs[1, 2:6]); chc = ["#c62828", "#2e7d32", "#1565c0"]
        for ch in range(3):
            sl = slice(10 + 8 * ch, 18 + 8 * ch)
            a.bar(np.arange(8) + 9 * ch, mu[sl], yerr=werr(mu[sl], sd[sl]), color=chc[ch], width=0.75, error_kw=ek)
        a.set_xticks([3.5, 12.5, 21.5]); a.set_xticklabels(["Red, 8 bins", "Green, 8 bins", "Blue, 8 bins"], fontsize=8)
        a.tick_params(axis="x", length=0); a.set_ylabel("Fraction of pixels")
        a.set_title("(e) RGB colour histograms (dark → bright within each channel)", fontsize=8.5); a.grid(axis="x", visible=False)
        save(fig, f"fig_hc_{c}")


# ---------------------------------------------------------------------------------------------------- model figures
def load_models():
    import torch, torchvision.transforms as T
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tr = pd.read_csv(os.path.join(A.split_dir, "train.csv"))
    print(f"computing handcrafted features of {len(tr)} training images (train mean/SD, as in training) ...")
    FTR = np.stack([mv.handcrafted_42(img(r)) for r in tr.relpath]); MU, SD = FTR.mean(0), FTR.std(0); SD[SD < 1e-6] = 1.0
    hc = lambda rel: torch.tensor((mv.handcrafted_42(img(rel)) - MU) / SD, dtype=torch.float32)[None]
    x_of = lambda rel: tf(Image.open(os.path.join(A.data_dir, rel)).convert("RGB"))[None]
    models = {}
    for s in SEEDS:
        m = mv.build_model("meff", pretrained=False).eval()
        ck = torch.load(os.path.join(A.ckpt_dir, f"best_s{s}.pth"), map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"]); models[s] = m
    return models, x_of, hc


def check_models(models, x_of, hc, runs, n=3):
    import torch
    te = runs[SEEDS[0]]["te"]; chk = te.groupby("true", group_keys=False).apply(lambda d: d.head(n))
    with torch.no_grad():
        X = torch.cat([x_of(r) for r in chk.relpath]); H = torch.cat([hc(r) for r in chk.relpath])
        for s in SEEDS:
            P = torch.softmax(models[s](X, H), 1).numpy(); K = runs[s]["te"].set_index("relpath").loc[chk.relpath, PC].values
            same, diff = (P.argmax(1) == K.argmax(1)).mean(), np.abs(P - K).max()
            print(f"seed {s}: same class {same:.0%}, max |prob diff| {diff:.4f}")
            assert same >= 0.95 and diff < 0.05


def fig_featuremaps(models, x_of, correct):
    import torch, torch.nn.functional as F
    model = models[42]
    stages = [("block1", "Dense block 1\n256 × 56 × 56"), ("block2", "Dense block 2\n512 × 28 × 28"),
              ("block3", "Dense block 3\n1,280 × 14 × 14"), ("final", "CCDGS + TA + SE\n1,664 × 7 × 7")]
    fig, ax, tw, th, fh = grid(len(correct), 5, left=0.24, top=0.36, bottom=0.62, ws=0.05, hs=0.2)
    for r, row in enumerate(correct):
        acts = {}
        hooks = [getattr(model.features, f"denseblock{i}").register_forward_hook(
            lambda m, i_, o, k=f"block{i}": acts.__setitem__(k, o.detach())) for i in (1, 2, 3)]
        with torch.no_grad():
            acts["final"] = model.se(model.ta(model.ccdgs(F.relu(model.features(x_of(row.relpath)))))).detach()
        for h in hooks:
            h.remove()
        A_ = {k: v[0].float().numpy() for k, v in acts.items()}
        im = img(row.relpath, 224)
        ax[r, 0].imshow(im); row_label(ax[r, 0], f"({'abcd'[r]}) {CLASSES[int(row.true)]}")
        for c, (k, lab) in enumerate(stages, 1):
            if k == "final":
                ax[r, c].imshow(im); ax[r, c].imshow(up(A_[k].mean(0)), cmap="inferno", alpha=0.55, vmin=0, vmax=1)
                ax[r, c].set_xlabel("channel mean", fontsize=8, labelpad=2)
            else:
                top = int(np.argmax(A_[k].mean((1, 2)))); ax[r, c].imshow(norm01(A_[k][top], True), cmap="inferno", vmin=0, vmax=1)
                ax[r, c].set_xlabel(f"channel {top}", fontsize=8, labelpad=2)
            if r == 0:
                ax[r, c].set_title(lab, loc="center", fontsize=8)
        if r == 0:
            ax[r, 0].set_title("Test image", loc="center", fontsize=8)
    hbar(fig, W / 2 - 1.0, 0.24, 2.6, 0.09, "inferno", "Activation (scaled)")
    save(fig, "fig_featuremaps_overview")


def fig_gradcam(models, x_of, hc, rows, name):
    import torch, torch.nn.functional as F

    def cam(model, rel, target):
        x, h = x_of(rel), hc(rel)
        with torch.enable_grad():
            a = model.se(model.ta(model.ccdgs(F.relu(model.features(x))))); a.retain_grad()
            logit = model.head(torch.cat([a.mean((2, 3)), h], 1)); model.zero_grad(); logit[0, target].backward()
        w = a.grad.mean((2, 3), keepdim=True)
        return F.relu((w * a).sum(1))[0].detach().numpy(), float(torch.softmax(logit.detach(), 1)[0, target])

    fig, ax, tw, th, fh = grid(len(rows), 5, left=0.24, top=0.36, bottom=0.62, ws=0.05, hs=0.2)
    heads = ["Test image"] + [f"Expert {i + 1}\n(seed {s})" for i, s in enumerate(SEEDS)] + ["MEFF\n(mean of 3 maps)"]
    for r, row in enumerate(rows):
        rel, t, p = row.relpath, int(row.true), int(row.pred)
        im = img(rel, 224); cams = []
        ax[r, 0].imshow(im)
        row_label(ax[r, 0], f"({'abcd'[r]}) true: {CLASSES[t]}")
        ax[r, 0].set_xlabel(f"p = P({CLASSES[p]})", fontsize=8, labelpad=2)   # the class every p in this row refers to
        for j, s in enumerate(SEEDS, 1):
            cm_, pr = cam(models[s], rel, p); cams.append(norm01(cm_))
            ax[r, j].imshow(im); ax[r, j].imshow(up(cm_), cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax[r, j].set_xlabel(f"p = {pr:.2f}", fontsize=8, labelpad=2)
        ax[r, -1].imshow(im); ax[r, -1].imshow(up(np.mean(cams, 0)), cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax[r, -1].set_xlabel(f"{CLASSES[p]}, p = {row[f'p_{CLASSES[p]}']:.2f}", fontsize=8, labelpad=2,
                             color=OK_COL if p == t else BAD_COL, fontweight="bold")
        ax[r, -1].spines[:].set_color(MEFF_COL); ax[r, -1].spines[:].set_linewidth(1.4)
        NOTES.append(f"{name}: {rel} true {CLASSES[t]} pred {CLASSES[p]}")
    for j, t_ in enumerate(heads):
        ax[0, j].set_title(t_, loc="center", fontsize=8, fontweight="bold" if j == 4 else "normal")
    hbar(fig, W / 2 - 1.0, 0.24, 2.6, 0.09, "jet", "Grad-CAM (scaled)")
    save(fig, name)


# ---------------------------------------------------------------------------------------------------- main
ALL = ["samples", "split", "augmentation", "curves", "perclass", "pred", "prob", "confusion", "roc_pr", "hc", "featuremaps", "gradcam"]


def main():
    global A, mv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("runs_dir", "data_dir", "split_dir", "ckpt_dir", "meff_py", "out_dir"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--results_dir", default=None); ap.add_argument("--font_dir", default=None)
    ap.add_argument("--dpi", type=int, default=600); ap.add_argument("--only", nargs="+", choices=ALL, default=ALL)
    A = ap.parse_args(); os.makedirs(A.out_dir, exist_ok=True)
    from matplotlib import font_manager as fm
    if A.font_dir:
        for f in sorted(os.listdir(A.font_dir)):
            if f.lower().endswith(".ttf"): fm.fontManager.addfont(os.path.join(A.font_dir, f))
    used = os.path.basename(fm.findfont(fm.FontProperties(family="Liberation Sans"), fallback_to_default=True))
    print("font:", used)
    if "Liberation" not in used:
        print("WARNING: Liberation Sans not found - text falls back to DejaVu Sans, which is wider; labels may overlap. Use --font_dir.")
    spec = importlib.util.spec_from_file_location("meff_main", A.meff_py); mv = importlib.util.module_from_spec(spec); spec.loader.exec_module(mv)
    assert list(mv.CLASSES) == CLASSES
    todo = set(A.only)
    if "samples" in todo: fig_samples()
    if "split" in todo: fig_split()
    if "augmentation" in todo: fig_augmentation()
    if "hc" in todo: fig_handcrafted()
    if todo & {"curves", "perclass", "pred", "prob", "confusion", "roc_pr", "featuremaps", "gradcam"}:
        runs, ens, mp, res = load_runs(); correct, errors = picks(ens)
        if "curves" in todo: fig_curves(runs); fig_learning_curves(runs)
        if "perclass" in todo: fig_perclass(res)
        if "pred" in todo: fig_pred_examples(ens)
        if "prob" in todo: fig_prob_examples(correct)
        if "confusion" in todo: fig_confusion(runs, ens)
        if "roc_pr" in todo: fig_roc_pr(runs, ens, mp)
        if todo & {"featuremaps", "gradcam"}:
            models, x_of, hc = load_models(); check_models(models, x_of, hc, runs)
            if "featuremaps" in todo: fig_featuremaps(models, x_of, correct)
            if "gradcam" in todo:
                fig_gradcam(models, x_of, hc, correct, "fig_gradcam_correct")
                fig_gradcam(models, x_of, hc, errors, "fig_gradcam_errors")
    with open(os.path.join(A.out_dir, "figure_notes.txt"), "w" if set(A.only) == set(ALL) else "a") as f:
        f.write("\n".join(NOTES) + "\n")
    old = pd.read_csv(os.path.join(A.out_dir, "figure_sizes.csv")) if os.path.exists(os.path.join(A.out_dir, "figure_sizes.csv")) else None
    new = pd.DataFrame(SIZES, columns=["figure", "width_in", "height_in"])
    if old is not None:
        new = pd.concat([old[~old.figure.isin(new.figure)], new])
    new["word_width_px_at_100pct"] = (96 * new.width_in).round().astype(int)
    new.sort_values("figure").to_csv(os.path.join(A.out_dir, "figure_sizes.csv"), index=False)
    print(new.sort_values("figure").to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
