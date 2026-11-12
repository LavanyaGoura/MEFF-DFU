#!/usr/bin/env python3
"""
MEFF v3 pipeline - one file, three commands.

  train       train ONE model (variant meff|dsg, one seed); resumable; stops cleanly before the wall limit
  eval        ensembles, metrics, paired group bootstrap, McNemar+Holm, decision rule (reads saved predictions)
  verify      check the dataset folder matches the frozen split (5,955 files, MD5s) before any training
  complexity  parameters, FLOPs and latency for one MEFF expert, the MEFF ensemble, and one DSG model

Variants
  meff : DenseNet-169 -> CCDGS -> Triplet Attention -> SE -> GAP -> [concat 42-D handcrafted] -> head
  dsg  : DenseNet-169 -> CCDGS -> Triplet Attention        -> GAP ->                          head
         (Dense-ShuffleGCANet re-implementation; Ajay, Bisht, Karthik, IEEE Access 13, 2025)

Examples
  python meff_v3.py train --variant meff --seed 42 --data_dir DFUC2021_deduplicated_5955 \
         --split_dir MEFF_v3/split --out_dir MEFF_v3/runs/meff_s42
  python meff_v3.py eval --runs_dir MEFF_v3/runs --split_dir MEFF_v3/split --out_dir MEFF_v3/results
  python meff_v3.py complexity --out_dir MEFF_v3/results
"""
import argparse, hashlib, json, os, platform, random, sys, time

import numpy as np
import pandas as pd

CLASSES = ["NONE", "INFECTION", "ISCHAEMIA", "BOTH"]
NC = len(CLASSES)

# ----------------------------------------------------------------------------------------------
# Frozen training recipe (pre-registered in PROTOCOL_v3.md - do not change after Part A)
# ----------------------------------------------------------------------------------------------
CFG = dict(
    img_size=224, batch_size=32, max_epochs=100, early_stop_patience=15,
    optimizer="Adam", lr=5e-5, weight_decay=1e-3,
    scheduler="ReduceLROnPlateau(mode=max on val macro-F1, factor=0.5, patience=5, min_lr=1e-7)",
    loss="FocalLoss(gamma=2.0, no alpha)", sampler="WeightedRandomSampler (inverse class frequency)",
    dropout=0.5, hidden=512, hc_dim=42, ccdgs_groups=4, se_reduction=16, ta_kernel=7,
    train_aug="RandomResizedCrop(224,0.8-1.0), HFlip, VFlip, Rotation(15), ColorJitter(0.2,0.2,0.2,0.02)",
    eval_aug="none (ToTensor + ImageNet normalisation)",
    checkpoint_selection="best validation macro-F1",
)


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *a, flush=True)


# ----------------------------------------------------------------------------------------------
# Handcrafted 42-D descriptor: LBP (10) + RGB colour histogram (24) + edge-orientation histogram (8)
# ----------------------------------------------------------------------------------------------
def handcrafted_42(rgb):
    import cv2
    from skimage.feature import local_binary_pattern
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # LBP uniform P=8,R=1 -> codes 0..9 -> 10 bins
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_h = np.histogram(lbp, bins=np.arange(11))[0].astype(np.float64)
    lbp_h /= max(lbp_h.sum(), 1)
    # RGB colour histogram, 8 bins per channel
    col = []
    for ch in range(3):
        h = np.histogram(rgb[:, :, ch], bins=8, range=(0, 256))[0].astype(np.float64)
        col.append(h / max(h.sum(), 1))
    # Edge-orientation histogram: Sobel orientation at Canny edge pixels, 8 bins over [0, pi)
    eq = cv2.equalizeHist(cv2.GaussianBlur(gray, (5, 5), 0))  # blur + histogram equalisation (as in original MEFF)
    med = float(np.median(eq))
    edges = cv2.Canny(eq, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
    gx = cv2.Sobel(eq, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_64F, 0, 1, ksize=3)
    ang = np.mod(np.arctan2(gy, gx), np.pi)[edges > 0]
    e_h = np.histogram(ang, bins=8, range=(0, np.pi))[0].astype(np.float64)
    e_h /= max(e_h.sum(), 1)
    f = np.concatenate([lbp_h, *col, e_h]).astype(np.float32)
    assert f.shape == (42,)
    return f


def load_features(data_dir, relpaths, cache_path):
    """Compute (or load cached) raw 42-D features for every relpath. Atomic write for parallel jobs."""
    from PIL import Image
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        cached = dict(zip(z["relpath"].tolist(), z["feat"]))
        if all(r in cached for r in relpaths):
            return cached
    log(f"computing handcrafted features for {len(relpaths)} images ...")
    feats = np.stack([handcrafted_42(np.array(Image.open(os.path.join(data_dir, r)).convert("RGB")))
                      for r in relpaths])
    tmp = f"{cache_path}.{os.getpid()}.tmp.npz"
    np.savez(tmp, relpath=np.array(relpaths), feat=feats)
    os.replace(tmp, cache_path)
    return dict(zip(relpaths, feats))


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
def build_model(variant, pretrained=True):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision

    class CCDGS(nn.Module):
        """Channel-Centric Depth-wise Group Shuffle block (spatial maps preserved, residual)."""
        def __init__(self, c, groups):
            super().__init__()
            self.g = groups
            self.gconv = nn.Conv2d(c, c, 3, padding=1, groups=groups, bias=False); self.bn1 = nn.BatchNorm2d(c)
            self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False); self.bn2 = nn.BatchNorm2d(c)
            self.pw = nn.Conv2d(c, c, 1, bias=False); self.bn3 = nn.BatchNorm2d(c)

        def forward(self, x):
            y = F.relu(self.bn1(self.gconv(x)))
            b, c, h, w = y.shape
            y = y.view(b, self.g, c // self.g, h, w).transpose(1, 2).contiguous().view(b, c, h, w)  # shuffle
            y = F.relu(self.bn2(self.dw(y)))
            y = self.bn3(self.pw(y))
            return F.relu(x + y)

    class ZPool(nn.Module):
        def forward(self, x):
            return torch.cat([x.max(1, keepdim=True)[0], x.mean(1, keepdim=True)], 1)

    class Gate(nn.Module):
        def __init__(self, k):
            super().__init__()
            self.z = ZPool(); self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False); self.bn = nn.BatchNorm2d(1)

        def forward(self, x):
            return x * torch.sigmoid(self.bn(self.conv(self.z(x))))

    class TripletAttention(nn.Module):
        """Misra et al. (WACV 2021): C-H, C-W and H-W interaction branches, averaged."""
        def __init__(self, k):
            super().__init__()
            self.ch, self.cw, self.hw = Gate(k), Gate(k), Gate(k)

        def forward(self, x):
            o1 = self.ch(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3)
            o2 = self.cw(x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1)
            o3 = self.hw(x)
            return (o1 + o2 + o3) / 3.0

    class SE(nn.Module):
        def __init__(self, c, r):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(c, c // r, bias=False), nn.ReLU(inplace=True),
                                    nn.Linear(c // r, c, bias=False), nn.Sigmoid())

        def forward(self, x):
            return x * self.fc(x.mean((2, 3)))[:, :, None, None]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            w = torchvision.models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
            self.features = torchvision.models.densenet169(weights=w).features
            self.ccdgs = CCDGS(1664, CFG["ccdgs_groups"])
            self.ta = TripletAttention(CFG["ta_kernel"])
            self.se = SE(1664, CFG["se_reduction"]) if variant == "meff" else nn.Identity()
            self.use_hc = variant == "meff"
            fin = 1664 + (CFG["hc_dim"] if self.use_hc else 0)
            self.head = nn.Sequential(nn.Dropout(CFG["dropout"]), nn.Linear(fin, CFG["hidden"]), nn.ReLU(inplace=True),
                                      nn.Dropout(CFG["dropout"]), nn.Linear(CFG["hidden"], NC))

        def forward(self, x, hc=None):
            f = F.relu(self.features(x))
            f = self.se(self.ta(self.ccdgs(f)))
            f = f.mean((2, 3))
            if self.use_hc:
                assert hc is not None and hc.shape[1] == CFG["hc_dim"]
                f = torch.cat([f, hc], 1)
            return self.head(f)

    assert variant in ("meff", "dsg")
    return Net()


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def read_split(split_dir, limit_per_class=None):
    sp = {}
    for n in ["train", "validation", "test"]:
        d = pd.read_csv(os.path.join(split_dir, f"{n}.csv"))
        if limit_per_class:
            d = d.groupby("label", group_keys=False).head(limit_per_class).reset_index(drop=True)
        sp[n] = d
    for k in ["relpath", "md5", "group_id"]:
        s = {n: set(d[k]) for n, d in sp.items()}
        assert not (s["train"] & s["validation"]) and not (s["train"] & s["test"]) and not (s["validation"] & s["test"]), \
            f"leakage: {k} shared between subsets"
    return sp


def make_loaders(sp, data_dir, feats, seed, batch, workers, img_size):
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    import torchvision.transforms as T
    from PIL import Image

    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = T.Compose([T.RandomResizedCrop(img_size, scale=(0.8, 1.0)), T.RandomHorizontalFlip(),
                          T.RandomVerticalFlip(), T.RandomRotation(15),
                          T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.ToTensor(), norm])
    eval_tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm])

    tr = np.stack([feats[r] for r in sp["train"].relpath])
    mu, sd = tr.mean(0), tr.std(0)
    sd[sd < 1e-6] = 1.0  # standardisation fitted on TRAIN only

    class DS(Dataset):
        def __init__(self, d, tf):
            self.p = d.relpath.tolist(); self.y = d.label.astype(int).tolist(); self.tf = tf
            self.h = [((feats[r] - mu) / sd).astype(np.float32) for r in self.p]

        def __len__(self):
            return len(self.p)

        def __getitem__(self, i):
            # transform is chosen by SPLIT only, never by label
            img = Image.open(os.path.join(data_dir, self.p[i])).convert("RGB")
            return self.tf(img), self.y[i], torch.from_numpy(self.h[i])

    y = sp["train"].label.values
    cw = 1.0 / np.bincount(y, minlength=NC)
    g = torch.Generator(); g.manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(cw[y], dtype=torch.double), len(y), replacement=True, generator=g)

    def wif(wid):
        s = torch.initial_seed() % 2 ** 32
        np.random.seed(s); random.seed(s)

    kw = dict(batch_size=batch, num_workers=workers, worker_init_fn=wif, pin_memory=torch.cuda.is_available())
    dl = dict(train=DataLoader(DS(sp["train"], train_tf), sampler=sampler, **kw),
              validation=DataLoader(DS(sp["validation"], eval_tf), shuffle=False, **kw),
              test=DataLoader(DS(sp["test"], eval_tf), shuffle=False, **kw))
    return dl, g, (train_tf, eval_tf)


def assert_eval_transform_ok(eval_tf):
    import torchvision.transforms as T
    rnd = (T.RandomResizedCrop, T.RandomHorizontalFlip, T.RandomVerticalFlip, T.RandomRotation, T.ColorJitter,
           T.RandomAffine, T.RandomCrop, T.RandomErasing, T.RandomApply, T.RandomChoice)
    assert not any(isinstance(t, rnd) for t in eval_tf.transforms), "eval transform contains random ops"


# ----------------------------------------------------------------------------------------------
# Metrics (fast numpy versions; used in training and bootstrap)
# ----------------------------------------------------------------------------------------------
def confusion(y, p):
    return np.bincount(y * NC + p, minlength=NC * NC).reshape(NC, NC)


def macro_f1(y, p):
    cm = confusion(y, p).astype(float)
    tp = np.diag(cm); prec_d = cm.sum(0); rec_d = cm.sum(1)
    f1 = np.where(prec_d + rec_d > 0, 2 * tp / np.maximum(prec_d + rec_d, 1e-12), 0.0)
    return float(f1.mean())


# ----------------------------------------------------------------------------------------------
# TRAIN
# ----------------------------------------------------------------------------------------------
def cmd_train(a):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    t_start = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    done_flag = os.path.join(a.out_dir, "DONE")
    if os.path.exists(done_flag):
        log(f"{a.out_dir} already DONE - nothing to do"); return 0

    smoke = a.smoke
    if smoke:
        CFG.update(max_epochs=a.smoke_epochs, early_stop_patience=99, batch_size=a.smoke_batch)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = dev.type == "cuda"

    sp = read_split(a.split_dir, limit_per_class=16 if smoke else None)
    allrel = sorted(pd.concat(sp.values()).relpath)
    cache = os.path.join(os.path.dirname(os.path.abspath(a.split_dir)), "features42_cache.npz")  # next to split/, not inside
    feats = load_features(a.data_dir, allrel, cache if not smoke else os.path.join(a.out_dir, "features42_smoke.npz"))
    dl, gen, (train_tf, eval_tf) = make_loaders(sp, a.data_dir, feats, a.seed, CFG["batch_size"], a.workers, CFG["img_size"])
    assert_eval_transform_ok(eval_tf)

    model = build_model(a.variant, pretrained=not a.no_pretrained).to(dev)
    if not a.no_pretrained:
        # ImageNet weights check: first conv must equal the torchvision ImageNet tensor
        import torchvision
        ref = torchvision.models.densenet169(weights=torchvision.models.DenseNet169_Weights.IMAGENET1K_V1).features.conv0.weight
        assert torch.allclose(model.features.conv0.weight.detach().cpu(), ref), "ImageNet weights not loaded"
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5, min_lr=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def focal(logits, y, gamma=2.0):
        ce = F.cross_entropy(logits.float(), y, reduction="none")
        return ((1 - torch.exp(-ce)) ** gamma * ce).mean()

    last_p, best_p = os.path.join(a.out_dir, "last.pth"), os.path.join(a.out_dir, "best.pth")
    st = dict(epoch=0, best_f1=-1.0, best_epoch=-1, bad=0, history=[])
    if os.path.exists(last_p):
        ck = torch.load(last_p, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); st = ck["state"]; gen.set_state(ck["gen"])
        torch.set_rng_state(ck["rng_torch"]); np.random.set_state(ck["rng_np"]); random.setstate(ck["rng_py"])
        if use_amp and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        log(f"RESUMED from epoch {st['epoch']} (best macro-F1 {st['best_f1']:.4f} @ epoch {st['best_epoch']})")

    meta = dict(variant=a.variant, seed=a.seed, smoke=smoke, pretrained=not a.no_pretrained, cfg=CFG,
                split_md5={n: md5_file(os.path.join(a.split_dir, f"{n}.csv")) for n in ["train", "validation", "test"]},
                script_md5=md5_file(os.path.abspath(__file__)), python=platform.python_version(),
                torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                n=dict((k, len(v)) for k, v in sp.items()),
                params=sum(p.numel() for p in model.parameters()))
    json.dump(meta, open(os.path.join(a.out_dir, "run_meta.json"), "w"), indent=2)
    log(f"variant={a.variant} seed={a.seed} device={meta['gpu']} params={meta['params']:,} n={meta['n']}")

    def run_eval(loader):
        model.eval(); P, Y, L = [], [], 0.0
        with torch.no_grad():
            for x, y, h in loader:
                x, y, h = x.to(dev, non_blocking=True), y.to(dev), h.to(dev)
                with torch.autocast(device_type=dev.type, enabled=use_amp):
                    out = model(x, h)
                L += focal(out, y).item() * len(y)
                P.append(torch.softmax(out.float(), 1).cpu().numpy()); Y.append(y.cpu().numpy())
        P, Y = np.concatenate(P), np.concatenate(Y)
        return P, Y, L / len(Y)

    epoch_times = []
    while st["epoch"] < CFG["max_epochs"] and st["bad"] < CFG["early_stop_patience"]:
        t0 = time.time(); model.train(); tl, tc, tn = 0.0, 0, 0
        for x, y, h in dl["train"]:
            x, y, h = x.to(dev, non_blocking=True), y.to(dev), h.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                out = model(x, h)
            loss = focal(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); tn += len(y)
        Pv, Yv, vl = run_eval(dl["validation"])
        vf1 = macro_f1(Yv, Pv.argmax(1)); vacc = float((Pv.argmax(1) == Yv).mean())
        sched.step(vf1)
        st["epoch"] += 1
        improved = vf1 > st["best_f1"]
        if improved:
            st.update(best_f1=vf1, best_epoch=st["epoch"], bad=0)
            torch.save(dict(model=model.state_dict(), epoch=st["epoch"], val_macro_f1=vf1), best_p)
        else:
            st["bad"] += 1
        st["history"].append(dict(epoch=st["epoch"], train_loss=tl / tn, train_acc=tc / tn, val_loss=vl,
                                  val_acc=vacc, val_macro_f1=vf1, lr=opt.param_groups[0]["lr"]))
        pd.DataFrame(st["history"]).to_csv(os.path.join(a.out_dir, "history.csv"), index=False)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), state=st, gen=gen.get_state(),
                        rng_torch=torch.get_rng_state(), rng_np=np.random.get_state(), rng_py=random.getstate(),
                        rng_cuda=torch.cuda.get_rng_state_all() if use_amp else None), last_p + ".tmp")
        os.replace(last_p + ".tmp", last_p)
        epoch_times.append(time.time() - t0)
        log(f"ep {st['epoch']:3d} | train loss {tl/tn:.4f} acc {tc/tn:.4f} | val loss {vl:.4f} acc {vacc:.4f} "
            f"macroF1 {vf1:.4f}{' *best*' if improved else ''} | lr {opt.param_groups[0]['lr']:.1e} | {epoch_times[-1]:.0f}s"
            + (f" | peak GPU mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if use_amp else ""))
        elapsed_h = (time.time() - t_start) / 3600
        if elapsed_h + 1.5 * max(epoch_times) / 3600 > a.max_hours:
            log(f"time guard: {elapsed_h:.2f} h used of {a.max_hours} h - state saved, exiting for resubmission")
            return 0

    log(f"training finished: epochs {st['epoch']}, best val macro-F1 {st['best_f1']:.4f} @ epoch {st['best_epoch']}")
    ck = torch.load(best_p, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    for n in ["validation", "test"]:
        P, Y, _ = run_eval(dl[n])
        d = sp[n][["relpath", "group_id"]].copy()
        d["true"] = Y; d["pred"] = P.argmax(1)
        for i, c in enumerate(CLASSES):
            d[f"prob_{c}"] = P[:, i]
        d.to_csv(os.path.join(a.out_dir, f"{'val' if n == 'validation' else n}_predictions.csv"), index=False)
        log(f"{n}: acc {(d.true == d.pred).mean():.4f} macroF1 {macro_f1(d.true.values, d.pred.values):.4f} "
            f"(best checkpoint, epoch {ck['epoch']})")
    meta.update(best_epoch=st["best_epoch"], best_val_macro_f1=st["best_f1"], epochs_run=st["epoch"],
                best_ckpt_md5=md5_file(best_p))
    json.dump(meta, open(os.path.join(a.out_dir, "run_meta.json"), "w"), indent=2)
    open(done_flag, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    if os.path.exists(last_p) and not a.keep_last:
        os.remove(last_p)
    return 0


# ----------------------------------------------------------------------------------------------
# EVAL
# ----------------------------------------------------------------------------------------------
def full_metrics(y, pred, prob):
    from sklearn.metrics import roc_auc_score, f1_score
    cm = confusion(y, pred)
    r = dict(n=len(y), accuracy=float((y == pred).mean()), macro_f1=macro_f1(y, pred),
             weighted_f1=float(f1_score(y, pred, average="weighted", labels=list(range(NC)), zero_division=0)))
    try:
        r["macro_auc"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=list(range(NC))))
    except ValueError:
        r["macro_auc"] = float("nan")
    for i, c in enumerate(CLASSES):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        r[f"{c}_precision"] = float(p); r[f"{c}_recall"] = float(rc)
        r[f"{c}_f1"] = float(2 * p * rc / (p + rc)) if p + rc else 0.0
        r[f"{c}_correct"] = int(tp); r[f"{c}_total"] = int(cm[i, :].sum())
    return r


def hard_vote(preds, probs):
    """Majority vote; 3-way tie -> class with highest mean softmax probability."""
    preds = np.stack(preds, 1); mp = np.mean(probs, 0)
    out, ties = np.empty(len(preds), int), 0
    for i, row in enumerate(preds):
        cnt = np.bincount(row, minlength=NC)
        top = np.flatnonzero(cnt == cnt.max())
        if len(top) > 1:
            ties += 1
            out[i] = top[np.argmax(mp[i, top])]
        else:
            out[i] = top[0]
    return out, mp, ties


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p); adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def cmd_eval(a):
    from scipy.stats import binomtest
    os.makedirs(a.out_dir, exist_ok=True)
    seeds = a.seeds
    runs, ref = {}, None
    for v in ["meff", "dsg"]:
        for s in seeds:
            rd = os.path.join(a.runs_dir, f"{v}_s{s}")
            if not os.path.exists(os.path.join(rd, "DONE")):
                sys.exit(f"ERROR: {rd} is not DONE - finish or resume that run first")
            meta = json.load(open(os.path.join(rd, "run_meta.json")))
            if (meta["smoke"] or not meta["pretrained"]) and not a.allow_smoke:
                sys.exit(f"ERROR: {rd} is a smoke/no-pretrained run - not valid for results")
            te = pd.read_csv(os.path.join(rd, "test_predictions.csv"))
            va = pd.read_csv(os.path.join(rd, "val_predictions.csv"))
            if ref is None:
                ref = te[["relpath", "group_id", "true"]]
            assert te.relpath.equals(ref.relpath) and te.true.equals(ref.true), f"{rd}: test rows differ"
            runs[(v, s)] = dict(te=te, va=va, meta=meta)
    y = ref.true.values; groups = ref.group_id.values
    pc = [f"prob_{c}" for c in CLASSES]
    rows, preds = [], {}
    for v in ["meff", "dsg"]:
        for s in seeds:
            te = runs[(v, s)]["te"]; nm = f"{v.upper()}_s{s}"
            preds[nm] = te.pred.values
            rows.append(dict(model=nm, **full_metrics(y, te.pred.values, te[pc].values)))
        seed_rows = pd.DataFrame([r for r in rows if r["model"].startswith(v.upper() + "_s")]).drop(columns="model")
        rows.append(dict(model=f"{v.upper()}_seed_mean", **seed_rows.mean().to_dict()))
        rows.append(dict(model=f"{v.upper()}_seed_sd", **seed_rows.std(ddof=1).to_dict()))
    ties = {}
    for v in ["meff", "dsg"]:
        ps = [runs[(v, s)]["te"].pred.values for s in seeds]
        pr = [runs[(v, s)]["te"][pc].values for s in seeds]
        hv, mp, nt = hard_vote(ps, pr); ties[v.upper()] = nt
        preds[f"{v.upper()}_hard"] = hv; preds[f"{v.upper()}_soft"] = mp.argmax(1)
        rows.append(dict(model=f"{v.upper()}_hard", **full_metrics(y, hv, mp)))
        rows.append(dict(model=f"{v.upper()}_soft", **full_metrics(y, mp.argmax(1), mp)))
        pd.DataFrame(confusion(y, hv), index=[f"true_{c}" for c in CLASSES],
                     columns=[f"pred_{c}" for c in CLASSES]).to_csv(os.path.join(a.out_dir, f"confusion_{v.upper()}_hard.csv"))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out_dir, "results.csv"), index=False)
    pd.DataFrame([dict(ensemble=k, three_way_ties=v, n=len(y)) for k, v in ties.items()]).to_csv(
        os.path.join(a.out_dir, "ensemble_ties.csv"), index=False)

    # best single MEFF seed chosen on VALIDATION macro-F1 (never on test)
    vf = {s: macro_f1(runs[("meff", s)]["va"].true.values, runs[("meff", s)]["va"].pred.values) for s in seeds}
    best_s = max(seeds, key=lambda s: vf[s])
    comps = [("MEFF_hard", "DSG_hard")] + [("MEFF_hard", f"DSG_s{s}") for s in seeds] + [("MEFF_hard", f"MEFF_s{best_s}")]

    # paired group-level bootstrap
    ug = np.unique(groups); idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(42)
    samples = [np.concatenate([idx_by_g[g] for g in rng.choice(ug, len(ug), replace=True)]) for _ in range(a.n_boot)]
    out = []
    for A, B in comps:
        pa, pb = preds[A], preds[B]
        df1 = [macro_f1(y[i], pa[i]) - macro_f1(y[i], pb[i]) for i in samples]
        dac = [(y[i] == pa[i]).mean() - (y[i] == pb[i]).mean() for i in samples]
        ca, cb = pa == y, pb == y
        b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        out.append(dict(A=A, B=B, delta_macro_f1=macro_f1(y, pa) - macro_f1(y, pb),
                        ci95_low_macro_f1=float(np.percentile(df1, 2.5)), ci95_high_macro_f1=float(np.percentile(df1, 97.5)),
                        delta_accuracy=float(ca.mean() - cb.mean()),
                        ci95_low_accuracy=float(np.percentile(dac, 2.5)), ci95_high_accuracy=float(np.percentile(dac, 97.5)),
                        mcnemar_b_A_right_B_wrong=b, mcnemar_c_A_wrong_B_right=c, mcnemar_p_exact=float(p)))
    comp = pd.DataFrame(out); comp["mcnemar_p_holm"] = holm(comp.mcnemar_p_exact.values)
    comp["best_meff_seed_by_val_macro_f1"] = best_s
    comp.to_csv(os.path.join(a.out_dir, "comparison.csv"), index=False)

    prim = comp.iloc[0]
    if prim.ci95_low_macro_f1 > 0:
        verdict = "MEFF outperforms Dense-ShuffleGCANet"
    elif prim.ci95_high_macro_f1 < 0:
        verdict = "Dense-ShuffleGCANet outperforms MEFF"
    else:
        verdict = "No significant difference between MEFF and Dense-ShuffleGCANet"
    txt = (f"Primary comparison: MEFF_hard - DSG_hard, test macro-F1\n"
           f"delta = {prim.delta_macro_f1:.6f}, 95% CI [{prim.ci95_low_macro_f1:.6f}, {prim.ci95_high_macro_f1:.6f}] "
           f"(paired group bootstrap, {a.n_boot} resamples, seed 42, {len(ug)} test groups)\n"
           f"Rule: CI entirely > 0 -> MEFF outperforms; entirely < 0 -> DSG outperforms; else no significant difference.\n"
           f"DECISION: {verdict}\n")
    open(os.path.join(a.out_dir, "decision.txt"), "w").write(txt)
    prot = os.path.join(os.path.dirname(os.path.abspath(a.split_dir)), "PROTOCOL_v3.md")
    rec = dict(protocol_md5=md5_file(prot) if os.path.exists(prot) else None, script_md5=md5_file(os.path.abspath(__file__)),
               runs={f"{v}_s{s}": runs[(v, s)]["meta"].get("best_ckpt_md5") for v in ["meff", "dsg"] for s in seeds})
    json.dump(rec, open(os.path.join(a.out_dir, "eval_meta.json"), "w"), indent=2)
    pd.set_option("display.width", 200)
    print(res[["model", "n", "accuracy", "macro_f1", "weighted_f1", "macro_auc", "ISCHAEMIA_recall", "BOTH_recall"]].round(4).to_string(index=False))
    print(comp[["A", "B", "delta_macro_f1", "ci95_low_macro_f1", "ci95_high_macro_f1", "mcnemar_p_holm"]].round(4).to_string(index=False))
    print(txt)
    return 0


# ----------------------------------------------------------------------------------------------
# VERIFY: dataset on disk must match the frozen split (every file present, MD5 identical, nothing extra)
# ----------------------------------------------------------------------------------------------
def cmd_verify(a):
    sp = read_split(a.split_dir)
    allr = pd.concat(sp.values())
    on_disk = {os.path.relpath(os.path.join(r, f), a.data_dir) for r, _, fs in os.walk(a.data_dir)
               for f in fs if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    missing = [r for r in allr.relpath if r not in on_disk]
    extra = sorted(on_disk - set(allr.relpath))
    bad = [r for r, m in zip(allr.relpath, allr.md5) if r in on_disk and md5_file(os.path.join(a.data_dir, r)) != m]
    print(f"split rows {len(allr)} | files on disk {len(on_disk)} | missing {len(missing)} | extra {len(extra)} | md5 mismatch {len(bad)}")
    for n, d in sp.items():
        print(f"  {n:10s} {len(d):5d}  " + "  ".join(f"{c}={int((d.class_name == c).sum())}" for c in CLASSES))
    man = os.path.join(a.split_dir, "SPLIT_MANIFEST.json")
    if os.path.exists(man):
        exp = json.load(open(man))["files_md5"]
        for f, m in exp.items():
            ok = md5_file(os.path.join(a.split_dir, f)) == m
            print(f"  {f:16s} md5 {'OK' if ok else 'CHANGED'}")
            bad += [] if ok else [f]
    if missing or extra or bad or len(allr) != 5955:
        print("VERIFY FAILED - do not train"); return 1
    print("VERIFY OK"); return 0


# ----------------------------------------------------------------------------------------------
# COMPLEXITY
# ----------------------------------------------------------------------------------------------
def cmd_complexity(a):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for v in ["meff", "dsg"]:
        m = build_model(v, pretrained=False).to(dev).eval()
        x = torch.randn(1, 3, 224, 224, device=dev); h = torch.randn(1, 42, device=dev)
        with torch.no_grad():
            fc = FlopCounterMode(display=False)
            with fc:
                m(x, h)
            flops = fc.get_total_flops()
            for _ in range(5):
                m(x, h)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts = []
            for _ in range(a.n_timing):
                t0 = time.perf_counter(); m(x, h)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        prm = sum(p.numel() for p in m.parameters())
        rows.append(dict(model="MEFF expert" if v == "meff" else "Dense-ShuffleGCANet (DSG)", params=prm,
                         params_M=prm / 1e6, gflops=flops / 1e9, gmacs=flops / 2e9,
                         latency_ms_batch1_median=1000 * float(np.median(ts)), device=str(dev)))
        if v == "meff":
            rows.append(dict(model="MEFF ensemble (3 experts)", params=3 * prm, params_M=3 * prm / 1e6,
                             gflops=3 * flops / 1e9, gmacs=3 * flops / 2e9,
                             latency_ms_batch1_median=3000 * float(np.median(ts)), device=str(dev)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out_dir, "complexity_v3.csv"), index=False)
    print(df.round(3).to_string(index=False))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--variant", choices=["meff", "dsg"], required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--data_dir", required=True)
    t.add_argument("--split_dir", required=True)
    t.add_argument("--out_dir", required=True)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--max_hours", type=float, default=46.0, help="stop cleanly before this many hours")
    t.add_argument("--keep_last", action="store_true", help="keep last.pth after DONE")
    t.add_argument("--no_pretrained", action="store_true", help="SMOKE TESTS ONLY")
    t.add_argument("--smoke", action="store_true", help="16 images/class/split, few epochs - pipeline test only")
    t.add_argument("--smoke_epochs", type=int, default=2)
    t.add_argument("--smoke_batch", type=int, default=4, help="batch size in smoke mode only (real runs: 32)")
    e = sub.add_parser("eval")
    e.add_argument("--runs_dir", required=True)
    e.add_argument("--split_dir", required=True)
    e.add_argument("--out_dir", required=True)
    e.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    e.add_argument("--n_boot", type=int, default=2000)
    e.add_argument("--allow_smoke", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--data_dir", required=True)
    v.add_argument("--split_dir", required=True)
    c = sub.add_parser("complexity")
    c.add_argument("--out_dir", required=True)
    c.add_argument("--n_timing", type=int, default=50)
    a = ap.parse_args()
    return dict(train=cmd_train, eval=cmd_eval, verify=cmd_verify, complexity=cmd_complexity)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
