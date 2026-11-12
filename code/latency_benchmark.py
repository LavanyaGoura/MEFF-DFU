"""Measured complexity and latency of the four variants (DSG, DSG+SE, DSG+HC, MEFF expert),
each as a single network and as a three-network hard-vote ensemble.

Network only : random input, forward passes only (batch 1).
End to end   : one test image -> read JPEG -> (42-D descriptor on the CPU, if the variant uses it) -> resize and
               normalise -> forward pass(es) -> softmax -> hard vote -> class index on the CPU (batch 1).
Weights are random (no checkpoints needed): the compute and therefore the latency do not depend on the weight values.
Usage: python latency_benchmark.py --code_dir <dir with meff_ablation.py> --data_dir <DFUC2021_5955> --test_csv split/test.csv --out_dir out
"""
import argparse, json, os, platform, sys, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
for k in ("--code_dir", "--data_dir", "--test_csv", "--out_dir"): ap.add_argument(k, required=True)
ap.add_argument("--n_images", type=int, default=200, help="test images timed end to end")
ap.add_argument("--n_warmup", type=int, default=20)
ap.add_argument("--n_runs", type=int, default=100, help="timed forward passes, network only")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
sys.path.insert(0, a.code_dir)
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode
import meff_ablation as mv

dev = torch.device("cuda:0" if a.device == "cuda" and torch.cuda.is_available() else "cpu")
os.makedirs(a.out_dir, exist_ok=True)
NETS = {}
for v in mv.VARIANTS:
    NETS[v] = []
    for k in range(3):
        torch.manual_seed(k)
        NETS[v].append(mv.build_model(v, pretrained=False).to(dev).eval())
eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # as in meff_ablation.py

te = pd.read_csv(a.test_csv)
rel = te.sample(n=min(a.n_images, len(te)), random_state=0).relpath.tolist()
paths = [os.path.join(a.data_dir, r) for r in rel]
F = np.stack([mv.handcrafted_42(np.array(Image.open(p).convert("RGB"))) for p in paths])   # only for the z-score constants
mu, sd = F.mean(0), F.std(0); sd[sd < 1e-6] = 1.0                                          # (values do not affect timing)


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def stats(ts):
    ms = 1000 * np.asarray(ts)
    return float(np.median(ms)), float(np.percentile(ms, 25)), float(np.percentile(ms, 75))


def net_only(v, k, amp):
    models = NETS[v][:k]
    x = torch.randn(1, 3, 224, 224, device=dev)
    h = torch.randn(1, 42, device=dev) if mv.USES_HC[v] else None
    ts = []
    with torch.inference_mode(), torch.autocast(device_type=dev.type, enabled=amp):
        for i in range(a.n_warmup + a.n_runs):
            t0 = time.perf_counter()
            for m in models:
                m(x, h)
            sync()
            if i >= a.n_warmup:
                ts.append(time.perf_counter() - t0)
    return stats(ts)


def end_to_end(v, k, amp):
    models, use_hc = NETS[v][:k], mv.USES_HC[v]
    order = paths[:min(10, len(paths))] + paths          # first images once more as warm-up
    ts, td = [], []
    with torch.inference_mode():
        for i, p in enumerate(order):
            t0 = time.perf_counter()
            img = Image.open(p).convert("RGB")
            hz = None
            if use_hc:
                t1 = time.perf_counter()
                f = (mv.handcrafted_42(np.array(img)) - mu) / sd
                hz = torch.from_numpy(f.astype(np.float32))[None].to(dev)
                d = time.perf_counter() - t1
            x = eval_tf(img)[None].to(dev)
            with torch.autocast(device_type=dev.type, enabled=amp):
                logits = [m(x, hz) for m in models]
            probs = [torch.softmax(l.float(), 1).cpu().numpy() for l in logits]   # .cpu() waits for the GPU
            if k == 1:
                pred = probs[0].argmax(1)
            else:
                pred, _, _ = mv.hard_vote([q.argmax(1) for q in probs], probs)
            t = time.perf_counter() - t0
            if i >= len(order) - len(paths):
                ts.append(t)
                if use_hc:
                    td.append(d)
    return stats(ts), (stats(td) if td else (np.nan, np.nan, np.nan))


rows = []
precisions = [("fp32", False)] + ([("amp_fp16", True)] if dev.type == "cuda" else [])
for v in mv.VARIANTS:
    m0 = NETS[v][0]
    x = torch.randn(1, 3, 224, 224, device=dev); h = torch.randn(1, 42, device=dev) if mv.USES_HC[v] else None
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            m0(x, h)
    flops = fc.get_total_flops(); prm = sum(p.numel() for p in m0.parameters())
    for k in (1, 3):
        for pname, amp in precisions:
            n_med, n_25, n_75 = net_only(v, k, amp)
            (e_med, e_25, e_75), (d_med, _, _) = end_to_end(v, k, amp)
            rows.append(dict(variant=v, SE=mv.USES_SE[v], handcrafted=mv.USES_HC[v], networks=k, precision=pname,
                             params_M=k * prm / 1e6, gflops=k * flops / 1e9,
                             net_ms_median=n_med, net_ms_p25=n_25, net_ms_p75=n_75,
                             e2e_ms_median=e_med, e2e_ms_p25=e_25, e2e_ms_p75=e_75, descriptor_ms_median=d_med))
            print(f"{v:7s} x{k} {pname:9s} net {n_med:7.2f} ms   end-to-end {e_med:7.2f} ms", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out_dir, "latency_all_variants.csv"), index=False)
cpu = platform.processor()
try:
    cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
except Exception:
    pass
env = dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
           cuda=torch.version.cuda, device=str(dev), gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
           cpu=cpu, cpu_threads=torch.get_num_threads(), n_images=len(paths), n_warmup=a.n_warmup, n_runs=a.n_runs,
           meff_v3_1_md5=mv.md5_file(os.path.join(a.code_dir, "meff_ablation.py")), date=time.strftime("%Y-%m-%d %H:%M:%S"))
json.dump(env, open(os.path.join(a.out_dir, "latency_env.json"), "w"), indent=2)
pd.set_option("display.width", 220)
print(df.round(2).to_string(index=False))
print(json.dumps(env, indent=2))
