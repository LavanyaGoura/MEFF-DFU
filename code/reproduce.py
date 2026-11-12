"""Reproduce the performance results and statistical tests of the paper from the saved predictions (CPU, no images).

Runs, in a working folder:
  meff_main.py eval                    main study: ensembles, metrics, bootstrap CIs, McNemar + Holm, decision
  meff_ablation.py ablation            ablation study: comparisons A1-A4, two-way ANOVA, decision
  exploratory_single_models.py         exploratory same-seed comparison of single models
  sensitivity_near_duplicates.py       exploratory analysis without the 39 near-duplicate test images
and compares every output file with the file of the same name in results/ (byte for byte).

The two protocol files are frozen and are checked by MD5. The evaluation code reads the main protocol under
its internal name PROTOCOL_v3.md next to the split folder, and the ablation code reads PROTOCOL_ABLATION.md
next to itself; this script places copies with those names in the working folder.

Usage (from the repository root):  python code/reproduce.py [--out_dir reproduced]
"""
import argparse, filecmp, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", default="reproduced")
a = ap.parse_args()
out = os.path.abspath(a.out_dir)
work = os.path.join(out, "_work")
if os.path.exists(out):
    shutil.rmtree(out)
os.makedirs(os.path.join(work, "code"))

shutil.copytree(os.path.join(ROOT, "split"), os.path.join(work, "split"))
shutil.copy(os.path.join(ROOT, "PROTOCOL_MAIN.md"), os.path.join(work, "PROTOCOL_v3.md"))
for f in ("meff_main.py", "meff_ablation.py"):
    shutil.copy(os.path.join(ROOT, "code", f), os.path.join(work, "code", f))
shutil.copy(os.path.join(ROOT, "PROTOCOL_ABLATION.md"), os.path.join(work, "code", "PROTOCOL_ABLATION.md"))

runs, py = os.path.join(ROOT, "runs"), sys.executable
eval_out, abl_out = os.path.join(out, "eval"), os.path.join(out, "ablation")
steps = [
    [py, os.path.join(work, "code", "meff_main.py"), "eval", "--runs_dir", runs,
     "--split_dir", os.path.join(work, "split"), "--out_dir", eval_out],
    [py, os.path.join(work, "code", "meff_ablation.py"), "ablation", "--runs_dir", runs, "--out_dir", abl_out],
    [py, os.path.join(ROOT, "code", "exploratory_single_models.py"), "--runs_dir", runs, "--out_dir", eval_out],
    [py, os.path.join(ROOT, "code", "sensitivity_near_duplicates.py"), "--runs_dir", runs,
     "--split_dir", os.path.join(ROOT, "split"), "--out_dir", eval_out],
]
for cmd in steps:
    print("running", os.path.basename(cmd[1]), cmd[2] if not cmd[2].startswith("--") else "", flush=True)
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{r.stderr}")

print("\nComparison with results/ (byte for byte)")
n_same, n_diff = 0, 0
for d, ref in ((eval_out, os.path.join(ROOT, "results")), (abl_out, os.path.join(ROOT, "results", "ablation"))):
    for f in sorted(os.listdir(d)):
        r = os.path.join(ref, f)
        same = os.path.exists(r) and filecmp.cmp(os.path.join(d, f), r, shallow=False)
        n_same += same; n_diff += not same
        print(f"  {'identical' if same else 'DIFFERENT'}  {os.path.relpath(r, ROOT)}")
shutil.rmtree(work)
print(f"\n{n_same} identical, {n_diff} different. Outputs kept in {out}")
sys.exit(1 if n_diff else 0)
