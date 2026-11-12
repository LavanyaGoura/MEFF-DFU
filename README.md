# MEFF-DFU: three-expert ensemble for diabetic foot ulcer classification

This repository contains the code, data split, analysis plans, per-image predictions and result files for the paper

> **MEFF: A Three-Expert DenseNet-169 Ensemble for Diabetic Foot Ulcer Classification with a Prespecified, Leakage-Controlled Evaluation** (under review).

All performance results and statistical tests in the paper can be reproduced from the saved predictions on a normal CPU, without the images.

## The method in brief

| Item | Description |
|---|---|
| Task | Classify a foot ulcer image into one of four classes: NONE, INFECTION, ISCHAEMIA, BOTH |
| Data | 5,955 labelled DFUC2021 training images, split 60:20:20 (train 3,573, validation 1,191, test 1,191) |
| MEFF expert | DenseNet-169 + CCDGS block + triplet attention + Squeeze-and-Excitation (SE) block, fused with a 42-D handcrafted descriptor (uniform LBP 10, RGB color histograms 24, edge-orientation histogram 8) |
| MEFF | Hard-vote ensemble of three experts trained with seeds 42, 43 and 44 |
| Baseline | Dense-ShuffleGCANet (DSG), trained with the same split, recipe and seeds |
| Ablation | DSG + SE and DSG + handcrafted features (2 × 2 design: SE yes/no × handcrafted yes/no) |

## Quick start: reproduce the results (CPU, no images needed)

```
python code/reproduce.py
```

The script runs the main evaluation, the ablation analysis and the two exploratory analyses from the saved predictions in `runs/`, writes the outputs to `reproduced/`, and compares each output with the file of the same name in `results/`. Expected last line: `15 identical, 0 different.`

## Repository layout

| Path | Contents |
|---|---|
| `PROTOCOL_MAIN.md` | Analysis plan of the main study: data, split, training recipe, primary comparison (MEFF vs DSG), statistics and decision rule |
| `PROTOCOL_ABLATION.md` | Analysis plan of the ablation study: comparisons A1–A4, two-way ANOVA and decision rule |
| `code/` | Python scripts |
| `notebooks/` | Kaggle notebooks |
| `split/` | The fixed train, validation and test lists |
| `runs/` | Training history and per-image predictions of all 12 trained models |
| `results/` | All result files used in the paper |
| `MD5SUMS.txt` | MD5 checksum of every file |

### `code/`

| File | Purpose |
|---|---|
| `reproduce.py` | Reproduces all performance results and statistical tests and checks them against `results/` |
| `meff_main.py` | Main study: MEFF and DSG models; commands `verify` (check the images against the split), `train`, `eval` (ensembles and statistics) and `complexity` |
| `meff_ablation.py` | Ablation study: the same models plus DSG + SE and DSG + handcrafted; commands `train` and `ablation` |
| `make_split.py` | Creates the stratified split. The split is fixed and provided in `split/`; this script is included for transparency only |
| `exploratory_single_models.py` | Exploratory comparison of each MEFF expert with the DSG model of the same seed |
| `sensitivity_near_duplicates.py` | Exploratory analysis: all comparisons repeated without the 39 test images that have a near-duplicate in the training set |
| `handcrafted_class_stats.py` | Class means of the handcrafted descriptor on the test images (needs the images) |
| `latency_benchmark.py` | Latency of all four variants, as a single network and as a three-network ensemble (needs the images and a GPU) |
| `make_paper_figures.py` | Result figures of the paper |

### `notebooks/` (Kaggle, GPU T4)

| Notebook | Purpose |
|---|---|
| `train_meff_dsg.ipynb` | Trains one MEFF or DSG model (set `VARIANT` and `SEED`) |
| `train_ablation.ipynb` | Trains one ablation model (`VARIANT` = `dsg_se` or `dsg_hc`) |
| `ensemble_statistics.ipynb` | Builds the ensembles and runs the statistics (CPU only) |
| `bootstrap_analysis.ipynb` | Bootstrap confidence intervals and expert-versus-ensemble comparisons |
| `paper_figures.ipynb` | Result figures |
| `latency.ipynb` | Latency measurement |

Each notebook contains its own copy of the code and checks its MD5 before running.

### `split/`

| File | Contents |
|---|---|
| `train.csv`, `validation.csv`, `test.csv` | Relative path (`CLASS/filename.jpg`), class, label and MD5 hash of each image |
| `SPLIT_MANIFEST.json`, `split_check.txt` | Split method, software versions, class counts per subset and near-duplicate audit |
| `groups.csv` | Perceptual-hash near-duplicate groups (Hamming distance ≤ 8 bits). Used only for the overlap audit and the sensitivity analysis, not for splitting. 39 test and 41 validation images have a near-duplicate in the training set |

### `runs/`

One folder per model: `meff`, `dsg`, `dsg_se`, `dsg_hc`, each with seeds 42, 43 and 44 (for example `runs/meff_s42/`). Each folder has `history.csv` (loss and metrics per epoch), `run_meta.json` (settings and MD5 hashes), `val_predictions.csv` and `test_predictions.csv` (class probabilities for every image) and `DONE`.

### `results/`

| File | Contents |
|---|---|
| `results.csv` | Accuracy, macro-F1, weighted-F1, macro-AUC and per-class metrics of every single model and every ensemble |
| `comparison.csv`, `decision.txt` | Primary comparison MEFF vs DSG and the secondary comparisons: paired bootstrap 95% CIs, exact McNemar tests with Holm correction, decision by the rule in `PROTOCOL_MAIN.md` |
| `confusion_MEFF_hard.csv`, `confusion_DSG_hard.csv` | Test confusion matrices |
| `ensemble_ties.csv` | Number of three-way ties in the hard vote |
| `meff_ensemble_metrics.csv`, `meff_ensemble_metrics_ci95.csv`, `meff_ensemble_vs_single_models.csv` | MEFF metrics with bootstrap CIs; MEFF ensemble vs each expert (from `bootstrap_analysis.ipynb`) |
| `exploratory_single_meff_vs_dsg.csv`, `.txt` | Same-seed comparison of single models (exploratory) |
| `sensitivity_near_duplicates_performance.csv`, `sensitivity_near_duplicates_comparisons.csv` | Results without the 39 near-duplicate test images (exploratory) |
| `handcrafted_class_means.csv`, `handcrafted_class_summary.csv` | Class means of the handcrafted descriptor |
| `complexity.csv`, `timing.csv` | Parameters, FLOPs and the latency printed during training; training and test times |
| `latency_all_variants.csv`, `latency_env.json` | Measured latency (network only and end to end) and the hardware used |
| `eval_meta.json` | MD5 hashes of the main protocol, the code and the six MEFF and DSG model checkpoints |
| `ablation/` | Ablation results, comparisons A1–A4, two-way ANOVA, decision and MD5 hashes |

## Data

The images are not included. DFUC2021 is distributed by the DFU Challenge organisers, and permission to use it must be obtained from them (https://dfu-challenge.github.io; see also https://dfu-2021.grand-challenge.org/Dataset/).

The study uses the 5,955 labelled training images (NONE 2,552, INFECTION 2,555, ISCHAEMIA 227, BOTH 621). Place them in one folder per class (`NONE/`, `INFECTION/`, `ISCHAEMIA/`, `BOTH/`) and check that your copy matches the split exactly:

```
python code/meff_main.py verify --data_dir <DFUC2021_5955> --split_dir split
```

## Retrain the models (GPU)

**On Kaggle** (as in the paper): upload the 5,955 images as a private dataset, import `notebooks/train_meff_dsg.ipynb`, set `VARIANT` (`meff` or `dsg`) and `SEED` (42, 43 or 44) in the configuration cell, and run all cells. For the ablation models use `notebooks/train_ablation.ipynb` with `VARIANT` = `dsg_se` or `dsg_hc`.

**Locally:**

```
python code/meff_main.py train --variant meff --seed 42 --data_dir <DFUC2021_5955> --split_dir split --out_dir runs/meff_s42
python code/meff_ablation.py train --variant dsg_se --seed 42 --data_dir <DFUC2021_5955> --split_dir split --out_dir runs/dsg_se_s42
python code/meff_main.py complexity --out_dir out/complexity
```

Environment of the training runs: Python 3.13.15, PyTorch 2.11.0 (CUDA 12.8), NVIDIA Tesla T4. Packages: torch, torchvision, numpy, pandas, scikit-learn, scikit-image, opencv-python, scipy, pillow, matplotlib.

## Terms

| Term | Meaning |
|---|---|
| Prespecified | Fixed in a protocol file before the models were trained (split, recipe, comparisons, statistics, decision rules) |
| Exploratory | Not in a protocol file; reported in the paper as exploratory (post hoc) |
| Primary comparison | MEFF ensemble vs DSG ensemble, test macro-F1 |
| A1–A4 | The four ablation comparisons defined in `PROTOCOL_ABLATION.md` |

## Internal names inside the frozen files

The protocols, `meff_main.py`, `meff_ablation.py` and the notebooks are kept exactly as they were used, so their text still uses the internal names below.

| Internal name | Meaning in this repository |
|---|---|
| `meff_v3.py` | `code/meff_main.py` |
| `meff_v3_1.py` | `code/meff_ablation.py` |
| `PROTOCOL_v3.md` | `PROTOCOL_MAIN.md` |
| `latency_bench.py` | `code/latency_benchmark.py` |
| `MEFF_v3/` (folder) | The repository root |
| MEFF v3, MEFF v3.1 | MEFF, the method in the paper (the only version) |
| Revision R1 (in `PROTOCOL_MAIN.md`) | The version of the main protocol used for all runs |
| Version A1 (in `PROTOCOL_ABLATION.md`) | A version label of the protocol, not comparison A1 |
| `meff_v3_1_md5` (in `results/latency_env.json`) | MD5 of `code/meff_ablation.py` |

## Integrity checks

- `md5sum -c MD5SUMS.txt` checks every file in this repository.
- MD5 hashes of the frozen files. The code MD5 is also recorded in every `runs/*/run_meta.json`, and the protocol MD5s in `results/eval_meta.json` and `results/ablation/ablation_meta.json`:
  - `code/meff_main.py` d4a5febc73c33620713a457530f100b4
  - `code/meff_ablation.py` 77d87ea56c3e4585e89058a9952154c7
  - `PROTOCOL_MAIN.md` 613e413ea60ecececd5d3d7e28c87b0d
  - `PROTOCOL_ABLATION.md` 9a654f9bb00a86dfce86a6926d24a0aa
- The protocol files use the term "pre-registered". They were not deposited in a public registry, so the paper uses the term "prespecified".
- Model weights are not included (about 92 MB per model). Their MD5 hashes are listed in `results/ablation/ablation_meta.json`.

## License

MIT License (see `LICENSE`).
