# PROTOCOL_ABLATION — SE × handcrafted ablation for MEFF v3 (pre-registered)

**Version:** A1 · **Written:** 1 November 2026, **before** any ablation run was started.
**Code:** `meff_v3_1.py` (MD5 recorded by the ablation analysis in `ablation_meta.json`).
**Parent protocol:** `PROTOCOL_v3.md` revision R1 (MD5 613e413ea60ecececd5d3d7e28c87b0d). Everything not stated here follows it unchanged.

## 1. Why this ablation
The main evaluation (1 Nov 2026) found no significant difference between the MEFF ensemble and the Dense-ShuffleGCANet (DSG) ensemble
(Δ macro-F1 +0.23 points, 95% CI −1.47 to +2.00). MEFF differs from DSG by two components: the SE block and the 42-D handcrafted
features. This ablation measures each component's effect separately. It is reported whatever it shows.

## 2. Design (2 × 2, three seeds per cell)
| Variant | SE | Handcrafted 42-D | Runs |
|---|---|---|---|
| `dsg` | no | no | existing (seeds 42, 43, 44) |
| `dsg_se` | yes | no | **new** (seeds 42, 43, 44) |
| `dsg_hc` | no | yes | **new** (seeds 42, 43, 44) |
| `meff` | yes | yes | existing (seeds 42, 43, 44) |

- The six new runs use exactly the v3 training recipe (CFG), data (5,955 images), frozen 60:20:20 split, GPU type (Kaggle Tesla T4) and seeds.
- `meff_v3_1.py` differs from `meff_v3.py` only by the two new variants, the `ablation` command and the complexity table.
  Verified before any run: with the same seed, `meff` and `dsg` are built with bit-identical initial weights in v3 and v3.1.
- The existing `meff` and `dsg` runs (trained with `meff_v3.py`) are reused unchanged. The analysis checks that all 12 runs share the
  same training recipe and the same split files.

## 3. Primary analysis (fixed now)
Hard-vote ensemble of the three seeds per variant (tie → highest mean softmax probability), test macro-F1 on the 1,191 test images.

| ID | Comparison | Question |
|---|---|---|
| A1 | DSG+SE − DSG | effect of SE, without handcrafted features |
| A2 | DSG+HC − DSG | effect of handcrafted features, without SE |
| A3 | MEFF − DSG+SE | effect of handcrafted features, with SE |
| A4 | MEFF − DSG+HC | effect of SE, with handcrafted features |

- 95% CI of Δ macro-F1: paired image-level bootstrap, 2,000 resamples, seed 42.
- McNemar exact test on correct/incorrect, Holm correction over A1–A4.
- **Per comparison:** CI entirely > 0 → "helps"; entirely < 0 → "hurts"; otherwise "no significant difference".
- **Per component:** SE (A1, A4) and handcrafted (A2, A3) are called **helpful** only if *both* of their comparisons say "helps";
  **harmful** only if both say "hurts"; otherwise "no consistent significant effect".

## 4. Secondary analysis (fixed now)
- Two-way ANOVA (factors SE and handcrafted, with interaction) on single-model test macro-F1, 3 seeds per cell, α = 0.05.
- Seed means ± SD, soft-vote ensembles, accuracy, macro-AUC and per-class recall: descriptive only.
- Model complexity (parameters, FLOPs, GPU latency) of all four variants.

## 5. Rules
1. All six new runs are reported, whatever they show. No run is repeated with another seed.
2. A run stopped by the time limit is **resumed** from `last.pth`; a crashed run is restarted with the same seed and code.
3. No additional seeds, variants, metrics or tests are added after results are seen. Any later analysis is labelled exploratory.
4. Hard vote stays primary, soft vote secondary, as in PROTOCOL_v3.
5. **Power limitation, stated in advance:** with 1,191 test images the 95% CI of Δ macro-F1 is about ±1.7 points, so effects smaller
   than about 1.5 points may not be detected. A "no significant difference" verdict will be reported as "no detectable effect at this
   sample size", not as proof of no effect.

## 6. Runs to perform (Kaggle, GPU T4 x2, notebook `MEFF_v3_1_ablation_train.ipynb`)
| Account | Run 1 | Run 2 |
|---|---|---|
| A | dsg_se, seed 42 (RUN_COMPLEXITY = True) | dsg_hc, seed 42 |
| B | dsg_se, seed 43 | dsg_hc, seed 43 |
| C | dsg_se, seed 44 | dsg_hc, seed 44 |

Analysis: `python meff_v3_1.py ablation --runs_dir runs --out_dir ablation_results` on the 12 runs
→ `ablation_results.csv`, `ablation_comparison.csv`, `ablation_anova.csv`, `ablation_decision.txt`, `ablation_meta.json`.
