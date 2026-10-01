# RealPDE 2026 Track 1 (Sim-to-Real): team Slash

Forecasting 20 future frames of real PIV velocity around a NACA4418 airfoil from a 20-frame input window, with
prediction intervals scored by the Safe Prediction Score (SPS).
Author: Shashwat Srivastava (independent researcher).

**Technical report:** [`report/RealPDE2026_Track1_Slash_technical_report.pdf`](report/RealPDE2026_Track1_Slash_technical_report.pdf)

| live development-phase result | final score |
|---|---|
| `exp1_sf`: our clean sim-to-real FNO + per-window interval tables | **79.238** |
| `v17`: the organisers' FNO + our estimator blend and interval tables (pooled bins; report, Disclosure 1) | **79.261** |
| 1st place (benchmark) | 82.151 |

## Repository layout

```
report/          the technical report (PDF), its figures, data and build scripts
submissions/     the submitted inference code and fitted tables, and the compliant per-window rebuilds
src/
  training/      Kaggle fine-tuning scripts (finetune_fno_kaggle.py trained every clean sim-to-real model)
  evaluation/    local scoring on the hold-out splits and the pre-submission gate (preflight.py)
  splits/        construction of the SELECT / FINAL hold-out and the training pool
experiments/     one folder per line of investigation, each cited by the report (see experiments/README.md)
data/splits/     the hold-out and training-pool definitions used throughout
results/         scores, console logs, and the raw live and Kaggle logs (results/live_logs)
docs/            the full campaign record, Kaggle run notes, protocols and a literature note
tools/           the decision gate used for submission-slot choices, with its logged decisions
MANIFEST.sha256  SHA-256 of the checkpoints, packages and files behind the reported numbers
```

## Rebuilding the report

```
pip install -r requirements.txt
python report/scripts/build_all.py              # report from the committed figures (seconds)
python report/scripts/build_all.py --figures    # regenerate all figures first (needs the data and caches below)
```

`build_all.py` runs every figure script, then assembles the text (`report_stage0.py` → `stage1–3.py` →
`build_report.py`). A figure script whose inputs are missing is reported, and its committed PNG is kept.

## Data and external inputs

Large and participant-only files are not in the repository (see `.gitignore`):

- **Real PIV data.** Set `RPDE_TRAIN_REAL` to the folder of the official `train_real` `.h5` files. Local forecast
  caches are rebuilt with `experiments/interval_recipes/cache_preds.py` (SELECT/FINAL) and
  `experiments/robustness/cache_all.py` (all 81 trajectories).
- **Organisers' starting kit.** Place `realpde_t1_starting_kit_v9/` at the repository root. Some early diagnostics
  also read `RPDE_KIT_DIR`, `RPDE_T2_KIT_DIR`, `REALPDEBENCH_DIR` and `RPDEBENCH_CKPTS`.
- **Checkpoints and packages.** Our Run 1 checkpoint (`checkpoints/exp1_67_best.pth`) and the `exp1_sf` package are
  attached to the Zenodo record. Packages that embed the organisers' checkpoint are identified by hash only. Verify
  with `sha256sum -c MANIFEST.sha256`.

## Reproducing the main results

| result (report) | how |
|---|---|
| clean sim-to-real FNO (Table 9) | `src/training/finetune_fno_kaggle.py` on a Kaggle T4; settings in `docs/KAGGLE_RUNS.md` |
| v17 component ablation (Table 5) and swap ablation (Table 6) | `experiments/estimator_blend/controls2.py`, `experiments/robustness/swap_ablation.py` |
| per-window rebuild and independence test | `experiments/per_window_rebuild/`, `submissions/*/test_independence.py` |
| 13 random splits, leave-one-AoA/Re-out, bootstrap intervals (§7, §10) | `experiments/robustness/k9_multi.py`, `loo.py`, `arms_bootstrap.py` |
| live scores | `results/live_logs/` |

The files in `submissions/` and `src/training/finetune_fno_kaggle.py` are byte-identical to what was submitted and
trained, so their comments still use the working-folder names of the campaign (for example `lab/pw` for
`experiments/per_window_rebuild`).
