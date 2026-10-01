# All-experiments push (26 → 27 Sep 2026)

One live slot remains (27 Sep UTC, closes 05:29 IST on 28 Sep). Everything below runs in parallel today; tomorrow the
single best-validated build is submitted (decided with a Jev gate). Jev rates reaching 81.9 at 0.23, a rough judgement
and not a measured probability.

## Kaggle queue (you run these)
Setup cell (every new notebook/session): paste the ROOT `src/training/finetune_fno_kaggle.py` (65,690 bytes; has
`allow_heldout_in_pool`, `residual_last`). For G2 also paste `src/training/finetune_cno_kaggle.py` after it.
Uploads needed as Kaggle dataset files: `data/splits/train_pool_all81.json` (G1), `<data-dir>\sim_transolver.pth` (G3).

### G1 — FNO clean fine-tune on ALL 81 trajectories (~55 min)
```python
run(ckpt="/kaggle/input/models/shashwatsrivastav46/sim-fno-pretrain/pytorch/default/1/sim_fno.pth",
    pool_file="<path to uploaded>/train_pool_all81.json",
    allow_heldout_in_pool=True, fixed_epochs=True, epochs=10, lr=1e-4, time_budget=3, seed=0)
import shutil; shutil.copy("/kaggle/working/fno_finetuned_best.pth", "/kaggle/working/exp1_81_last.pth")
```
Why: adds the 14 trajectories closest to the live flow (including SELECT = AoA 15, Re 21.6–25.4k, exactly the live
regime, where Run 1 was weakest on rel). No honest evaluation is possible (it trains on SELECT/FINAL); its SELECT/FINAL
numbers are training scores.

### G2 — CNO clean fine-tune (~3.5 h budget; second notebook if Kaggle allows two GPU sessions)
```python
import finetune_cno_kaggle as C   # or exec the pasted cell
C.run_cno1(pool_file="/kaggle/input/datasets/shashwatsrivastav46/newdata/train_pool_v12_3.json",
           epochs=8, time_budget=3.5)
import shutil; shutil.copy("/kaggle/working/fno_finetuned_best.pth", "/kaggle/working/cno_67_best.pth")
```
Why: CNO is the other official architecture; never fine-tuned to completion (29 Aug run stalled). Honest (67-pool).
Batch 16 × accum 8 with AMP is the configuration that fits the T4. Expect ~5 s/batch, i.e. only ~5 epochs within budget.

### G3 — Transolver clean fine-tune (optional, after G1)
```python
run(ckpt="<path to uploaded>/sim_transolver.pth",
    pool_file="/kaggle/input/datasets/shashwatsrivastav46/newdata/train_pool_v12_3.json",
    lr=1e-4, epochs=10, time_budget=3, seed=0)
import shutil; shutil.copy("/kaggle/working/fno_finetuned_best.pth", "/kaggle/working/transolver_67_best.pth")
```
Why: the third official architecture; honest. Speed on the T4 unknown; watch the first epoch's ms/batch.

For each run: paste the "CONFIRMATION ON FINAL" block (or the last SELECT lines) into the chat and download the
checkpoint into `<repo>\`.

## Local queue (done here, one at a time: ~1 GB RAM free)
| id | experiment | output |
|---|---|---|
| C1 | Scale-free interval widths (width ∝ each window's own σ) for the Run-1 build and v19pw; scored per regime (SELECT = live-like, FINAL AoA split) | local deltas, and new builds if positive |
| C2 | Official `sim_real_cno` + refitted interval/blend tables → a live-ready build (never scored live) | `submissions/submission_cno.zip`, preflight, independence |
| C3 | FNO + Run-1 ensemble feasibility (value of averaging on SELECT; size: 402 MB > 256 MB, so compression would be needed) | go/no-go only |
| B* | Builds for G1/G2/G3 checkpoints: raw + honestly fitted interval tables (`experiments/clean_fno_build/build_exp1.py <ckpt> <tag>`), regime check, independence, preflight | `submissions/submission_<tag>.zip` |

## Final decision (27 Sep, Jev gate)
Candidates: every build that passes independence + preflight. Evidence used: SELECT (live regime) composite vs v17 live,
the FINAL AoA-15 split, and today's live result of Run 1 (if it scores, it calibrates how honest-local transfers).

## Local results (26 Sep, evening)
| id | result | verdict |
|---|---|---|
| C1 scale-free widths | fully scale-free worse (SELECT over-covered 0.83); h ∝ √(median σ_local) wins every honest comparison by ~+0.1 (SELECT −0.754 vs −0.760; FINAL +0.796 vs +0.668); built `submissions/exp1_sf/submission_exp1_sf.zip` (independence PASS, preflight SAFE) | small gain; use instead of exp1 only if Run 1 beats 79.26 live |
| C2 official CNO | under proxy AT −3.10 behind official FNO + v17pw on SELECT, −2.65 on FINAL; also behind on raw Y; ~3× slower | NO-GO |
| C3 FNO + Run 1 ensemble | official alone beats every blend on SELECT (+0.671 vs +0.424 at 50/50); FINAL best blend +0.06; 402 MB > 256 MB | NO-GO |

Slot rule for 27 Sep: Run 1 live > 79.26 → `submissions/exp1_sf/submission_exp1_sf.zip`; otherwise → `submissions/v19_per_window/submission_v19pw.zip`.
