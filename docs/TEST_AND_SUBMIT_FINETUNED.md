# Testing and submitting the fine-tuned checkpoint

Run this in the project directory:
`<repo>`

You can run Steps 1-4 right now, on whatever `fno_finetuned_best.pth` is
currently sitting in Kaggle's `/kaggle/working/` (it's already been overwritten
through epoch 7) — that validates the whole download -> pack -> zip -> preflight
pipeline mechanically works, before you need to do it for real once training
actually finishes or plateaus. Re-run Steps 1-4 with the real final file when
you're ready to decide whether to submit.

## 0. Prerequisites

- `data/splits/holdout_v12_3.json` and `holdout_v12_3_stock_scores.json` must already exist
  in this directory (they do — built 22 Aug).
- `$RPDE_TRAIN_REAL` must be reachable (used to re-score
  locally; nothing here needs Kaggle once the checkpoint is downloaded).

## 1. Download from Kaggle

In the Kaggle notebook's **Output** tab (or File Browser), download:

- `fno_finetuned_best.pth`
- `finetune_log.json` (has the full epoch-by-epoch history and the run's own
  FINAL confirmation numbers — useful to compare against Step 2's local read)

Save both into a new local folder:

```
mkdir kaggle_download
```

Put `fno_finetuned_best.pth` at `kaggle_download\fno_finetuned_best.pth`.

## 2. Score the raw checkpoint locally (catches a bad download before anything else)

```
python src/evaluation/score_finetuned_ckpt.py kaggle_download\fno_finetuned_best.pth
```

Compare this script's `final(10)` numbers against `finetune_log.json`'s own
`final_finetuned` entry — they should match closely (same windows, same
metrics, same code path). A large mismatch means something went wrong in the
download or the environment, not the model; stop and diagnose before
continuing.

## 3. Pack to fp16 (required, not optional — the raw checkpoint is fp32, ~403 MB, which alone exceeds the 256 MB submission cap)

```
python "<track-2-starting-kit>\pack_ckpt_fp16.py" kaggle_download\fno_finetuned_best.pth kaggle_download\fno_finetuned_v14_fp16.pth
```

This prints its own round-trip quantization error (typical: ~1e-3 relative,
matches the stock checkpoint's own fp16 pack). Confirm the printed output size
is comfortably under 256 MB (expect ~190-200 MB, matching the stock
`sim_real_fno_fp16.pth`'s size class).

## 4. Score the packed checkpoint (catches a quantization regression)

```
python src/evaluation/score_finetuned_ckpt.py kaggle_download\fno_finetuned_v14_fp16.pth
```

This should read almost identically to Step 2's numbers (fp16 quantization
costs a small fraction of a point, not several). If it's noticeably worse,
something is wrong with the pack step, not the model — do not ship it.

**Decision point.** Look at this run's printed "delta vs the frozen stock
checkpoint on the SAME untouched final(10)" line:
- **Positive delta** (beats stock's rel 96.379 / tke 81.039 / mvpe 97.064 on
  the untouched 10) -> worth a live submission slot. Continue to Step 5.
- **Negative or flat** -> per this campaign's standing rule, local numbers are
  a sanity check, not a selector, but a checkpoint that doesn't even beat its
  own comparison baseline locally is not a live candidate. Do not spend a
  submission slot on it; go back to the training loop instead (longer run,
  different LR, etc.).

## 5. Fork the submission directory and swap in the new checkpoint

```
xcopy /E /I .dist_eta0 .dist_finetuned_v14
del .dist_finetuned_v14\sim_real_fno_fp16.pth
copy kaggle_download\fno_finetuned_v14_fp16.pth .dist_finetuned_v14\sim_real_fno_fp16.pth
```

No code changes needed: `.dist_eta0\submission.py` loads whatever file is
named `sim_real_fno_fp16.pth` next to it (`_CKPT_PATH` is a relative join, not
a hardcoded absolute path), and the architecture is identical — only the
weights differ. Everything else (heteroscedastic SPS bands, batching,
host-sync-safe output, import-time loading) carries over unchanged from the
current best live build.

## 6. Build the zip and verify the payload

```
python src/evaluation/build_fno_zip.py .dist_finetuned_v14 submission_fno_finetuned_v14.zip
```

Must print `PAYLOAD VERIFIED` — confirms `submission.py`, `load_baseline.py`,
the checkpoint (>=200,000,000 bytes), `rpde_baselines/` and `_vendor/` are all
actually inside the zip, and the zip is under 256 MB.

## 7. Preflight

```
python src/evaluation/preflight.py submission_fno_finetuned_v14.zip
```

Must print `RESULT: SAFE TO SUBMIT`. Read TEST 5's `rel`/`tke`/`mvpe`/`sps`
line too — this is preflight's own tiny FINAL-pair check
(`21600_15`/`25425_15`, a DIFFERENT, smaller pool than `data/splits/holdout_v12_3.json`'s
14), so don't be alarmed if the numbers don't exactly match Step 4's — the
pair itself overlaps `data/splits/holdout_v12_3.json`'s FINAL set, so this is still an
honest read, just on fewer windows.

## 8. (Optional) broader local sanity check against the current incumbent

Only if you want one more cross-check before submitting — reuses the same
238-window AoA-15 broad-pool harness as `experiments/early_diagnostics/compare_v11_v12.py`, run sequentially
to avoid the machine's memory limits:

```
python experiments/early_diagnostics/compare_v11_v12.py
```

(Edit that script's two `run_build(...)` lines to point at `.dist_eta0` and
`.dist_finetuned_v14` if you want this specific comparison — it currently
compares `.dist_flatband` and `.dist_hetero_nomod`.) Remember: rel/tke/mvpe
differing here is real signal (the trained model actually changed them,
unlike the pure band-formula builds this script was written for); sps/coverage
differences are the usual local-only caveat.

## 9. Submit

Manual step: upload `submission_fno_finetuned_v14.zip` to Codabench
(competition 17363, Track 1). This is irreversible in the sense that it
spends today's one submission slot — the board keeps your best score
regardless, so there's no downside to trying it once Steps 2-7 all pass, but
don't submit a second candidate today on top of it.

## After it scores live

Paste the returned subscores back. Per this campaign's own repeated finding,
a trained-model change transfers live at an unpredictable fraction of its
local size — so whatever Step 4 showed locally, the live delta is the number
that actually decides whether this branch continues (more epochs, a
different LR, spectral augmentation) or gets closed.
