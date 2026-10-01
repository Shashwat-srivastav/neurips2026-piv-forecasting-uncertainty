# Kaggle runs of 26 Sep 2026 — what was done, why, what came out, and how to read it

Status as of 14:25 IST, 26 Sep. Sections marked **PENDING** get filled in when the result arrives.

---

## 1. Why these runs exist

### 1.1 Where we stood
- Best live score: **79.261** (v17, 25 Sep): rel 94.104 · tke 75.368 · mvpe 93.085 · time 91.130 · SPS 37.287. Rank 87 of 182.
- Top-10 cut-off on 26 Sep: **81.878** (+2.62). Leader: 82.151 (roysegal).
- v17 = the **organisers' official FNO** (`sim_real_ft`, unchanged) + our post-processing (estimator blend, interval
  tables, tke boost).

### 1.2 The diagnosis that led here
1. **Every top-30 team has its own model.** Teams using the same model get bit-identical rel/tke/mvpe, because the
   board is deterministic. On the public leaderboard, all top-30 teams have *unique* accuracy triples, while the
   official FNO's triple (94.168 / 74.026 / 92.836) is shared by 11 teams at ranks 124–164. No bands-only or
   post-processing trick lifts the official FNO into the top tier.
2. **The gap is mostly model accuracy on tke, plus SPS.** Against the top-10 mean we trail by tke −4.8, SPS −7.7,
   mvpe −1.2, rel −0.6. SPS partly follows accuracy, because it contains (1 − pm) terms computed from the same errors.
3. **Our local evaluation of the official FNO is contaminated.** It was fine-tuned on every released real trajectory,
   including our local hold-out (SELECT 4 + FINAL 10 trajectories), so its local scores (96.4 / 81.0 / 97.1 on FINAL)
   are much better than its live ones (94.2 / 74.0 / 92.8). A model that **never saw** those trajectories can be
   evaluated honestly on them.
4. **Unseen models transfer to live roughly at their local level.** The only data point: our numpy DMD went from local
   rel 91.95 to live 92.50 (+0.55), tke −1.04, mvpe −1.86. (n = 1, and the local pool for that number is not
   recorded, so treat it as weak evidence.)
5. **Experiment 1 (23 Aug)** had already fine-tuned the *simulation-only* FNO (`sim_fno.pth`) on 67 real
   trajectories (hold-out excluded), with honest FINAL 94.27 / 78.78 / 95.51. It was never submitted live, and its
   checkpoint was lost, so it had to be retrained.

### 1.3 The bet
Retrain Experiment 1 (Run 1). Also train one variant (Run 2, residual-to-last-frame) that targets the one defect we
measured: the FNO over-smooths at lead 1, where persistence beats it. Evaluate both **honestly** on FINAL, attach
the interval machinery, and submit the best build that beats the live v17 recipe.

---

## 2. Run 1 — plain Experiment 1 recipe (DONE)

### 2.1 Exactly what was run
```python
run(ckpt="/kaggle/input/models/shashwatsrivastav46/sim-fno-pretrain/pytorch/default/1/sim_fno.pth",
    pool_file="/kaggle/input/datasets/shashwatsrivastav46/newdata/train_pool_v12_3.json",
    lr=1e-4, epochs=12, time_budget=3, seed=0)
import shutil; shutil.copy("/kaggle/working/fno_finetuned_best.pth", "/kaggle/working/exp1_67_best.pth")
```

| setting | value | why |
|---|---|---|
| start checkpoint | `sim_fno.pth` (organisers' simulation-only FNO, never saw real data) | a clean start keeps the hold-out honest |
| training data | 67 real trajectories (`data/splits/train_pool_v12_3.json`; hold-out 14 and duplicate `7575_0` excluded) | SELECT/FINAL stay unseen |
| windows | 27,193 training windows (stride 2); SELECT 1,660; FINAL 4,150 | as logged |
| schedule | 1,200 steps/epoch × batch 4 → 4,800 windows/epoch; 12 epochs ≈ 2.1 passes | Exp. 1 plateaued by epoch 10 |
| loss | weighted rel/tke/mvpe: 0.734 / 0.106 / 0.160 (derived from the composite sensitivities at the start errors) | aligns training with the score |
| augmentation | input noise σ = 0.02, no amplitude jitter | same as Exp. 1 |
| precision / hardware | fp32 on a Tesla T4; ~222 ms/batch, ~298 s/epoch; total ~60 min | |
| epoch selection | best SELECT sum (SELECT is unseen, so selecting on it is legitimate) | |

### 2.2 Epoch-by-epoch (SELECT: 4 unseen trajectories, 1,660 windows)
| epoch | train loss | rel | tke | mvpe | sum | note |
|---|---|---|---|---|---|---|
| start | – | 79.035 | 66.838 | 74.243 | 220.116 | `sim_fno` as is |
| 0 | 0.1888 | 92.326 | 77.204 | 94.441 | 263.971 | +43.9 in one epoch |
| 1 | 0.1388 | 92.560 | 77.813 | 94.279 | 264.652 | |
| 2 | 0.1316 | 92.770 | 77.931 | 94.506 | 265.207 | |
| 3 | 0.1275 | 92.950 | 78.177 | 95.045 | 266.173 | |
| 4 | 0.1258 | 92.947 | 78.176 | 94.359 | 265.482 | mvpe dips |
| 5 | 0.1237 | 93.032 | 78.371 | 94.665 | 266.069 | |
| 6 | 0.1215 | 93.150 | 78.284 | 94.880 | 266.313 | |
| 7 | 0.1190 | 93.170 | 78.535 | 94.580 | 266.286 | |
| **8** | 0.1188 | **93.219** | **78.533** | **95.056** | **266.808** | **best, saved** |
| 9 | 0.1177 | 93.220 | 78.429 | 94.821 | 266.470 | |
| 10 | 0.1163 | 93.188 | 78.417 | 94.639 | 266.244 | |
| 11 | 0.1163 | 93.236 | 78.611 | 94.892 | 266.739 | |

**Reading:** almost all the learning happens in epoch 0 (+43.9). Epochs 3–11 move the sum within about ±0.6, a
plateau. Training loss keeps falling (0.119 → 0.116) while SELECT is flat, the usual early sign of fitting the
training trajectories more than the unseen ones. More epochs would not help. rel creeps up slowly; mvpe is the
noisiest metric epoch to epoch (±0.5).

### 2.3 Honest confirmation on FINAL (10 unseen trajectories, touched once)
| model | rel | tke | mvpe | sum |
|---|---|---|---|---|
| `sim_fno` (start) | 79.709 | 66.872 | 75.199 | 221.781 |
| **Run 1 (today)** | **94.170** | **79.289** | **95.542** | **269.001** |
| original Exp. 1 (23 Aug) | 94.27 | 78.78 | 95.51 | 268.56 |
| official FNO **live** (for scale; different data) | 94.168 | 74.026 | 92.836 | 261.03 |
| official FNO on the same FINAL (it had seen these) | 96.379 | 81.039 | 97.064 | 274.48 |
| top-10 mean **live** (for scale) | 94.78 | 79.64 | 94.16 | – |

**Interpretation:**
- **The retrain reproduced Exp. 1** within noise (rel −0.10, tke +0.51, mvpe +0.03), so the recipe is stable.
- **Sim-to-real transfer is large and real:** +14.5 rel, +12.4 tke, +20.3 mvpe over the simulation-only start, all on
  unseen real trajectories.
- **On unseen data, Run 1's accuracy profile resembles the top-10 live profile:** tke 79.3 vs 79.6, rel 94.2 vs
  94.8, mvpe 95.5 vs 94.2. Local FINAL is not the live set, though. It may be easier or harder; see §5.
- **The official model looks better on FINAL (96.4 / 81.0 / 97.1) because it very likely trained on those
  trajectories.** Its live numbers show what it does on unseen data.

---

## 3. Building Run 1 into a submission (DONE except final verification)

### 3.1 First build: v17 post-processing refitted honestly
`python experiments/clean_fno_build/build_exp1.py exp1_67_best.pth exp1`. The steps: pack to fp16 (402.8 → 201.4 MB, round-trip
error 4.6e-4), cache Run 1's forecasts on SELECT/FINAL, refit the blend / interval-centre / interval-width tables
**on SELECT with k = 1** (real targets, no proxy, because Run 1 never saw SELECT), score FINAL once.

| (FINAL, honest) | rel | tke | mvpe | SPS |
|---|---|---|---|---|
| Run 1 raw + v8-style bands | 94.155 | 79.178 | 95.504 | 31.65 |
| Run 1 + full v17 post-processing | 94.334 | **75.941** | 95.517 | 37.10 |

The build's automatic GATE compared this with "official FNO + v17pw under proxy AT" and printed **FAIL (−1.145)**.

### 3.2 Why that GATE was wrong
Proxy AT (Y′ = P + 1.447·(Y − P)) was calibrated to reproduce v17's live **band factor**, not its accuracy. Checked
against v17's actual live result, it over-predicts every accuracy metric:

| v17 recipe | rel | tke | mvpe | SPS |
|---|---|---|---|---|
| proxy AT (used by the gate) | 94.95 | 78.05 | 95.89 | 39.40 |
| **actual live** | **94.10** | **75.37** | **93.09** | **37.29** |

The gate therefore asked Run 1 to beat a bar about 1.4 composite too high. The fair reference is v17's **actual live**
subscores.

### 3.3 The real problem the first build exposed: post-processing hurts this model's tke
| variant (Run 1 model, FINAL honest) | rel | tke | mvpe | SPS | composite vs v17 **live** |
|---|---|---|---|---|---|
| **raw forecast + interval tables (centre + width)** | 94.155 | **79.178** | 95.504 | **37.70** | **+0.666** |
| + prediction blend + tke boost (first build) | 94.334 | 75.941 | 95.517 | 37.10 | +0.296 |
| + prediction blend, no boost | 94.458 | 71.572 | 95.517 | 36.31 | −0.251 |
| raw forecast, intervals centred on the prediction | 94.155 | 79.178 | 95.504 | 33.03 | −0.658 |

(Composite uses the leaderboard-fitted sensitivities: rel .453, tke .087, mvpe .081, SPS .284; time assumed equal
because the architecture is identical.)

**Interpretation:**
- **The prediction blend (toward persistence / window mean) cuts tke by 7.6.** It smooths the forecast's temporal
  fluctuations. That blend helped the official FNO, whose weakness was lead-1 over-smoothing in calm regions. Run 1 does
  not need it: it already carries realistic fluctuation energy (tke 79.2 raw).
- **The tke boost** (scale fluctuations by (E_in/E_pred)^¼, capped at 1.6) was designed for a variance-deficient
  model. Here it only partly repairs the damage the blend does.
- **The interval tables are what help:** moving the interval centre and setting widths from the input's frame
  differences raises SPS from 33.0 (centre = prediction) to 37.7, without touching the forecast.
- **Decision:** the submission is **Run 1 raw forecast + honestly fitted interval-centre and width tables.** No blend,
  no boost. Implemented in `the exp1 build (superseded by submissions/exp1_sf)/` (identity prediction table, `BOOST = False`).

### 3.4 Verification of the final build — DONE (14:30 IST)
- The actual `submission.py` over all 420 FINAL windows reproduces **94.155 / 79.178 / 95.504 / SPS 37.70** exactly.
- **Per-window independence: PASS** (worst |batched − alone| = 8.9e-8).
- **Preflight: SAFE TO SUBMIT** (186.3 MB; the 4 warnings are the usual log cap and stale reference numbers).

### 3.5 The regime check that matters (live = AoA 15, Re ≈ 20.6–25.2k)
Build scored per condition, on trajectories Run 1 never trained on, vs v17's **live** score:

| test set | condition | rel | tke | mvpe | SPS | vs v17 live |
|---|---|---|---|---|---|---|
| FINAL AoA 15 (3 traj) | Re 17.8–20.3k | 94.467 | 79.798 | 96.039 | 39.38 | +1.382 |
| FINAL AoA 10 (4 traj) | Re 21.6–25.4k | 94.230 | 78.100 | 95.691 | 38.57 | +0.871 |
| FINAL AoA 20 (3 traj) | Re 19.1–21.6k | 93.744 | 80.030 | 94.731 | 34.95 | −0.287 |
| FINAL all (10 traj) | mixed | 94.155 | 79.178 | 95.504 | 37.70 | +0.666 |
| **SELECT (4 traj)** | **AoA 15, Re 21.6–25.4k = live regime** | **93.212** | 78.515 | 95.008 | 34.53 | **−0.758** |

**Interpretation.** SELECT is the closest match to the live set (same AoA, same Reynolds range; originally chosen for its
similarity to the live flow). There Run 1 wins clearly on **tke (+3.1)** and **mvpe (+1.9)** but loses on **rel (−0.9)**
and **SPS (−2.8)**, and those carry the most weight. Caveat in both directions: SELECT's SPS is *optimistic*, because the
interval tables were fitted on SELECT. SELECT is only 4 trajectories.

**Most likely live outcome:** somewhere between −0.8 and +1.4 around 79.26. Being in the live regime, SELECT argues for
the lower half. It's a genuine gamble, not an expected gain.

### 3.6 Decision (Jev d020): submit Run 1 today, keep v19pw for 27 Sep
- Jev: submit_exp1 0.65 (confidence 0.47); P(beat 79.26): Run 1 0.42, v19pw 0.43, both near a coin flip.
- Why Run 1 first: the board keeps the best, so the high-variance candidate goes first (upside up to +1.4), and the
  safer v19pw stays as the 27 Sep slot. The live result also measures, for the first time, how an **unseen** model
  transfers. That answers the report's central open question.

## 4. Run 2 — residual-to-last-frame (IN PROGRESS)

### 4.1 What and why
Identical to Run 1, plus `residual_last=True`. The output becomes FNO(x) + g[lead, channel] · (x_last − mean), with
the gate g starting at 0 (so step 0 equals `sim_fno` exactly) and learning at lr 1e-2. This lets the network lean on
the last measured frame, where persistence is known to be strong (leads 1–2). It's honest: same 67-trajectory pool.

### 4.2 Progress so far (SELECT sums; Run 1 in brackets)
| epoch | Run 2 | (Run 1) | learned gate u/v at leads 1 · 2 · 5 · 10 · 20 |
|---|---|---|---|
| 0 | 264.084 | (263.971) | .15/.46 · .13/.41 · .11/.18 · .09/.11 · .10/.05 |
| 1 | 264.716 | (264.652) | .11/.54 · .08/.47 · .05/.21 · .04/.15 · .05/.08 |
| 2 | 265.311 | (265.207) | .09/.55 · .06/.47 · .04/.18 · .03/.13 · .03/.07 |
| 3 | 266.267 | (266.173) | .08/.57 · .06/.49 · .04/.21 · .03/.13 · .03/.08 |

### 4.3 Final result (DONE)
| | best epoch | SELECT sum | FINAL rel | FINAL tke | FINAL mvpe | FINAL sum |
|---|---|---|---|---|---|---|
| Run 1 (plain) | 8 | 266.808 | 94.170 | 79.289 | 95.542 | 269.001 |
| **Run 2 (residual-last)** | 8 | 266.820 | 94.200 | 79.292 | 95.548 | 269.041 |
| difference | – | +0.012 | +0.030 | +0.003 | +0.006 | +0.040 |

Final gate (u/v): lead 1 0.07/0.58 · lead 2 0.04/0.50 · lead 5 0.02/0.19 · lead 10 0.01/0.11 · lead 20 0.01/0.06.

**Interpretation.** The residual path learned something physically sensible: lean on the last measured frame for **v**
at short leads, hardly at all for **u**. But it changed the scores by **+0.04 of 269** on FINAL and +0.01 on SELECT,
well inside run-to-run noise (Run 1 vs the 23 Aug Exp. 1 already differed by ±0.5 per metric). The full FNO already
reconstructs what the last frame offers: its output head can learn the same thing internally. **Run 2 is effectively
a clone of Run 1.** Submitting it would re-measure the same model, not test a new idea, so it's not a good use of the
last slot on its own merits.

## 5. What we can and cannot conclude

**Supported by evidence:**
- Clean-start sim-to-real fine-tuning gives a large, reproducible accuracy gain on unseen real data (§2.3).
- On unseen data, Run 1's accuracy profile resembles the top-10 live profile.
- For this model, the interval tables help and the prediction blend and boost hurt (§3.3).

**Not established (the real uncertainty):**
- **Is local FINAL as hard as the live set?** The official model loses 2–6 points going from local to live, but it
  memorised local. For an unseen model we have one data point (DMD, +0.55 rel). If FINAL is easier than live, Run 1's
  live score will be lower than its local one.
- **Interval widths are absolute,** fitted at Run 1's local error level. If live errors are larger, the bands are too
  tight and SPS falls quickly.
- **Single seed, single run;** no variance estimate.

**Expectation (not a prediction):**
- If local ≈ live, the raw+intervals build lands near **79.9–80.0** (≈ +0.67 over 79.26).
- If live errors are larger, it could fall below 79.26. The board keeps the best score, so the downside is only the slot.
- Top 10 (81.9) is unlikely from this alone.

---

## 6. What to do next (exact)
1. §3.4 passed. **Submit `submissions\submission_exp1.zip` on Codabench now** (26 Sep UTC slot, closes 05:29 IST 27 Sep).
2. On 27 Sep: if Run 1 scored > 79.26 and Run 2 beats Run 1 on SELECT → Run 2 build; otherwise `submissions\submission_v19pw.zip`.
3. When Run 2 finishes: paste its "CONFIRMATION ON FINAL" block, download `exp1_67_resid_best.pth`, and it gets built
   the same way (raw + interval tables) for the 27 Sep slot, if it beats Run 1.
4. Paste every live score into `results/live_logs/`. The report is updated with the live result.

## 7. Files
| file | what |
|---|---|
| `exp1_67_best.pth` | Run 1 checkpoint (fp32, 402.8 MB) |
| `results/live_logs/kaggle_run1_exp1.txt` | Run 1 summary |
| `experiments/clean_fno_build/build_exp1.py`, `build_exp1_out.txt` | first build + (flawed) gate output |
| `experiments/clean_fno_build/variants.py` | the §3.3 variant comparison |
| `experiments/clean_fno_build/verify_build.py`, `verify_out.txt`, `preflight_exp1.txt` | final-build verification (§3.4) |
| `the exp1 build (superseded by submissions/exp1_sf)/` | the submission build (raw + interval tables, `BOOST = False`) |
| working plan (not deposited) | the overall plan and slot rules |
