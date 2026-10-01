# RealPDE Track 1 — complete campaign record

NeurIPS 2026 RealPDE, Track 1 (Sim2Real), Codabench 17363, participant "Slash".
Predict 20 future PIV frames (u, v; p unmeasured) from a 20-frame input window.
Development leaderboard closes **27 Sep 2026**; Decision Phase 28 Sep – 25 Oct
(top 10 by dev score retrain on private test); code review 25 Nov.

Compiled **13 Aug 2026**. Every figure is measured unless tagged `[PROJECTED]`.

| tag | meaning |
|---|---|
| `[LIVE]` | returned by the scoring server |
| `[LOCAL]` | measured offline on held-out trajectories |
| `[PROJECTED]` | inferred, not measured |
| `[REFUTED]` | tested and killed |

---

## 1. Standing

| | value |
|---|---|
| **Best live** | **78.140964** — `submission_fno_eta0_v8.zip`, 13 Aug |
| Leader (13 Aug) | 81.15749 — iapetos1918 |
| Rank 12 | 80.58052 — amrsaid |
| **Gap to rank 12** | **−2.440** |
| Submissions made | 10 (one scored 0.000000) |

**The best build contains nothing we trained.** It is the organizers' `sim_real_ft`
FNO, loaded at import, plus heteroscedastic SPS bands — no corrector, no
fine-tune. Every model we fitted this season is measurably worse.

Best-live subscore breakdown:

| subscore | value | derived |
|---|---|---|
| rel_l2 | 94.168148 | rel-L2 error 0.12386 |
| tke | 74.025874 | error 0.70175 |
| mvpe | 92.836233 | error 0.15433 |
| time | 91.408779 | 6.40 ms/sample |
| sps | 33.337914 | W̄ 0.6784, T 0.4914 |

*(The v4 breakdown these replaced — 94.019858 / 75.724990 / 92.806641 /
91.405230 / 32.549227 — is submission #7 and is retained in the ledger below.)*

### Leaderboard trajectory

| date | leader | rank 10-13 | us | gap |
|---|---|---|---|---|
| 8 Aug | 80.400 | 79.831 (r10) | 78.033 | −1.80 |
| 10 Aug | 80.792 | 80.027 (r13) | 78.033 | −1.99 |
| 13 Aug | 81.157 | 80.581 (r12) | 78.033 | −2.55 |

**The gap is widening.** Returning entrants gained a median **+0.390** between
8 and 10 Aug alone, and four new names arrived straight in above 80.

---

## 2. Scoring system

```
score_error(e) = 100 / (1 + 0.5·e)              -> rel_l2, tke, mvpe
score_time(t)  = 100 / (1 + sqrt(t / 0.72896))  -> t in seconds per sample
score_sps(w)   = 100 · clip(w, 0, 1)
                 w = 0.5·dm + 0.3·tke + 0.2·mvpe
                 branch = mean over scored elems of (1−pm)·exp(−(u−l)/σ)·1{covered}
                 pm = err/(0.5+err),  σ = 0.0563870259  (FROZEN)
                 scored = target != 0.0

tke  = per-pixel TEMPORAL VARIANCE of u,v over the 20 output frames, rel-L2 on that map
mvpe = 4 probe columns; TIME-AVERAGE first, separate rel-L2 per column, then average
```

Only scorer in the tree: `realpde_t1_starting_kit_v9/.../scoring.py`. Any pre-v9
sps figure used a **logistic** SPS (`100/(1+exp(−w))`, floor 50) and must be
converted via `w = −ln(100/s − 1)` then `100w` before comparison.

### The composite cannot be predicted

Unpublished, and demonstrably **not** the mean. Three live points give three
different ratios of Δfinal to Δ(subscore sum):

| transition | Δsum | Δfinal | ratio |
|---|---|---|---|
| v1 → v4 | +1.690 | +0.129 | 0.076 |
| v4 → v6 | −1.257 | −0.287 | 0.228 |

Five unknowns, two independent equations. **Standing rule: never compute an
expected `final_score`.** All gains in this document are quoted as subscores.

### The SPS factorisation

All three SPS branches share the same `inside` mask and the same `exp(−nil)`
term; only the per-sample scalar `pm` differs. So exactly:

```
sps  = 100 · W̄ · T
W̄    = 0.5(1−pm_dm) + 0.3(1−pm_tke) + 0.2(1−pm_mvpe)     accuracy only
T    = mean over scored elems of exp(−2h/σ)·1{|e| ≤ h}     bands only
```

Maximising `T` gives the **reversed-hazard condition** `f(h)/F(h) = 2/σ = 35.469`.
Measured optimal coverage is **0.70–0.80**, not 0.95. Over-coverage is punished
exponentially, under-coverage only linearly.

Live W̄ values: 0.6584 (numpy DMD), 0.6817 (FNO η=0.4), 0.6827 (FNO η=1.4).
Local W̄ on the FINAL pool: **0.7690** — do not mix the two.

---

## 3. Complete live submission ledger

| # | build | rel | tke | mvpe | time | sps | **final** | W̄ | T | ms/sample |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | numpy DMD 0.006/0.005/1.75 + heavy logging | — | — | — | 90.655 | 30.721 | 76.6845 | — | — | 7.75 |
| 2 | numpy DMD 0.007/0.006/**0.85** | — | — | — | — | 30.831 | — | — | — | — |
| 3 | numpy DMD **0.004/0.003/1.5** (incumbent) | 92.501 | 75.129 | 92.629 | 90.866 | 31.879 | **77.0117** | 0.6584 | 0.4842 | 7.37 |
| 4 | + log-POD denoise, lazy decode in `predict()` | — | — | — | — | — | 76.4927 | — | — | — |
| 5 | stock FNO, **no bands** (default ±5%) | 94.168 | 74.026 | 92.836 | — | 14.080 | 72.6542 | 0.6784 | 0.2076 | — |
| 6 | FNO + corrector η=0.4 + bands | 94.189 | 74.770 | 92.870 | 90.590 | 32.397 | **77.9045** | 0.6817 | 0.4752 | 7.87 |
| 7 | **v4** FNO η=1.4 + rank band modifier | 94.020 | 75.725 | 92.807 | 91.405 | 32.549 | **78.0331** | 0.6827 | 0.4768 | 6.45 |
| 8 | **v5** + manual CUDA graphs | 0 | 0 | 0 | 0 | 0 | **0.0000** | — | — | — |
| 9 | **v6** + band ×0.85 + fp16 bounds | 94.020 | 75.725 | 92.807 | 91.139 | 31.559 | 77.7459 | 0.6827 | 0.4623 | 6.89 |

### What each submission established

**#1–#3, the band bracket.** Three live points on identical fields (bands do not
change fields, so W̄ is constant and T ∝ sps). At live σ_local(u) = 0.016263:

| A_u | K | mean h_u | sps | T |
|---|---|---|---|---|
| 0.007 | 0.85 | 0.02082 | 30.831 | 0.4683 |
| 0.004 | 1.50 | 0.02840 | **31.879** | **0.4842** |
| 0.006 | 1.75 | 0.03446 | 30.721 | 0.4666 |

The middle setting is the peak. Band scale for the DMD is **closed empirically
from the board itself**.

**#4, the methodology lesson.** log-POD variance denoise had every safeguard — a
regime-matched pool tracking the board to 0.816 mean absolute gap, nested
TRAIN/SELECT/FINAL, rank chosen by leave-one-trajectory-out. Local +2.73, live
**+0.03**. It also shipped with a lazy POD-basis decode inside `predict()`,
costing **−0.552** on time.

**#5, the band bug.** Shipping without `lower`/`upper` falls back to a default
±5% interval. T collapsed to **0.2076**. Adding the heteroscedastic bands to
identical fields moved sps 14.08 → 32.40, i.e. **+18.3**. Largest single gain of
the campaign.

**#6 → #7, the corrector.** η 0.4 → 1.4 delivered live tke **+0.955** (predicted
0 to +1.86 at the 87% incremental-transfer rate — the only prediction that landed
inside its stated range). rel −0.169, mvpe −0.064 as expected.

**#8, the zero.** `torch.cuda.graph()` puts the CUDA RNG generator into capture
mode; a `torch.randn` after the capture block left it inconsistent and raised
`RuntimeError: CUDA generator expects graph capture to be underway` during
**import**. `_load_at_import()` had no outer guard, the process died before
serving a batch, and every metric scored 0.000000. Root causes: an explicitly
untestable feature shipped alongside four others, and per-function `try/except`
where a top-level guard was required.

**#9, two failed bets.** Band ×0.85 cost **−0.990 sps**; fp16 bounds cost
**−0.267 time**. Both are analysed in §6.

---

## 4. The container and harness — everything measured

| fact | value | source |
|---|---|---|
| GPU | NVIDIA A800-SXM4-80GB | every ingestion log |
| torch / numpy | 2.2.2 / 1.26.4 | submission log |
| processes | **exactly 17** | v1 log, untruncated |
| calls per process | 2 × n=5 | atexit timer |
| scored samples | ~90 (call 1 only) | see below |
| checkpoint | 201,396,349 bytes, loads 0.73–1.38 s | log |
| submission limit | 256 MB, enforced pre-run | ingestion |
| participant stderr cap | **10,240 bytes shown** | truncation notice |
| env | 3 gateway variables stripped before the run | ingestion |

**Import is untimed.** Proven: numpy import alone costs 190 ms/process, a 38
ms/sample floor if charged; the board charged 7.7. Model construction, checkpoint
load and warm-up are all free. This single fact unlocked checkpoints and the GPU
after weeks of wrongly excluding them.

**Call 0 is not charged.** v4 had call 0 at 138.60 ms and call 1 at 17.62; v6 cut
call 0 to 16.5 ms and the board charged **more**. So the ~117 ms of first-touch
kernel loading was never costing anything.

**Call 0 is synthetic.** Gaussian matched to the data's mean/std, p1/p99 at
±2.3263σ to within 0.002, `zerofrac = 0.0000`, `tau = 0.500`, `n_eff = 20.0`,
σ_local 11.6× local. Unrepresentative by construction.

**No GPU contention.** Scored-call time / import-benchmark time = **1.00–1.07**
across all 17 processes, variance under 2%. Seventeen processes sharing one A800
makes no measurable difference. Hypothesis refuted.

**~2.9–3.6 ms/sample is charged outside `predict()`.**

| build | predict() | board | outside |
|---|---|---|---|
| v4 | 3.523 | 6.445 | **2.922** |
| v6 | 3.284 | 6.891 | **3.607** |

It is **not fixed** — it grew 0.685 when we returned fp16 bounds, because the
scorer does `np.asarray(lower, dtype=np.float32)` and the harness pays to widen
them. The inter-call hand-off measures **19–39 ms, mean ~27 ms** per process.

**`torch.compile` is unavailable.** `PermissionError [Errno 13]` in all 17
processes even after redirecting `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`,
`TORCHINDUCTOR_FX_GRAPH_CACHE` and `XDG_CACHE_HOME` to a directory verified
writable by an actual write probe (`cache=y` in the log). The block is not the
cache location. **Closed.**

**TF32 is unsafe on this model.** Deviation against fp32 measured 3.09e-03 to
6.09e-03 across processes — 15–30× the 2e-4 gate — and auto-reverted every time.
Implication beyond TF32: the spectral weights are numerically brittle, so bf16
and int8 inference are unlikely to be accuracy-neutral either.

**fp16 is blocked for the model regardless**: it is spectral, and cuFFT requires
power-of-two signal dimensions for half precision. Ours are (26, 38, 70).

---

## 5. The evaluation set

Established entirely from per-process ingestion statistics printed by our own
submission. No evaluator state, no hidden targets.

| | |
|---|---|
| flow condition | a **single** condition, AoA 15 |
| Reynolds | 20,600–25,200 (`u_std` is the only Re-monotone statistic) |
| grid | 32×64, 3 channels, p identically zero |
| masked cells | 9–11%, exactly zero in the input |

Live fingerprint, mean over the visible worker processes, call 2 only:

| statistic | live | local ref | ratio |
|---|---|---|---|
| u_mean | 0.21810 | 0.15194 | 1.44 |
| u_std | 0.15039 | 0.10972 | 1.37 |
| v_std | 0.02323 | 0.01506 | 1.54 |
| σ_local u | 0.016263 | 0.00809 | **1.94** |
| σ_local v | 0.007310 | 0.00480 | 1.52 |
| framediff | 0.014485 | 0.00782 | 1.85 |
| τ | 2.110 (1.514–2.721) | 2.345 | 0.90 |
| n_eff | 4.95 (3.68–6.60) | 4.264 | 1.16 |
| zerofrac | 0.0900 | 0.0938 | 0.96 |

**Live σ_local distribution** (u): p10 0.00058, p50 0.00747, p90 0.04030,
mean 0.01571. **p90/p10 = 70×**, mean 2.1× the median. Heavily right-skewed —
under the shipped band the median element gets h = 0.0152 while a p90 element
gets h = 0.0645, where `exp(−2h/σ) = 0.10` and the element contributes almost
nothing whatever we do.

**Live rel-L2 error is 1.5× local. No combination of error and band inflation
reproduces the board's measured T = 0.4752** (every simulated scenario lands at
0.41 or below while live sits above our local 0.4653). The uniform-scaling model
of the shift is refuted; we cannot simulate the live regime and do not try.

### Defects found in the released data

- **`6300_0` and `7575_0` are byte-identical.** Reported to the organizers; they
  published a data notice on 2026-08-12 confirming `7575_0` reuses `6300_0` and
  must be excluded from training, crediting "the team who inspected the data".
- **78 of 82 files carry an `re` attribute disagreeing with their filename.**
  True step is 1278–1279, not 1275. Real sequence: 3750, 5028, 6306, 7585, 8863,
  10142, 11420, 12698, 13977, 16534, 17812, 19090, 20369, 21647, 22926, 24204,
  25482, 26761. The long-quoted "Re 15225" for the eval set is wrong twice over.

---

## 6. Experiment register

### 6.1 Bands and SPS

| method | effect | status |
|---|---|---|
| heteroscedastic `h = A + K·σ_local` | **+18.3** sps | `[LIVE]` shipping |
| band scale bracket (3 live points) | peak at 0.004/0.003/1.5 | `[LIVE]` closed |
| rank-based per-element modifier | +6.96 local / **+0.152 live** | `[LIVE]` marginal |
| uniform ×0.85 rescale | **−0.990 live** | `[REFUTED]` |
| fixed-width bands | −2.76 | `[REFUTED]` |
| bias shift (DMD is unbiased) | ~0.00 | `[REFUTED]` |
| per-group constants | +0.18 | marginal |
| per-lead-time widths | +0.13 | marginal |
| James–Stein variance shrinkage | ≈0 | `[REFUTED]` collapses to the existing blur at true n_eff ≈ 2.9 |
| per-AoA damping | +0.000 | `[REFUTED]` |
| per-pixel conformal calibration | — | part of the ~+1.85 envelope |
| learned difficulty regression | — | part of the envelope |
| spatial smoothing, noise augmentation | — | part of the envelope |
| joint 2-D/3-D/4-D lookups | — | part of the envelope |
| temporal prediction disagreement probe | Spearman 0.364 but only +0.0101 T, against −1.47 time | `[REFUTED]` net negative |
| Adaptive Conformal Inference | targets coverage, not T; ~90 samples in batches of 5 | `[REFUTED]` on analysis |

**The three live FNO band points:**

| band | sps |
|---|---|
| ×1.00, no modifier | 32.397 |
| ×1.00 + rank modifier (median .70) | **32.549** |
| ×0.85 uniform | 31.559 |

Uniform narrowing hurts; the per-element modifier helps slightly. **The scale was
already correct**, and the inference that we were "40% too wide" — rescaling the
DMD's live optimum by the rel_l2 error ratio — was invalid.

**Band frontier, measured on 84 held-out windows, 6,144,971 scored elements:**

| rule | T | sps at local W̄ |
|---|---|---|
| shipped `A + K·σ_local` | 0.4653 | 35.78 |
| best flat constant `h = 0.009` | 0.5217 | 40.12 |
| ceiling, σ_local alone | 0.5417 | 41.66 |
| ceiling, \|prediction\| alone | 0.5535 | 42.56 |
| ceiling, spectral energy ratio alone | 0.5551 | 42.69 |
| **implementable ceiling, all three jointly** | **0.5663** | **43.55** |
| oracle `h = |e|` (needs the target) | 0.7851 | 60.37 |

The implementable ceiling uses bin edges from input-valid cells only, per-bin `h`
fitted on SELECT, evaluated on FINAL — no target information at test time. The
naive in-sample version is 0.5690, so the honest bound is only 0.0027 lower.

**Feature correlations with per-element abs(e):**

| feature | Pearson | Spearman |
|---|---|---|
| σ_local | +0.482 | **+0.520** |
| \|prediction\| | +0.322 | +0.513 |
| input spectral energy ratio | −0.450 | −0.498 |
| temporal prediction disagreement | +0.250 | +0.364 |
| lead time | +0.050 | ~0 |

**Payoff curve** (synthetic features at controlled rank-correlation):

| Spearman | T | sps | vs flat constant |
|---|---|---|---|
| 0.00 | 0.5217 | 40.12 | +0.00 |
| 0.37 | 0.5457 | 41.96 | +1.84 |
| 0.52 | 0.5596 | 43.03 | +2.91 |
| 0.70 | 0.5842 | 44.92 | +4.81 |
| 0.80 | 0.6028 | 46.35 | +6.23 |
| 0.90 | 0.6421 | 49.38 | +9.26 |

Strongly convex — 0.52 → 0.60 buys +0.6 sps, which is noise. And the curve is an
**upper bound**: real features of equal rank-correlation deliver only 0.53×,
0.85× and 0.93× the synthetic gain (replicated on both pools). A real signal at
Spearman 0.90 is worth roughly +3.0 to +6.1 over what we ship.

**Error distribution** (6,144,971 scored elements): p10 0.0006, p25 0.0015,
p50 0.0040, p75 0.0103, p90 0.0231, p99 0.0637, mean 0.00884. **p99/p50 = 16×.**

### 6.2 TKE denoising — 14 variants

| variant | local Δ | status |
|---|---|---|
| log-POD rank 4 | +2.31 | best of family — `[REFUTED]` live (+0.03) |
| reliability-weighted POD inversion | +2.1 | close second |
| vash mixture priors | +0.72 | weak |
| SPOD objective thresholding | negative | `[REFUTED]` |
| velocity-field truncation | negative | `[REFUTED]` |
| per-pixel n_eff shrinkage | negative | `[REFUTED]` |
| tensor-basis invariants | **−6.62** | `[REFUTED]` |
| ensemble-Kalman inversion | negative | `[REFUTED]` |

**TKE's own ceiling is ~80, not the ~92 once assumed.** The target is itself a
20-sample variance estimate: split-half disagreement rel-L2 0.84–0.89, and it
differs from a long-window estimate of the same field by rel-L2 ≈0.50. A
predictor knowing the converged field exactly scores 79–81.

The −6.62 on invariants corroborates the physics: RMS of ∂u/∂x + ∂v/∂y is
**1.49×** the individual terms, because the PIV plane is a 2-D slice of a 3-D
flow and is **not** divergence-free.

### 6.3 Models

| model | result | status |
|---|---|---|
| numpy DMD rank-6 + variance top-up | 77.012 live; 13 KB, nothing learned | held best for weeks |
| stock FNO (`sim_real_ft`) | rel 96.06 / tke 80.29 / mvpe 97.20 local | **adopted** |
| FNO + operator boosting, η grid to 3.0 | η=1.4 best local, η=0.4 then 1.4 shipped | `[LIVE]` |
| CNO | rel 95.95 / tke 78.77 / mvpe 96.50 | `[REFUTED]` **superseded** — FNO wins all three; also blocked by `dnnlib` → `import requests` |
| Transolver | tke 70.69 | `[REFUTED]` below the incumbent's 75.13 |
| from-scratch 9.45M invariant model | 266.0 vs FNO 273.6 | `[REFUTED]` |
| PCA+ridge distillation of CNO | R² 0.42, 11% of the MVPE gain | `[REFUTED]` |
| FNO fine-tuned on `train_real` | net **−1.86** | `[REFUTED]` — **see §9, diagnosis was never done** |
| FNO ⊕ DMD ensemble | −1.98 at even 10% DMD weight, monotone | `[REFUTED]` |

**The CNO correction.** The often-quoted "CNO beats us by +3.60 rel / +1.73 mvpe"
was measured against the *numpy DMD incumbent* we later dropped. Against the FNO
it loses on all three metrics.

**The FNO architecture, measured from the checkpoint:**

| component | parameters |
|---|---|
| **total** | **50,357,955** |
| spectral convolutions | 50,331,648 — **99.9%** |
| `fc0` lift, Linear 7→64 | 448 |
| pointwise `Conv3d` 1×1×1 ×4 | 16,640 |
| `BatchNorm3d` ×4 | 512 |
| `fc1`, Linear 64→128 | 8,320 |
| `fc2`, Linear 128→3 | 387 |

modes (4, 12, 16), width 64, 4 layers, domain padding 6 → FFT sizes (26, 38, 70).
**No dropout layers anywhere.** At 4 bytes/param the checkpoint is 201.4 MB
against a 256 MB limit, so **a second ensemble member cannot fit** — deep
ensembles are ruled out by storage, not by the time budget.

### 6.4 Accuracy experiments, 8 Aug

| lever | result | why it failed |
|---|---|---|
| anchor predicted time-mean to input time-mean | −0.1 to −4.2, monotone | the FNO's predicted mean is already **2.1× more accurate** than the input's (0.00409 vs 0.00912). tke was *exactly* invariant at every α, confirming the mechanism is real even though this estimator isn't |
| retune FFT padding to 7-smooth sizes | p=8 costs **−9.78** | changing the padded length changes which physical frequency each retained Fourier mode represents; the model is welded to padding 6 |
| ensemble with the DMD | −1.98 at 10% weight | error fields correlate at only +0.53, but the DMD is simply too much worse |

Padding results in full (SELECT, 84 windows):

| pad | FFT dims | 7-smooth | Δ sum3 |
|---|---|---|---|
| 6 | (26,38,70) | no | — |
| 4 | (24,36,68) | no | −7.453 |
| 7 | (27,39,71) | no | −3.965 |
| 8 | (28,40,72) | **yes** | −9.780 |
| 10 | (30,42,74) | no | −18.167 |
| 12 | (32,44,76) | no | −22.638 |

### 6.5 Distribution-shift diagnostics

**The generalisation map.** Frozen FNO scored on all 81 training trajectories,
10 windows each:

| metric | min | p5 | median | max | LIVE | verdict |
|---|---|---|---|---|---|---|
| rel_l2 | 93.46 | 94.17 | 95.97 | 97.91 | 94.17 | at the 5th percentile |
| tke | 68.62 | 70.01 | 76.82 | 88.21 | 74.03 | inside the spread |
| mvpe | 93.34 | 94.41 | 96.45 | 98.74 | **92.84** | **below all 81** |

Live mvpe is worse than every trajectory the model was fitted on, with **no decay
against distance** from the live fingerprint.

**Causes ruled out by direct experiment:**

| hypothesis | test | result |
|---|---|---|
| data-prep mismatch (2×2 block mean vs `::2`) | `test_blockmean.py` | `[REFUTED]` no closer |
| evaluator crops toward the wake | `find_preprocessing.py`, every 32×64 crop on a stride-4 grid | `[REFUTED]` |
| input covariate shift in scale | `test_tta_norm.py`, rescale local to the live fingerprint | `[REFUTED]` explains 3% |
| stale normaliser statistics | per-window test-time normalisation | `[REFUTED]` hurts by 12.5 |

**Transfer record.** Changes validated offline then shipped:

| change | local | live | transfer |
|---|---|---|---|
| conditional-TKE map | +3.25 | **−1.02** | −31% |
| whole-model swap to the FNO | +8.05 | +0.561 | 7% |
| log-POD variance denoise | +2.73 | +0.03 | 1% |
| rank band modifier | +6.96 | +0.152 | 2% |

Incremental changes to the *same* model transfer at roughly +87%; model-level
changes do not transfer at all. **Matching levels does not imply matching deltas.**

---

## 7. Bugs found and fixed

| bug | cost | resolution |
|---|---|---|
| shipped with no `lower`/`upper` → default ±5% bands | ≈−18.3 sps | heteroscedastic bands |
| lazy POD-basis decode inside `predict()` | −0.552 time | moved to import (untimed) |
| variance top-up left inside the `elif` branch | A/B measured the wrong thing | restructured to nested if/else |
| input normaliser statistics used for **both** input and target | unmeasured | `_MEAN_TGT != _MEAN_IN`; fixed |
| `autocast` + FNO → cuFFT crash | build failure | fp16 FFT needs power-of-two dims; forced fp32 |
| mvpe implemented without time-averaging first | 0.783 vs true 0.202 | time-average, then per-column rel-L2 |
| EnKF `C_gg` at 200000×200000 | 320 GB OOM | reduced to 2000 observations |
| `.h5` suffix in npz `names`; missing `lengths` | silent | padded trajectories were training on zeros |
| `prediction ± half` broadcast on the host | 2.088 ms/call | computed on device |
| `concatenate([1 item]).astype(float32)` | 1.479 ms/call | single-chunk shortcut |
| `np.isfinite(x).all()` host scan every call | 0.275 ms/call | device-side `nan_to_num`; the comment claiming it was conditional was false |
| normaliser tensors re-uploaded every call | 4 transfers/call | hoisted to import |
| TF32 enabled but verified only against itself | correctness hole | verified against a true fp32 reference; auto-reverts |
| **`_BATCH = 16` while every probe used 5** | would have padded 5→16: 3.2× compute **and** a CUDA-graph recompile inside the timed region | one `_SHAPE` derived from `_BATCH` |
| in-place `nan_to_num_` aliased the caller's numpy buffer | silent mutation of the harness's input | out-of-place |
| dropped `.float()` guard | fp16 could reach the scorer | restored (free no-op on float32) |
| **no top-level guard on `_load_at_import()`** | **scored 0.000000** | hard guard + `_minimal_load()` fallback, tested by injection |
| logging 1,370 B/process against a 10,240 B cap | 9 of 17 processes invisible | compacted to ~180 B/process |

---

## 8. Tooling

| file | purpose |
|---|---|
| `src/evaluation/preflight.py` | **pre-submission gate.** Fresh-subprocess import, API contract, ragged batches, dirty input, caller-mutation, log budget, guard-injection test, real scorer. Exit 0 = safe to submit |
| `src/training/finetune_fno_kaggle.py` | GPU fine-tuning; metrics match `scoring.py` to 1e-7 |
| `src/training/boost_operator_kaggle.py` | `TinyOperator`, 668,316 params, zero-init proj2 — the corrector's only source |
| `experiments/early_diagnostics/generalisation_map.py` | frozen-FNO scores over all 81 trajectories |
| `src/splits/make_matched_splits.py` | regenerates the regime-matched pool |
| `src/evaluation/build_fno_zip.py` | packaging |
| `submission.py` | numpy-DMD rollback source |

Result JSONs retained: `boosted_scores.json`, `finetuned_scores.json`,
`generalisation_map.json`, `data/splits/matched_splits.json`, `eda_profile.json`.

**Use `data/splits/matched_splits.json`, not `v9_splits.json`** — the latter overstates SPS
by +8.458 against the board.

---

## 9. What is open — and the direction that was wrongly closed

### The competition changed under us

> **2026-08-09: Data augmentation clarified. Standard augmentation of the
> released data is allowed in both tracks. Every training sample must trace back
> to the released training data.**

**On 9 August, ten-plus competitors submitted and every one landed 80.0–80.5.**
Returning entrants gained a median +0.390 that day; four new names entered above
80. The top has climbed to 81.157 since.

### Our fine-tune was under-determined and we never noticed

`src/training/finetune_fno_kaggle.py` used `stride=20`:

```
windows at stride 20 :   3,362
windows at stride  1 :  66,748    <- 20x more
```

**A 50,357,955-parameter model was fine-tuned on 3,362 windows.** The result —
tke +2.66, rel −0.89, mvpe −0.62, sps −2.01 — is the textbook signature of
overfitting to `train_real`, not a capability ceiling. Overlapping windows are not
even augmentation; they are simply using the released data properly, and they
trivially satisfy the "must trace back to the released training data" rule.

**This was recorded as a closed direction for a week.** The phrase "the accuracy
side is measured out" appears throughout the earlier version of this report and
in every revision of the research brief. It rested on two undiagnosed null
results.

### The composite weights, approximately solved (13 Aug)

Five complete live points fit `final = Σ wᵢsᵢ` by least squares:

| rel | tke | mvpe | time | sps |
|---|---|---|---|---|
| **0.476** | **0.086** | 0.095 | 0.110 | **0.243** |

In-sample residual max **0.0088**. Leave-one-out predicts the four FNO-family
points to **±0.10** but the distant DMD point to **−3.74**, and `cond(S)=1107`.
**Use for ranking neighbouring candidates only** — the standing rule against
quoting an expected `final_score` stands.

Confirmed by direct experiment, not just fit: v8 surrendered **1.699 tke** and
the composite **rose +0.108**, a sign equal weighting cannot produce. **rel is
worth ~5.5× tke, sps ~2.8× tke.** The season was spent optimising the cheapest
metric on the board.

### Sized remaining headroom — restated 13 Aug

| lever | now | status | composite |
|---|---|---|---|
| rel / mvpe | 94.168 / 92.836 | **untrainable, proven above** | **0** |
| tke | 74.026 | priced at 0.086/pt, unbuyable without paying rel | ~0 |
| amplitude (η, λ) | native | **closed both sides** — η=0 and λ=1 are optimal | 0 |
| band **scale** | A+K·σ_local | bracketed live both sides (0.0208/0.0276/0.0335 on DMD; ×0.85 lost 0.99 on FNO) | 0 |
| band **shape**, 3-feature | T=0.4914 live | **CLOSED 13 Aug — it is a one-feature effect already shipping.** See below | **~0** |
| time | 91.409 | predict() 3.529 ms of a 6.4 ms bill; predict()→0 caps at +2.68 subscore | **≤ +0.295** |

### Band shape, closed 13 Aug — and §6.1's framing corrected

Nested TRAIN/SELECT/FINAL over 8 AoA-15 trajectories, features computed from the
input and our own prediction only, **ranks taken within-call** so no fitted local
scale can transfer.

| rule | T_TRAIN | T_SELECT | T_FINAL |
|---|---|---|---|
| base `A+K·σ_local`, no modifier | 0.4933 | 0.4775 | 0.4646 |
| **σ_local rank alone** | 0.5747 | 0.5616 | **0.5552** (+19.50%) |
| + spectral energy ratio | 0.5755 | 0.5623 | 0.5560 |
| **3-feature joint** | 0.5770 | 0.5639 | **0.5576** (+20.01%) |

**That +19.50% is the modifier we already ship.** §6.1 records the rank modifier
at +6.96 sps local, which at local W̄ = 0.7690 is T 0.4653 → 0.5558 = **+19.4%** —
the same number — and it went live at **+0.152 (2.2% transfer)**. `_BAND_SHAPE =
True` is active in the shipped build.

**Against the correct baseline, features 2 and 3 add 0.5552 → 0.5576 = +0.43%**,
measured on 42 windows. That is zero. At full transfer it is +0.03 composite; at
the measured rate, +0.0007.

Two corrections to the record: joint lookups do **not** overfit (TRAIN 0.5770 →
FINAL 0.5576 generalises cleanly), so the "joint 2-D/3-D/4-D scored worse than
1-D" note is not reproduced. But §6.1's "implementable ceiling, **all three
jointly** → T 0.5663" invited reading a multi-feature gain where the effect is
**entirely one feature**. Do not rebuild this.

### Augmented fine-tune, run and closed (13 Aug)

`amp_jitter=0.70` (log-uniform γ∈[0.50,1.99], sized to the measured σ_local
shift) + `noise_std=0.02`, 12 epochs, `w_tke=0`, on the untouched FINAL pool:

| config | rel | tke | mvpe | sum |
|---|---|---|---|---|
| tke-suppressed, no aug | −0.063 | −0.934 | −0.352 | −1.350 |
| tke-suppressed + augmentation | **−0.111** | −0.464 | **−0.403** | −0.979 |

Worse on mvpe, similar on rel. SELECT recovered from −3.293 (epoch 0) to −0.880
(epoch 8) as tke rose even with no tke gradient available — the amplitude-rescale
mechanism finding a foothold through noise robustness — but FINAL confirms it
doesn't survive. **All four attempted training configurations now lose on FINAL.**

`train_sim` was also checked directly rather than assumed useful: grid geometry
matches train_real exactly, but the amplitude ratio is not a single scalar
(u_std 3.59×, v_std **10.81×** — different anisotropy, not just scale), KE-map
correlation is 0.69 (real signal, not identity), and zerofrac is 0.20% vs real's
9.91% (no PIV dropout in sim). The checkpoint in hand is `sim_real_ft` —
already sim-pretrained by the organizers. Not ruled out, but not a clean lever;
needs a cheap sanity check before any training spend.

**Every sized lever is now closed.** Training (four configurations), amplitude,
band scale, and band shape have each been measured to zero or negative. What
remains is `time` (≤ +0.296 composite even if `predict()` cost nothing) and
`train_sim`, unresolved rather than closed.

### The decisive next test — RUN 13 AUG. ANSWER: NO.

Retrain with `stride=1–5` plus noise augmentation, excluding `7575_0`, and score
on the untouched pool against the frozen FNO.

**Executed twice on a Kaggle T4, 31,343 windows at stride 2 — 9.3× the 3,362 the
under-determination hypothesis blamed.** `7575_0` excluded, input-only
augmentation on.

| run | loss weights | FINAL rel | FINAL tke | FINAL mvpe | FINAL sum |
|---|---|---|---|---|---|
| A | sensitivity (.373/.239/.387) | — | — | — | SELECT flat after ep 0 |
| B | **0.8 / 0.0 / 0.2** | **−0.063** | **−0.934** | **−0.352** | **−1.350** |

Run B gave rel 80% of the loss weight and removed the tke escape hatch entirely.
**Every subscore fell on the untouched FINAL pool**, so `Δfinal < 0` under *any*
non-negative weighting — no composite-weight assumption required. The best of
five epochs on SELECT was still **−1.255 versus not training at all**.

**Mechanism (this is what was missing for a week).** The only productive
direction reachable from the stock checkpoint is an output-variance rescale — it
buys tke and pays in rel. Offer the model that trade and it takes it (run A);
forbid it and there is no productive direction left, so training simply overfits
(run B, train loss −15% while held-out fell). That is why neither 9.3× more data
nor more epochs ever mattered: ~100% of the movement lands in epoch 0.

**Mechanism — and it is not "the data holds no signal."** The shipped checkpoint
is **`sim_real_ft`**: sim-pretrained and **already fine-tuned by the organizers on
`train_real`** (its meta carries `iteration`, `best_iteration`, `best_val_loss`).
We were re-fine-tuning an already-converged model on the identical dataset. On top
of that, the only productive direction reachable from it is an output-variance
rescale — buys tke, pays in rel. Offer that trade and the model takes it (run A);
forbid it and nothing productive remains, so training overfits (run B: train loss
−15% while held-out fell).

**The under-training diagnosis was wrong.** Not "six failures" — one mechanism
that explains all of them.

### Scope limit — what is NOT closed

Both runs used `noise_std=0.02` — a figure chosen arbitrarily, never derived from
the measured PIV noise level — and `amp_jitter=0.0`, i.e. **off**. So the measured
result is *"fine-tuning with 2% additive noise fails."*

**Meaningful augmentation has never been tested**, and it is the one lever that
can plausibly beat an already-converged checkpoint, because it changes the
effective training distribution rather than re-extracting a signal the organizers
already took. It is explicitly permitted (organizer notice, 9 Aug). And §9 records
that on **9 August — the day augmentation was clarified — ten-plus competitors
landed 80.0–80.5, returning entrants gaining a median +0.390.** Circumstantial,
but the strongest signal in this document about what the field is doing that we
are not.

---

## 10. Standing rules

- **Never exploit the harness leak.** Window *k*'s target is window *k+1*'s input.
  Cross-window relationships, evaluator state and hidden-target reconstruction all
  *work* — that is precisely the problem. They exploit an implementation bug,
  cannot survive the Decision-Phase retrain, and code is inspected 25 Nov.
  **Disclosure only, never exploitation.**
- **Never compute an expected `final_score`.**
- **Nested validation.** TRAIN fits, SELECT chooses, FINAL is touched once.
- **Run `src/evaluation/preflight.py` before every submission.**
- **Preserve the incumbent artifact, hash-verified.**
- **Metadata is not an ethics rule** — it is simply an empty dict on scored calls.
  Use `.get()` with a default; never branch on a key being present.
- **One submission per day; the board keeps the maximum.** A worse submission
  costs a day and nothing else, so always ship the highest-expected-value
  candidate — but a *crash* costs the day and returns nothing at all.

---

## 11. Methodology failures, and the countermeasures

Ranked by cost.

**1. Closing a direction on undiagnosed null results.** Two failed training runs
became "the accuracy side is measured out", repeated for a week while the field
pulled away. *Countermeasure: a null result is not a closed direction until there
is a mechanism. If the mechanism cannot be stated, the direction stays open.*

**2. Shipping untestable code.** The CUDA-graph path was declared untestable and
shipped anyway, bundled with four other changes, and returned 0.000000.
*Countermeasure: untestable changes ship alone or not at all; every entry point
carries a top-level guard; `src/evaluation/preflight.py` runs first.*

**3. Carrying numbers forward instead of recomputing.** Live W̄ pasted into a
local table; a stale +4.06 bar; "100M parameters" repeated all session against an
actual 50.36M; the unreachable oracle quoted as an available reservoir.
*Countermeasure: recompute at the point of assertion, never from memory.*

**4. Predicting below one's own floor.** Three stated worst cases, three
outcomes below them (v4 sps, v6 sps, v6 time). *Countermeasure: busting a floor
means the model of the problem is wrong; stop and fix the model rather than ship
the next thing.*

**5. Building on unread inputs.** `_BATCH = 16` in a file edited minutes earlier;
`stride = 20` on line 278 of our own script; the augmentation rule change unread.
*Countermeasure: grep the constant, read the config, check the rules page.*

**6. Reactive rather than systematic log analysis.** The 2.9 ms/sample charged
outside `predict()` was inferable weeks earlier from `warmup 3.20` against
`board 7.866`; instead the whole 4.67 ms difference was assumed to be ours to
optimise, and effort went into host micro-tuning worth 0.77 ms.
*Countermeasure: extract every numeric line, every discrepancy against our own
instrumentation, and every delta from the previous log — on arrival, unprompted.*

**7. Burying critical findings.** The `torch.compile` `PermissionError` was
reported at position G of an eight-item list. *Countermeasure: actionable
findings lead.*

The single unifying pattern: **elaborate rigour built on unexamined inputs.**
Check the input before building the apparatus.

---

## 12. Session of 14-16 August: nine directions closed, one opened

Compiled 16 Aug. Everything below was measured in one session; none of it was in
the 13 Aug revision.

### 12.1 The board is deterministic on accuracy -- so every delta in this report is real

`submission_fno_diag_v9.zip` returned bit-identical output to v8. Result:

| | v8 | v9 | delta |
|---|---|---|---|
| rel_l2 | 94.168148 | 94.168148 | **0.000000** |
| tke | 74.025874 | 74.025874 | **0.000000** |
| mvpe | 92.836233 | 92.836233 | **0.000000** |
| sps | 33.337914 | 33.337914 | **0.000000** |
| time | 91.408779 | 91.235395 | -0.173384 |
| final | 78.140964 | 78.124456 | -0.016508 |

**Four subscores are exactly reproducible; only `time` varies (~5%, 6.4 -> 6.7
ms/sample).** Composite noise is +/-0.017. So the rank modifier's +0.152 is 9x
noise and the eta=0 gain of +0.108 is 6x: **nothing in the measurement history
was noise.** This question is now closed.

**And it pins a weight directly.** Only `time` changed, so
`w_time = -0.016508 / -0.173384 = 0.0952`. The 5-point least-squares fit said
0.110. **Use 0.0952** -- it is a single-variable measurement, not an inference.

### 12.2 The anomaly is MODEL-side, not data-side (pre-registered read)

Four target-free statistics logged live, compared against the 10 nearest
training trajectories in energy space. The read was stated BEFORE submitting.

| statistic | live (11 procs) | local NN range | verdict |
|---|---|---|---|
| **dec** -- the flow's own decorrelation, no model | 0.1956 | [0.1564, 0.2024] | **INSIDE** |
| **cont** -- model's boundary continuity | 0.0793 | [0.0563, 0.0760] | **ABOVE MAX** |
| **vr** -- fluctuation energy preserved | 0.697 | [0.6975, 0.8187] | **BELOW MIN** |

**The eval flow decorrelates at a normal rate; the model continues it 21% worse
and preserves 8.5% less energy, both outside the entire training range.** The
data is ordinary and our model underperforms on it -- so a better model can
score higher here. With fifteen teams above 80.7 on the same eval set, this is
the one live-confirmed opening.

### 12.3 The founding measurement error

`sim_real_ft` was fine-tuned by the organizers on ALL of `train_real`. Every
"local held-out" figure in this report (rel 96.07, mvpe 97.27) is therefore a
TRAINING score. The board's **94.168 / 74.026 / 92.836** is the baseline's only
honest generalisation measurement, and is the sole valid comparison target.
This explains the four transfer failures at a stroke: every local delta was
measured on memorised data.

**The only valid local experiment:** hold out >=12 trajectories completely,
train on the rest, compare against 94.168 / 74.026 / 92.836.

### 12.4 The error is large-scale and not linearly correctable

Spectral decomposition of the FNO residual on holdout:

| | share of error |
|---|---|
| modes the FNO CAN represent | **86.9%** |
| truncated modes | 13.1% -- worth only +0.190 rel if fixed perfectly |
| W modes 0-4 | 52.4% |
| H modes 0-4 | 60.0% |

Not a resolution problem -- a fit problem, concentrated in the largest scales.
A convex per-mode complex gain (4,224 params, exact least squares) then gained
**+0.007 rel on holdout and +0.026 in-sample**: the residual is not a linear
function of the prediction. Fitted |gain| median 0.838 -- least squares wants to
shrink FURTHER, confirming the FNO already sits at the rel-optimal amplitude.

### 12.5 The regime clusters, and the live flow sits in the worst one

k-means (k=4) on the 6-D flow fingerprint. The live fingerprint lands in the
same cluster at k=2,3,4 -- distance 0.687 vs 3.105 / 3.377 / 5.180.

| cluster | n | rel | tke | mvpe | AoA |
|---|---|---|---|---|---|
| c0 | 15 | 97.41 +/-0.54 | 70.92 | 98.20 | 0,5,10 |
| **c1 <- LIVE** | **19** | **95.00 +/-0.59** | **77.39** | **95.92** | **10,15,20** |
| c2 | 30 | 96.40 +/-1.24 | 77.11 | 96.77 | all |
| c3 | 17 | 95.58 +/-0.74 | 77.68 | 96.07 | 10,15,20 |

The live regime is where the FNO is weakest on rel/mvpe AND the tightest cluster
-- the case for a specialist. A corrector trained on 11 of those 19 gained
**+0.178 weighted** on 5 untouched holdout trajectories, but rel went **-0.110**:
the sixth consecutive instance of buy-tke/sell-rel, this time with tke weighted
0.131 against rel's 0.725 in the loss.

### 12.6 Everything correlates -- blending is closed as a class

|error| correlation with the FNO on holdout:

| component | correlation | standalone w |
|---|---|---|
| cno | +0.746 | 60.96 |
| tmean | +0.632 | 59.31 |
| dmd | +0.560 | 59.75 |
| persistence | +0.525 | 58.43 |

**DMD -- no shared weights, no shared training, an entirely different inductive
bias -- correlates at 0.560, barely below plain persistence.** Every predictor
available makes substantially the same errors, so there is no decorrelation for
a blend to exploit. A 3,624-combination search returned pure FNO.

*Caveat on that search: its objective omitted `sps` (weight 0.243) and so
returned a false PASS on the corrector. Corrected for the corrector's measured
-0.789 sps, the gain is ~0, consistent with the live result.*

### 12.7 Also closed this session

| direction | result |
|---|---|
| masked-cell zeroing | +0.014 -- FNO already emits 50x smaller there |
| lambda shrink (amplitude < native) | monotone worse; lambda=1 optimal |
| temporal gap between input and target | opposite signature to the anomaly; refuted |
| stall / AoA-15 bimodality | AoA-15 residual +0.058, i.e. zero |
| probe/whole-field masking | correlation has the WRONG SIGN; live masking is 48th pct |
| CNO and Transolver blends | both lose all three metrics; blends monotone worse |
| learned per-element error predictor | Spearman **0.6534** vs sigma_local's 0.5056; floor was 0.75 -> fail |
| `train_sim` raw alignment | anisotropic (u_std 3.59x, v_std 10.81x), KE corr 0.69, zerofrac 0.20% vs 9.91% |

Notable single-feature finding: **`framediff` predicts |e| better than
`sigma_local` (0.5484 vs 0.5056)** and is not in the shipped band. Also,
`spec_ratio` measured **-0.0195** here against the -0.498 recorded in 6.1 -- one
of those two numbers is wrong, and if it is 6.1's then one of the three features
in the "3-feature joint band" was inert.

### 12.8 Standing after this session

Best live **78.140964** (v8). Rank 10 is 80.74, rank 1 is 81.16 (15 Aug), so the
gap is ~2.60. Development phase closes 27 Sep: **42 days, ~42 unused submission
slots, ~180 Kaggle GPU hours.**

Reachable by arithmetic: **rel 96.5 / tke 78 / mvpe 96.5 live -> 80.8**, and
**rel 96.07 / tke 80.14 / mvpe 97.27 -> 81.0**. Those are this model's own
demonstrated local numbers. Per 12.2 the eval data is ordinary, so the target is
generalisation, not a breakthrough architecture.

---

## 13. Step 1 diagnostic submission (18 Aug): a new contention mechanism, and live calibration data

`submission_fno_step1_v10.zip` -- bit-identical prediction path to v8, four
target-free diagnostics (radial spectrum, sigma_local histogram, mask
connected-components, temporal autocorrelation) computed after the returned
arrays are built. 17/17 processes visible (194 B/process, vs the historical
~1180 B/process that hid 12 of 17).

### 13.1 A real, non-obvious latency mechanism: host-sync contention

    fno forward     mean  3.221 ms/sample
    bench (import)  mean  3.340 ms/sample   -- measured ALONE, no contention
    scored (real)   mean 26.010 ms/sample   -- measured WITH ~17 concurrent processes
    ratio: 7.78x        (campaign's prior baseline, code with no host syncs: 1.00-1.07x)
    ingestion's own summary: 30.2 ms/sample

    time_score: 83.088 (was 91.433 at v8's 6.40ms)   delta -8.345
    x w_time (0.0952, MEASURED not fitted) = -0.7944 composite for THIS submission
    Board keeps the max -- standing 78.140964 unaffected.

**Mechanism.** The diagnostic code has ~15-20 individual `.cpu()`/`.item()`-style
host synchronization points (8 spectral-bin sums x 2 channels, a histogram
reduction, a mask transfer, 3 autocorrelation lags). Each is a full CUDA
device sync. Measured ALONE at import (`bench`), this cost nothing -- bench is
within 4% of the bare FNO forward. Under REAL contention (~17 processes
sharing one A800, each hitting its own sync points at similar moments), the
cost is 7.78x.

**This overturns the season's "no GPU contention" finding for code with many
host syncs.** The prior result (ratio 1.00-1.07x, memory `realpde-t1-*`) was
measured on code with almost no intermediate synchronization -- `predict()`'s
own hot path does exactly ONE `.cpu().numpy()` per returned array, at the very
end. Contention is not a property of the GPU or the harness in general; it is
a property of how many synchronization points a specific build introduces.

**Design constraint for any future correction head (the plan's Step 3, latency
cap <=0.30ms):** local/import-time single-process benchmarking CANNOT validate
this. A block with many small host syncs can bench near-free and still cost
7-8x live. The only known-safe pattern is the one `predict()` already uses:
pure tensor ops until a single final transfer.

### 13.2 Calibration data (mean across 17 processes)

| diagnostic | result |
|---|---|
| SPEC_u, 4 radial bins (low->high) | [1477.5, 129.3, 30.0, 14.7] -- bin0/bin1 **11.4x**, bin1/bin2 **4.3x** |
| SPEC_v, 4 bins | [367.9, 16.7, 4.2, 2.0] -- same steep low-frequency dominance |
| SIGHIST_u, 6 bins (edges 0/.003/.008/.016/.028/.045/inf) | [0.252, 0.281, 0.151, 0.121, 0.097, 0.098] |
| MASKCC, 5 size bins (1/2/4-7/8-15/16+) | mean [0.82, 0.18, 0.12, 0.18, **1.18**] |
| AC lags 1/2/3 | **[0.686, 0.417, 0.146]**, std [0.006, 0.011, 0.014] |

**Energy is overwhelmingly at large spatial scales** (bin0/bin1 = 11.4x),
consistent with (and now measuring, for the live flow specifically) the same
large-scale concentration already found in the FNO's ERROR spectrum (86.9% in
representable modes, 52-60% in the lowest four -- see section 12.4).

**53.2% of sigma_local mass sits below 0.008**; only 19.6% above 0.028. A full
empirical histogram, not just the mean/percentile summary used until now.

**The live mask is essentially always ONE large connected blob**, not
scattered dropout: every process (100%) has a component in the 16+-pixel bin,
only 47% have any additional sub-16 fragment. This directly corrects Step 4.3's
implicit assumption of scattered structured dropout -- the real structure is a
single coherent occlusion region (consistent with a physical laser-obstruction
or reflection zone, not speckle noise).

**Autocorrelation decays FASTER than a geometric (AR(1)) process would
predict**: lag1=0.686 implies lag2=0.471/lag3=0.323 under pure AR(1); observed
is 0.417/0.146 -- roughly half the AR(1) prediction by lag 3. Consistent with
an oscillatory component (the ~21.9-frame shedding period) causing correlation
to fall off faster than first-order autoregression. Extremely tight across
processes (std 0.006-0.014) -- confirms a single flow condition.

**Implications for Step 4:** frequency-domain gain (4.1) has a real target
spectrum to match. Correlated noise (4.2) should NOT be a simple AR(1)/white
process -- it decays faster and should carry an oscillatory component. Mask
dropout (4.3) should synthesize ONE connected blob at realistic sizes, not
independent per-pixel holes.


### 13.3 Local-vs-live reference check (18 Aug) -- Step 4's premise mostly fails

Same 4 diagnostics, computed on 788 real train_real windows from the cluster-1
regime pool (19 trajectories), identical bin definitions to the live build.

| diagnostic | local vs live | verdict |
|---|---|---|
| SPEC_u gain | [1.00,1.08,0.98,0.95] | matched |
| SPEC_v gain | [0.81,0.95,0.88,0.87] | local has MORE energy, opposite of 4.1's assumption |
| AC lags 1/2/3 | local 0.697/0.426/0.147 vs live 0.686/0.417/0.146 | **essentially identical** |
| mask: frac. w/ extra sub-16 fragment | local 87.3% vs live 47% | **local is MORE fragmented**, not less |

The oscillatory (faster-than-AR(1)) temporal structure is ALREADY present in
real training data at nearly the same ratio as live -- it is the physics
(vortex shedding), not a gap synthetic noise needs to replicate. Spectrum
shape is matched to within 8% on u. Mask fragmentation runs OPPOSITE to what
was assumed when this section was first written: live is cleaner, not local.

**Reproduces the founding error (12.3) in a new form**: within the correct
regime, local and live are close on every LOW-ORDER statistic measurable this
way. There is no first-order distributional gap for augmentation (plan Steps
4.1/4.2) to close. Step 4.3 (mask dropout) is not ruled out but points the
opposite direction from its original write-up.

Also found: the plan's Step 3.1 "first select" high-u_std/framediff pool (659
windows) is 29-of-31 from a SINGLE trajectory (26700_15, Re 26,761) at stride
20 -- not a real multi-trajectory generalization test. Correction head training
uses the cluster-1 regime pool (19 trajectories) instead.


### 13.4 Four required gates, closed (18 Aug)

**Gate 1 -- v(no corrector)->v(corrector) transfer audit.** The plan's literal
"v5->v6" pair changes bands AND introduces the corrector at once. Used the
confound-free live pair instead: v8 (bare FNO, bands, no corrector, 78.140964)
vs v4 (same bands, +corrector eta1.4, 78.033132), against the matching LOCAL
stock->eta14 delta (boosted_scores.json).

| | local | live | transfer |
|---|---|---|---|
| rel | -0.226 | -0.148 | 65.5% |
| tke | +2.711 | +1.699 | 62.7% |
| mvpe | -0.190 | -0.029 | 15.3% |

Mean 47.8%. **PASS** (>=30%).

**Gate 2 -- multivariate live-similarity window selection**, replacing the
two-threshold filter. The old filter (u_std>0.14 AND framediff>0.012) kept 31
windows from 2 trajectories at stride 20 (matches the earlier 659-at-stride-1
finding: same near-single-trajectory concentration, just different stride).
Ranked 2,113 AoA-{10,15,20} windows by distance to the live fingerprint in the
SAME standardized 6-D space as the regime clustering:

| top-K | distinct trajectories | max single-traj. share |
|---|---|---|
| 500 | 20 | 8.4% |
| 1000 | 29 | 4.2% |
| 2000 | 50 | 2.1% |
| 5000 | 51 (all) | 0.8% |

**Caveat found while building this:** several of the raw top-K windows come
from the 5 established holdout trajectories (21600_15, 22875_15, 20325_15,
24150_15, 19050_15) and MUST be excluded from any training pool -- the
multivariate score alone does not know about the holdout split.

**Gate 3 -- linear corrector micro-gate.** Closed-form ridge regression (~18
params: FNO pred u/v, sigma_local u/v, framediff u/v, mask, bias -> residual
u/v), fit on the multivariate top-2000 pool (holdout excluded), evaluated on
the untouched 5-trajectory holdout. Different question from the 13 Aug
per-mode complex gain (which saw only the FNO's own output, found +0.007 rel):
this one also sees the auxiliary features Step 3.2 proposes for the full head.
**FIRST PASS RESULT (superseded, see 13.5): +0.325% residual reduction** on
77,946,880 elements (1,903 windows, 46 trajectories -- v1 crashed on eager
np.stack, same OOM class already fixed once this session in Windows()).
Evaluated on 35 holdout windows. Training pool mixed AoA 10/15/20 (the
"regime-plausible" set from clustering, not AoA-15 specifically) and the
holdout was a fixed trajectory subsample, not ranked by live-similarity.

**NOT counted as a third independent refutation.** The 13 Aug per-mode gain,
and this fit, differ in feature set, fitting method, AND data pool -- treating
them as replications overstated the evidence. GATE 3 IS NOT CLOSED. Rerunning
with AoA-15-only selection and live-similarity-ranked holdout -- see 13.5.

**Gate 4 -- probe-weighting, RE-SPECIFIED with a proper scale normalization
(18 Aug).** The first version combined rel_l2 and mvpe_rel as two additive
terms and derived a pixel-count-based lambda (0.0088/0.0176). That form is
broken at a deeper level than the lambda value: rel_l2 and mvpe_rel are each
ALREADY normalized against DIFFERENT target norms (the whole field's vs. the
probes' own, with mvpe_rel additionally time-averaged first) -- there is no
common denominator to derive a clean weight ratio against.

**Corrected spec: a single weighted relative-L2, not two combined terms.**

    weighted_rel_l2(p, t, w) = sqrt(sum(w_i*(p_i-t_i)^2)) / sqrt(sum(w_i*t_i^2))

with w_i = w_probe at the 36 graded (PROBE_Y x PROBE_X) locations and w_i = 1
elsewhere -- structurally identical to the existing rel_l2, substitutes
directly for it in the training loss, one weight vector, one normalization
basis. "Probe weight = 2x" now means exactly that.

Ablation: sweep w_probe in {1.0, 1.5, 2.0} directly (1.0 reproduces plain
rel_l2 exactly), selected on SELECT via E_local, w_probe=1.0 as the safe
default if none beat it.


### 13.6 Masterplan v11 execution (18-19 Aug)

**Section 3, spectral augmentation -- power/amplitude bug caught before any
training code shipped.** The measured live/local table (13.5) is a POWER
ratio; multiplying FFT coefficients directly by it squares the intended
effect. Correct multiplier: amplitude_gain = sqrt(power_ratio). Caught by
review before implementation, not after a wasted run.

**Implementation bug found and fixed independently**, via a synthetic-signal
self-test: `verify_gate()`'s own power measurement used `axes=(0,1)` on a
(T,H,W,2) array -- the TIME axis, not the spatial one. `spectral_boost()`
itself was correct throughout (verified by an impulse test: gain 1.371
applied exactly at rad=0, 1.266 at rad=14.1, matching AMP_GAIN_U exactly).
Fixed to `axes=(1,2)`.

**Pool-selection interacts with gain calibration -- they cannot be done
separately.** Applying the AoA-15 POOLED-AVERAGE gain to the actual "11 most
live-similar" training pool OVERSHOOTS: those trajectories are already closer
to live than the AoA-15 average (that is why they were selected), so they
need less correction. Verified live: pooled-average gains pushed the pool's
own spectrum from live/local ~1.0-1.36 to ~0.65-0.88 (overshot past live).
Re-derived gains from the POOL'S OWN spectrum (473 windows, 11 trajectories):
amp gain u [1.295,1.203,1.132,1.114], v [1.236,1.142,1.159,1.157] --
noticeably smaller than the pooled-average figures.

**GATE: PASS on the full pool** (473 windows, matching the calibration
sample): every bin lands within 0.1-1.5% of the live target, both channels.
An 11-window subsample first showed a false bin0 failure (+23.5%/+20.9%) from
sampling noise alone; the full-pool check resolved it. Masked cells verified
preserved exactly (never boosted, always restored to 0).

**Training pool selection.** window_similarity_aoa15.npy ranked by live-
distance, aggregated to trajectory level, EXCLUDING src/evaluation/preflight.py's hardcoded
FINAL pair (21600_15, 25425_15) so its own sanity check stays uncontaminated
-- not the full 5-trajectory holdout, since the plan's decision rule is
live-only (local is sanity, never a selector) and excluding all 5 would push
the pool out to distance 3-5 for lack of anywhere closer to draw from.
Result: 24150_15, 22875_15, 26700_15, 20325_15, 19050_15, 17775_15, 16500_15
(dist 1.19-2.45, genuinely live-similar) + 13950_15, 12675_15, 11400_15,
10125_15 (dist 3.0-3.9, included to reach 11, honestly NOT live-similar --
flagged rather than glossed over).

See `src/training/spectral_augment.py`, `live_similar_pool.npy`, `pool_spectrum.npy`.

## 14. Experiment 12.3, finally run (22 Aug): the eval set is not anomalous

12.3 specified this exact experiment five weeks ago and it was never executed:
*"The only valid local experiment: hold out >=12 trajectories completely,
train on the rest, compare against 94.168 / 74.026 / 92.836."* Every "local
held-out" figure used until today (including 13.5, 13.6, and every SELECT/
FINAL number in `src/training/finetune_fno_kaggle.py`) used the 4-trajectory SELECT/FINAL
quad, which -- per 12.3's own diagnosis -- is not a real holdout: `sim_real_ft`
was fine-tuned by the organizers on ALL 81 usable trajectories, so every local
figure was a training score.

### 14.1 The true composite weights (correcting 12.1's table)

`scoring.py`'s `aggregate_sps` builds SPS from the SAME rel/tke/mvpe errors the
other three subscores use (`weight_dm=0.5, weight_tke=0.3, weight_mvpe=0.2` of
a `(1-pm)*exp(-band)` term, `pm=err/(0.5+err)`). The fitted composite table
(rel .476/tke .086/mvpe .095/sps .243) is therefore a set of PARTIAL
derivatives; every accuracy metric also moves SPS. Total derivative at the
live v8 operating point (rel 94.168/tke 74.026/mvpe 92.836, band term T=0.4914):

| metric | partial (old table) | true (partial + via SPS) | understated by |
|---|---|---|---|
| rel | 0.476 | **0.649** | 36% |
| tke | 0.086 | **0.131** | 53% |
| mvpe | 0.095 | **0.160** | 68% |

Consequence: "rel is worth 5.5x tke" (12.1) is really 4.9x; mvpe was
undervalued the most of the three. `src/training/finetune_fno_kaggle.py`'s loss weights
used `sensitivity(err)` alone (implicitly TRUE_W=(1,1,1)), which at the stock
operating point put rel and mvpe within a few percent of each other -- very
likely why the "sensitivity weights" config (12.1's table: FINAL Drel -0.89,
Dtke +2.66) bought tke and sold rel: the loss itself was structurally biased
toward the cheap metrics, independent of any data problem. Fixed 22 Aug.

Also closed analytically: the live band term T=0.4914 sits AT the
reversed-hazard optimum `f(h)/F(h)=2/sigma` for every plausible local error
scale (see `true_gradients.py`) -- the absolute maximum any band scheme can
add is +0.15 composite, at 100% coverage with zero-width bands, which is a
contradiction in terms. This closes 12.6's SPS-headroom estimate to zero, not
"mostly unreachable."

### 14.2 The real holdout (`src/splits/build_holdout_v12_3.py`, `data/splits/holdout_v12_3.json`)

14 trajectories, selected as the 14 CLOSEST to the live fingerprint in
`data/splits/matched_splits.json`'s standardized 6-D space -- deliberately the hardest
version of the test, not an arbitrary one. First attempt held out almost the
entire top of the Reynolds ladder and left 12/14 members with no training
neighbour ABOVE them (an extrapolation test, harder than the live task, which
memory confirms is interpolation). Fixed by keeping the 3 trajectories at the
Re ceiling (26700) in training unconditionally; re-ran the bracketing check:
**14/14 held-out trajectories now have a training neighbour on both sides at
the same AoA.** SELECT = the original 4 (already used for SPS band-multiplier
fitting, which never touches model weights or rel/tke/mvpe, so still valid for
epoch-picking); FINAL = 10 new trajectories untouched by anything in this
campaign until now.

### 14.3 Result (`src/evaluation/score_holdout_v12_3.py`): the gap is real, ordinary, and closable

The FROZEN shipped checkpoint (`sim_real_fno_fp16.pth`, zero training by us),
scored on this holdout, no band/SPS machinery, plain rel/tke/mvpe:

| | live (board) | holdout (588 windows, 14 traj) | final (420 windows, 10 traj, single-touch) |
|---|---|---|---|
| rel | 94.168 | 96.296 (+2.13) | 96.379 (+2.21) |
| tke | 74.026 | 80.715 (+6.69) | 81.039 (+7.01) |
| mvpe | 92.836 | 97.154 (+4.32) | 97.064 (+4.23) |

Consistent with the independent 5-trajectory check run earlier the same day
(rel 96.25/tke 79.89/mvpe 97.33) -- two different holdout constructions agree
to within ~1 point on every metric. **This retires the 14 Aug "eval set is
anomalous, cause unknown" framing.** The four refuted hypotheses (flow energy,
stall/AoA, temporal gap, masking) were never going to find a cause, because
there wasn't a live-specific anomaly to explain -- an ordinary neural-network
train/test gap, on a 50M-parameter model shown its entire training distribution
as training data, produces exactly this signature.

Also closes the "live PIV is intrinsically noisier" hypothesis quantitatively,
not just qualitatively: white measurement noise attenuates autocorrelation by
`S/(S+N)` at every lag. Live/local AC ratios (13.2 vs 13.3) are 0.984/0.979/
0.993 -- excess live noise bounded at 1.5% relative. Explaining a 2.13-point
rel gap as noise needs 10-20x more than that. Not the mechanism.

Feeding the holdout numbers through 14.1's true-gradient composite (keeping
LIVE's own worse band term T=0.4914 -- crediting the model nothing there):

    holdout-14  (588 windows):  rel 96.296  tke 80.715  mvpe 97.154  -> composite 81.21
    final-10 (single-touch):    rel 96.379  tke 81.039  mvpe 97.064  -> composite 81.29

**This is not a hypothetical ceiling -- it is what THIS SAME FROZEN MODEL
already does on data it was never shown.** It lands inside the 22 Aug
leaderboard's top cluster (80.99-81.64, nine of thirteen scores dated 17-21
Aug) almost exactly. The gap from live 78.166 to ~81.2-81.3 is a
generalization gap, and the fix is training for generalization -- a real
holdout, not a bigger local sample of memorized data.

### 14.4 What this closes and what it opens

CLOSED: masterplan v11's SPS band track (14.1: bands are provably at their
analytic optimum). The "7% model-level transfer law" as a ceiling on trained
interventions -- it was measured on deltas fit against memorized local data
(12.3), so it describes how badly a MEMORIZING fine-tune transfers, not an
intrinsic property of training. AdaBN / test-time BatchNorm (`experiments/early_diagnostics/adabn_test.py`,
22 Aug): -0.81 composite, negative on 5/5 -- live sits inside the training
range, so there is no distribution shift left for it to correct once the real
issue (holdout, not memorization) is fixed elsewhere. WiSE-FT weight
interpolation between `sim_fno` and `sim_real_fno` (`experiments/early_diagnostics/wiseft_test.py`, 22 Aug):
monotonically catastrophic (-7.5 at alpha=0.95) -- `||theta_ft-theta_sim||/
||theta_sim|| = 1.094`, the fine-tune moved further than the pretrained
weights' own norm, so the two checkpoints do not share a basin and
interpolation crosses high loss between them.

OPEN, and now the campaign's single highest-value action: train on
`data/splits/train_pool_v12_3.json`'s 67 trajectories (holdout excluded), starting from
`sim_fno.pth` (sim-pretrained ONLY -- `sim_real_fno.pth` has already seen all
81 real trajectories including this holdout, so it is not a clean starting
point for THIS experiment). `src/training/finetune_fno_kaggle.py` updated 22 Aug: new
SELECT/FINAL, corrected TRUE_W loss weights (14.1). Ran 23 Aug as Experiment
1 -- see 15.1, it plateaued and lost.

## 15. The 27 Aug - 6 Sep session

### 15.1 Experiments 1-3, run (23 Aug): all three lose on FINAL

**Experiment 1** (`sim_fno.pth`, lr=1e-4, 80 epochs, unfrozen): plateaued at
epoch 10 and never improved for the remaining ~70. FINAL(10): rel 94.266 /
tke 78.782 / mvpe 95.505 (sum 268.554) -- a LOSS of -5.928 sum vs this same
checkpoint's own frozen holdout score (274.482).

**Experiment 2** (`sim_real_fno.pth`, lr=2e-5, 30 epochs, unfrozen -- not a
clean holdout test, since this checkpoint already saw these 14 trajectories
during the organizers' own training, but the one genuinely untested
configuration left at the time): official stride-20 rescore on FINAL came
back TRUE-composite -0.10 (rel -0.480 / tke +2.099 / mvpe -0.396) -- a wash
tipping negative, the exact "buy tke, sell rel/mvpe" shape every unfrozen
attempt has shown.

**Experiment 3** (identical to Experiment 2 except seed=1, restarts=2, to
rule out one seed's bad luck): same outcome, same mechanism. **All four
unfrozen fine-tune attempts (1 through 3, plus the earlier stride-1..5 runs
in sec.13) now lose on FINAL.** Full-model unfrozen fine-tuning is CLOSED --
see 15.8 for the exception found 6 Sep.

### 15.2 The holdout-contamination finding (27 Aug) -- corrects sec.14's ceiling

Sec.14.3's "~81.2-81.3 ceiling" reasoning has a hole: `sim_real_fno` scores
HIGHER on the "true holdout" than on its own live board.

| checkpoint | rel | tke | mvpe | on what |
|---|---|---|---|---|
| `sim_fno` (never touched real data) | 79.709 | 66.872 | 75.199 | holdout final-10 |
| `sim_real_fno` (org real-finetuned) | 96.379 | 81.039 | 97.064 | SAME holdout final-10 |
| `sim_real_fno` | 94.168 | 74.026 | 92.836 | actual live |

That is impossible if the holdout were genuinely unseen by this checkpoint.
The holdout is drawn from `train_real`, which sec.0's founding-error logic
already says `sim_real_fno` was trained on in full -- nothing excludes those
14 trajectories from the ORGANIZERS' original training, only from THIS
CAMPAIGN'S later fine-tuning. Confirmed not just argued: `sim_fno` (which
genuinely never touched real data) scores dramatically worse on the same
holdout (79.7/66.9/75.2), proving the holdout CAN detect a real
generalization gap -- `sim_real_fno`'s suspiciously-good score on it is the
signature of memorization, not evidence the holdout is unseen.

**Consequence: there is no known-clean local proxy for live performance
using `sim_real_fno` as the reference point.** Every "beats baseline
locally" claim from 22 Aug onward that compared against `sim_real_fno`'s
holdout score needs re-reading with this in mind.

**Corrected methodology going forward**: judge a clean-start candidate
(fine-tuned from `sim_fno`/`sim_cno`, never touched by us before the honest
pool) against the FROZEN version of that SAME clean starting checkpoint on
the holdout -- not against `sim_real_fno`'s contaminated holdout score. Both
sides equally clean = a fair ablation ("did training help at all"). Still
true regardless: only a LIVE submission is fully trustworthy. Local numbers
answer "did training help vs not training," nothing about whether it beats
78.14 or the field's 81+.

### 15.3 AOT-compile probe (submitted, `submission_fno_probe_aot_v15.zip`)

Zero-risk diagnostic, bit-identical prediction path to v8 (preflight
confirmed baseline-match). Tried `torch._export.aot_compile`,
`torch._inductor.aot_compile`, `aoti_compile_and_package` (whichever exists
in the container's torch 2.2.2), fully guarded, result never used for real
prediction. Live result: `requests=Y:2.31.0` (present in the real eval
container, so CNO's dnnlib dependency was never actually blocking anything
there), only 2 of 3 candidate APIs exist in torch 2.2.2, and
`aot=FAIL:torch._export.aot_compile:PermissionError:[Errno 13] Permission
denied` -- the identical root cause as JIT `torch.compile`'s prior failure.
**CLOSED: AOT compilation is blocked at the container level (compiler-
toolchain execution restriction), same as JIT. Not a cache-directory issue,
not fixable from inside the submission.**

### 15.4 UNet and DPOT-S off-the-shelf (pilot only, not fine-tuned)

Both fail to beat FNO's task-tuned checkpoint, but both clearly beat the
fair "no adaptation at all" floor (frozen `sim_cno`) by a similar margin:

| model | rel | tke | mvpe | note |
|---|---|---|---|---|
| FNO (task-tuned by org) | 96.088 | 79.915 | 97.379 | SELECT pool |
| UNet off-the-shelf | 89.019 | 70.818 | 86.112 | wrong resolution (trained 128x64, fed 32x64) |
| DPOT-S off-the-shelf | 88.818 | 70.072 | 84.364 | has a real resize() adaptation mechanism, still wrong-res |
| frozen `sim_cno` (no adaptation) | 77.531 | 67.077 | 73.807 | the floor |

Neither is proven to close the gap to FNO without an actual resolution-
matched fine-tune (bigger, less-certain bet than CNO, not started). DPOT's
`resize()` (FFT-based, `dpot_libs/utils/utilities.py`) is a genuine built-in
resolution-adaptation mechanism UNet lacks entirely -- prefer DPOT if this
path is ever pursued. Gotcha hit once: DPOT's wrapper has `normalize=False`
-- caller must Gaussian-normalize before/after forward() exactly like
FNO/CNO; skipping this first gave a catastrophic wrong-signed result (rel
64.0).

### 15.5 Genuinely untried, confirmed by directory-wide search (not prose recall)

DPOT-L, WDNO, MWT, Galerkin Transformer, DeepONet, `train_sim` as a training
source -- no checkpoints, no scripts, no prior mentions beyond the vendored
model-factory listing every architecture it CAN load. WDNO has the strongest
mechanistic case (diffusion-based, doesn't share MSE-regression's
conditional-mean-collapse failure mode that costs FNO its tke score) but is
a bigger lift than CNO, not started. MWT/Galerkin/DeepONet are NOT
promising -- the org's own foil benchmark has them equal-to-worse than FNO
(0.0210/0.0237/0.0375 rel L2 vs 0.0206). `train_sim` has real problems
(anisotropic amplitude ratio, u_std 3.59x/v_std 10.81x -- not a single
scalar; near-zero PIV dropout, 0.20% vs real's 9.91%) -- not a clean lever.

### 15.6 The freeze-bn-fc breakthrough (6 Sep) -- first clean-start fine-tune to win on FINAL

`--freeze-bn-fc` (masterplan v11 sec.5 -- freezes every parameter except
`fc0/fc1/fc2/bns.*`, so spectral conv weights physically cannot move) and
`--spectral-aug` (masterplan v11 sec.3) had been implemented in
`src/training/finetune_fno_kaggle.py` since ~23 Aug with ZERO execution record in this
report -- found via grepping the flag names directly, not prose recall.
Same story for SWA/EMA: zero prior mentions, and `run()` never snapshotted
per-epoch checkpoints before this session added `--snapshot-last-k` +
`src/training/swa_average.py`.

Ran `--freeze-bn-fc --spectral-aug` from `sim_fno.pth` (clean start) on
`data/splits/train_pool_v12_3.json`, 30 epochs, best epoch 21 by SELECT:

| | rel | tke | mvpe | sum |
|---|---|---|---|---|
| stock (frozen `sim_fno`) | 79.709 | 66.872 | 75.199 | 221.781 |
| finetuned | 92.462 | 71.939 | 93.527 | 257.927 |
| delta | +12.753 | +5.067 | +18.327 | +36.147 |

**First of five fine-tune attempts (this + the four in 15.1) to win on ALL
THREE FINAL subscores at once.** Confirms rather than works around the
buy-tke-sell-rel diagnosis: freezing the weights that carry out the
amplitude-rescale escape hatch removes the trade, and a real positive
gradient direction was available the whole time.

**Calibration -- does NOT yet beat what's already live.** 79.709 is the
do-nothing floor, not the bar. Against `sim_real_fno`'s actual live numbers:

| | rel | tke | mvpe |
|---|---|---|---|
| finetuned (this run) | 92.462 | 71.939 | 93.527 |
| `sim_real_fno` (live) | 94.168 | 74.026 | 92.836 |
| delta | -1.706 | -2.087 | +0.691 |

Through TRUE_W (0.649/0.131/0.160) as a comparison delta (same convention as
15.1's "TRUE-composite" figures, not a live-score prediction): approximately
-1.27 composite units below what is already shipped. **Do not submit this
checkpoint** -- it was fine-tuned from the weaker starting point (`sim_fno`,
never seen real data), so it hasn't had the chance to also carry forward
what the organizers' real-data fine-tune already gave `sim_real_fno`.

### 15.8 `--freeze-bn-fc` from `sim_real_fno.pth` -- RUN, CLOSED, NEGATIVE (6 Sep)

The obvious follow-up to 15.6: `--freeze-bn-fc --spectral-aug` starting from
`sim_real_fno.pth` instead of `sim_fno.pth`, 30 epochs, lr=2e-5, same pool.
SELECT sum sat BELOW the stock checkpoint's own SELECT score (267.382) on
all 30 epochs -- a tight, flat, slightly-negative band (-0.6 to -1.6 sum, no
upward trend), best epoch 3 (266.775). FINAL confirmation (touched once):

| | rel | tke | mvpe | sum |
|---|---|---|---|---|
| stock | 95.398 | 77.491 | 96.184 | 269.072 |
| finetuned | 95.366 | 76.568 | 96.156 | 268.090 |
| delta | -0.032 | -0.923 | -0.028 | -0.982 |

**CLOSED, do not re-run.** Notably NOT the "buy tke, sell rel" trade seen in
every unfrozen attempt (that shows tke up, rel down) -- here rel and mvpe
are untouched at noise level (±0.03) while tke alone drops nearly a full
point. Mechanism: fc/bn is the model's output-scale/calibration layer, and
the organizers already calibrated it near-optimally for `sim_real_fno`'s tke
specifically. With the spectral backbone frozen (so rel/mvpe structurally
cannot move), the only thing left for fc/bn to do is drift away from that
already-good calibration, which can only cost the one subscore it controls.

**Confirms 15.6's win is conditional on the starting checkpoint being far
from convergence with real headroom left -- `--freeze-bn-fc` does not help
(and mildly hurts) a checkpoint already near its own optimum for this loss.**
This narrows the path forward: more epochs on the same loss from the same
already-good checkpoint is not the missing piece.

**What's next, given 15.6 and 15.8 together**: two genuinely untried
directions remain live --
(a) meaningful augmentation (`--amp-jitter`) COMBINED with `--freeze-bn-fc`
    on `sim_fno.pth` -- the one config with real headroom, never yet paired
    with real augmentation;
(b) an INTERMEDIATE freeze ratio from `sim_real_fno.pth` -- unfreezing one
    more spectral block instead of jumping straight from 0% trainable
    (15.8, closed) to 100% trainable (Experiments 2/3, closed) -- the middle
    ground between the two tested extremes has never been tried.

Also pending: run `src/training/swa_average.py` on 15.6's 5 snapshots and score the
result against the fair frozen-`sim_fno` baseline -- NOT yet done (only the
5 raw snapshot files and the ready-to-run command exist so far).

### 15.7 Tooling bugfixes (6 Sep, `src/training/finetune_fno_kaggle.py`)

`pool_file` was hardcoded to `/kaggle/working/train_pool_v12_3.json`, which
only holds if the file is manually copied into session scratch first -- a
dataset attached the normal way mounts read-only under
`/kaggle/input/<slug>/...` instead, and this path was never run through
`_autodiscover()` (which only fires when data/ckpt are ALSO missing).
Crashed every fresh session with the json attached as an input dataset,
after already spending minutes decompressing the main .npz. Fixed: falls
back to searching `/kaggle/input` before giving up, same as data/ckpt/kit.

The file's own `__main__` block auto-launched Experiment 3 (15.1, already
closed, TRUE-composite -0.10) every time it was pasted into a fresh Kaggle
cell -- burning real GPU minutes on a dead experiment before any new command
could run. Fixed: pasting now only defines helpers (build/run/main/etc.),
trains nothing automatically; old config kept as `_experiment_3()` for
reproducibility only.

### 15.9 v17 blend, LIVE 79.260788 (25 Sep) -- first gain since 13 Aug, rank 121 -> 87

`submissions/v17/submission_blend_v17.zip`: v8 + per-(lead, channel, sigma_local sextile) blend of the FNO
toward last input frame (early leads) / window mean (late leads) in calm regions, applied to
both prediction and band centre; mean-preserving tke boost a=clip((E_in/E_pred)^0.25,1,1.6);
band half-widths from a lookup on input frame-difference RMS. Live vs v8: rel -0.064, tke +1.342,
mvpe +0.249, time -0.278, sps +3.949, final +1.120. Proxy A (Y'=P+k(Y-P), k fitted to v8 live
T) predicted sps +4.01 / composite +1.33 -- the first local proxy validated against live.
Root cause of the Aug-Sep stall: every decision was judged on a holdout the stock FNO had
memorized, which hid that trivial window estimators beat the FNO in calm regions.
