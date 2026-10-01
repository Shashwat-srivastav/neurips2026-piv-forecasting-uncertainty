# Interval methods for the Safe Prediction Score (T term) — literature scan, 25 Sep 2026

Objective: maximise T = mean_e exp(-(u-l)/σ)·1{l ≤ y ≤ u}, σ = 0.0563870, bounds from the input window + prediction only.

## 0. What the theory says (read this before the methods)

**Per-element optimum.** Given a conditional residual density f and CDF F for element e, the objective separates by element. You maximise exp(-(u-l)/σ)(F(u)-F(l)) for each one separately:
- Optimal [l,u] satisfies f(l) = f(u) = (F(u)-F(l))/σ. So the interval is a **highest-density interval**. It is centred on the residual **mode**, which is neither the mean nor the FNO prediction.
- In the symmetric case: f_|e|(h)/F_|e|(h) = 2/σ.
- **Laplace residual, mean |e| = b (closed form):** h* = b·ln(1 + σ/(2b)), coverage = r/(1+r), T* = (1+r)^(-1/r)·r/(1+r), where r = σ/(2b).
- **Gaussian residual (numeric, computed here):**

| resid std s | h* | h*/s | coverage | T* |
|---|---|---|---|---|
| 0.002 | 0.0044 | 2.21 | 0.97 | 0.83 |
| 0.005 | 0.0089 | 1.78 | 0.93 | 0.68 |
| 0.010 | 0.0140 | 1.40 | 0.84 | 0.51 |
| 0.015 | 0.0175 | 1.17 | 0.76 | 0.41 |
| 0.020 | 0.0200 | 1.00 | 0.68 | 0.34 |
| 0.030 | 0.0231 | 0.77 | 0.56 | 0.25 |
| 0.050 | 0.0258 | 0.52 | 0.39 | 0.16 |

Consequences:
1. **h\*(s) is concave and saturates at σ/2 ≈ 0.028.** Near 0 it goes as ~2.2 s, and for large s it is ≈ σ/2. A linear rule h = a + K·s is structurally wrong. The hand-fitted sextile table in `experiments/interval_recipes/v8/submission.py` (multipliers 1.5 down to 0.25) is an empirical approximation of this curve. Replace it with the curve itself.
2. **T\*(s) is strongly convex in s.** Two populations with the same mean error give a much higher T when the error predictor separates them well: T*(0.005)+T*(0.025) > 2·T*(0.015). This is exactly the "top teams localise low-error elements" gap. **Error-predictor resolution matters more than the band formula.**
3. Conformal "guaranteed 1-α marginal coverage" is **not** the goal here. Conformal is useful only as a **scale calibrator**, and only if it is scale-free under local→live shift (live σ_local ≈ 1.95× local).

**Gaps in the current shipped band (v8).** It is (a) **constant over the 20-step horizon**, (b) symmetric around the prediction, and (c) built on one feature: temporal std of the input window. Each gap maps onto an established method below.

---

## Candidates (ranked)

### 1. Plug-in decision-optimal band from a learned residual-scale predictor (heteroscedastic error head / "error predictor")
- **Literature:**
  - Nix & Weigend 1994, and Lakshminarayanan et al. 2017: heteroscedastic Gaussian/Laplace NLL heads.
  - Lei et al. 2018: locally-weighted conformal, where the residual model is fitted on a held-out split.
  - Error-prediction side-networks for image restoration, e.g. *Trustworthy Endoscopic Super-Resolution*, arXiv 2604.18001: a light net predicts pixel-wise error.
  - *Direct Learning of Calibration-Aware Uncertainty for Neural PDE Surrogates*, arXiv 2602.11090: FNO noise parameters learned on a held-out split, whose "uncertainty fields concentrate in high-error regions".
  - Decision-theoretic step: the interval that maximises expected exp(-w/σ)·coverage is the HPD rule in §0.
- **Assumption:** the conditional |residual| of the **frozen** point predictor is roughly Laplace/Gaussian with a scale predictable from window + prediction features.
- **Training:** yes, but tiny. Use a GBM (LightGBM/sklearn HistGB) or a 2-layer MLP on per-element features. Use out-of-fold residuals of the frozen FNO on ~70 trajectories. That is ~10^6–10^7 element rows after subsampling, so it takes minutes on CPU.
- **Input-window rule:** yes. All features come from x (the 20-frame window) and ŷ.
- **Mechanism:** raises the resolution of s(e), which captures the convexity gain from §0 point 2, and swaps the linear/sextile rule for the exact saturating h*(s).
- **Recipe:**
  1. Out-of-fold residuals: split trajectories into 5 folds. The organizers' FNO is not trained on our pool, so plain residuals are fine. If a corrector is involved, use OOF corrector predictions.
  2. Per-element features:
     - lead time k (1..20), the most important missing feature;
     - σ_local (temporal std), and σ_local of the last 5 frames;
     - |x_T − x_{T−1}| and |x_T − x_{T−5}| (recent tendency);
     - |ŷ_k − x_T| (how far the forecast moves);
     - |∇ŷ_k| and |∇x_T| (shear, wake edges);
     - |ŷ_k|, and the window mean;
     - channel u/v;
     - (i, j) position and distance to the masked (airfoil/shadow) region;
     - window-level mean σ_local (regime / angle-of-attack proxy).
  3. Target: log|e|. Fit the conditional **mean |e|** (Laplace b) with L1 loss on |e|, or as a quantile regression at q = 0.5 and 0.8 for a shape check.
  4. Band: h = b̂·ln(1 + σ/(2b̂)), possibly times a global factor c fitted on a held-out pool. Or use the Gaussian table if residuals look Gaussian: check the kurtosis of e/b̂.
  5. **Shift robustness:** give features to the model as within-call ranks, or normalise by the window's median σ_local. Predict b̂/median(σ_local_window) and multiply back, so a 1.95× live scale shift scales b̂ automatically instead of breaking fixed edges.
- **Risk:** overfitting AoA/Re-specific error levels. Use trajectory-grouped CV and score T on FINAL only.

### 2. Direct T-optimising interval head (width-penalised interval learning; RQR / QD / Tube loss)
- **Literature:**
  - Pouplin et al., *Relaxed Quantile Regression*, ICML 2024 (arXiv 2406.03258): learns [l,u] directly with a width-minimising term and no fixed quantiles, and allows asymmetry.
  - Pearce et al. 2018: quality-driven PI loss with a sigmoid-smoothed coverage indicator.
  - *Tube loss*, arXiv 2412.06853.
  - *Deep interval prediction with gradient descent*, arXiv 1911.08160.
  - None optimises exp(-w)·coverage exactly, but all use the same smooth-indicator trick.
- **Assumption:** a smooth surrogate of 1{covered} gives gradients that reach the true optimum. The features must carry the signal.
- **Training:** yes. Same feature table as #1, and a small MLP/GBM with a custom objective. The MLP route is CPU-trivial and needs no residual-distribution assumption.
- **Input-window rule:** yes.
- **Mechanism:** maximises the actual metric end-to-end. It learns asymmetric offsets (bias/mode shift, §0) and non-Gaussian tails automatically.
- **Recipe:**
  - Heads output d_lo, d_up > 0 (softplus), with l = ŷ − d_lo and u = ŷ + d_up.
  - Loss = −mean[ exp(−(d_lo+d_up)/σ) · sig((e+d_lo)/τ) · sig((d_up−e)/τ) ].
  - Anneal τ from 0.3·median|e| down to 0.02·median|e|.
  - Initialise from #1's h so the start is not in the zero-gradient regime.
  - Evaluate with the hard indicator on held-out trajectories.
  - A cheaper variant: keep #1's b̂ and learn only two scalars (c_lo, c_up), or a small per-lead-time table, by grid search on the exact T.

### 3. In-window self-calibration: adaptive/online conformal on a proxy forecaster inside the 20-frame window
- **Literature:**
  - Gibbs & Candès 2021 (ACI) and Angelopoulos et al. 2023 (conformal PID): scale is calibrated online from the series' own recent errors.
  - Stankeviciute et al. 2021 (CF-RNN): horizon-indexed conformal for multi-step forecasts.
  - Normalised nonconformity e/σ̂(x) (Papadopoulos 2008; Lei 2018).
- **Assumption:** how predictable a pixel is **within the window** (e.g. how well persistence or linear extrapolation predicts frames 11–20 from 1–10) is monotonically related to the FNO's future error at that pixel. The ratio is learnable offline.
- **Training:** calibration only. Fit a monotone map (isotonic, or a one-parameter power law) from the in-window proxy error to the FNO residual scale. Runs in seconds.
- **Input-window rule:** yes, fully. It self-adapts per window, so it is inherently robust to the live 1.95× shift and to unseen AoA/Re.
- **Mechanism:** gives a per-window, per-pixel, per-horizon error estimate from the window's own dynamics. It is a direct replacement for σ_local, which measures fluctuation amplitude rather than predictability. A periodic vortex-shedding pixel has large σ_local but may be highly predictable. Frame-to-frame PIV noise does the opposite.
- **Recipe:**
  - For lag k ∈ {1,2,5,10}, compute the persistence proxy error p_k = RMS over t of |x_t − x_{t−k}| per pixel. Optionally add a linear-AR/DMD fit error on frames 1–15 → 16–20.
  - Use p_{k(lead)} as the feature for lead time `lead` (map the FNO lead to proxy lag with a capped lookup).
  - Add these as features to #1. That is the cheapest integration and lets the GBM decide.
  - Standalone: b̂ = g(p), with isotonic g fitted on OOF data; then h via the §0 formula.

### 4. Group-conditional (Mondrian / Kandinsky) split conformal scaling
- **Literature:** Vovk's Mondrian CP; Kandinsky CP (arXiv 2311.11837), which clusters pixels with similar error behaviour and calibrates per cluster; Angelopoulos et al. 2022, image-to-image distribution-free UQ (arXiv 2202.05265), which calibrates a global scalar on heuristic pixel uncertainty maps.
- **Assumption:** exchangeability within groups (lead-time bin × σ̂ bin × channel × region).
- **Training:** none. You only compute per-group statistics on held-out trajectories.
- **Input-window rule:** yes, provided the group keys are computable from x and ŷ.
- **Mechanism:** fixes per-group scale bias. With T as the target, fit per group **the scale c_g maximising exact T**, not a coverage quantile. This is a generalisation of the current sextile table: add the lead-time axis and a spatial-cluster axis.
- **Recipe:** groups = 20 leads × 6 within-call rank bins × 2 channels × 4 spatial k-means clusters (clustered on OOF |e| maps). Grid search c_g ∈ [0.2, 2] on the SELECT pool, shrink toward the group-marginal value, and verify on FINAL. This is the fastest win, runnable today.

### 5. Conformalised quantile regression (CQR) on top of quantile heads
- **Literature:** Romano et al. 2019; *Calibrated UQ for Operator Learning via Conformal Prediction* (Ma, Pitt, Azizzadenesheli, Anandkumar, TMLR 2024, arXiv 2402.01960), a risk-controlling quantile neural operator with calibration of function-level pointwise coverage; *Split conformal in function space with neural operators* (arXiv 2509.04623).
- **Assumption:** that the target coverage level is known in advance. Here it is not: it varies per element from 0.2 to 0.97 (§0 table). You must choose the quantile per element, which collapses back to #1/#2.
- **Verdict:** use the quantile *heads* (q = 0.5, 0.75, 0.9 of |e|) as features/shape information for #1. Do **not** use a CQR coverage target. The risk-controlling quantile NO requires training an FNO-sized model and has no advantage for this metric.

### 6. Heteroscedastic / noise-injected neural-operator heads trained end-to-end
- **Literature:** 2602.11090 (cross-regularised FNO uncertainty); LE-PDE-UQ (AAAI 2024, the same Westlake group); deep ensembles.
- **Assumption:** retraining or augmenting the operator on ~70 real trajectories does not hurt the point forecast, and the variance head generalises.
- **Training:** heavy (GPU/Kaggle). It is also risky: the prior `.dist_hetero_nomod` attempt exists in the repo.
- **Verdict:** lowest priority. #1 with a GBM on engineered features gets most of the value on CPU. A 2-member ensemble disagreement |ŷ_FNO − ŷ_corrector| is a cheap free feature, though: include it in #1.

**RealPDEBench upstream:** the ICLR 2026 repo (github.com/AI4Science-WestlakeU/RealPDEBench) has **no** UQ, interval or conformal baseline. Its 9 metrics are RMSE, MAE, Rel L2, R², Update Ratio, fRMSE, FE, KE and MVPE. The competition's Safe Prediction Score is new. The group's related UQ work (SafeDiffCon, ICML 2025, arXiv 2502.02205; LE-PDE-UQ) targets conformal coverage guarantees, not an exp-width objective. None of it uses test labels, but none of it transfers directly.

**Rejected:**
- Anything needing cross-window pooling at test time: transductive/full conformal over the test set, or test-time ACI across windows. Each window runs in an isolated process.
- External pretrained UQ models.

## Suggested order
1. #4, with a lead-time axis added to the existing rank table. Hours, zero training.
2. #1 plus #3's proxy-error features: GBM on log|e| with the Laplace h* formula.
3. #2 fine-tuning of (c_lo, c_up, per-lead) on exact T, if residuals are skewed.

Validate all on the FINAL pool only, using grouped-by-trajectory CV.
