# TKE deficit of stock `sim_real_fno`: diagnosis and a post-hoc fix (25 Sep)

Scripts: `diag_tke.py`, `eval_rescale.py`, `phase_skill.py`, `per_traj.py` (outputs in the `*_out.txt` files next to them).
Data: `experiments/interval_recipes/cache_{select,final}_s20.npz` (v8 fp16 FNO predictions on the holdout, 168 + 420 windows).
Scorer: official v9 `scoring.py`, run with the shipped v8 band (`experiments/interval_recipes/common.py`).
**The holdout is memorized (C13).** The absolute levels here are training scores, so only the deltas mean anything.

## 1. Diagnosis: the error is mostly amplitude, not position or phase

| | SELECT | FINAL |
|---|---|---|
| tke error / score | 0.503 / 79.9 | 0.468 / 81.0 |
| energy ratio ΣKE_pred/ΣKE_tgt (median) | **0.57** | **0.62** |
| spatial corr(KE_pred, KE_tgt) (median) | **0.89** | **0.90** |
| tke score with an oracle per-window scalar on KE_pred | 82.9 (+2.9) | 83.8 (+2.8) |
| input-window energy ΣKE_X/ΣKE_tgt (median) | 0.99 | 1.00 |
| share of tke sq-error on partially-masked pixels | 0.1% | 0.0% |

- **Amplitude.** The variance map has the right shape (ρ≈0.9) but only about 60% of the energy. The flow is statistically stationary: the input window's energy matches the target's (0.99–1.00), which gives a target-free estimate of the correct amplitude.
- **Temporal.** The pred/target fluctuation-energy ratio decays with lead time: FINAL goes 0.83→0.65 over leads 1-10 and →0.50 at lead 20, and SELECT goes 0.71→0.38. The fluctuation pattern correlation stays at 0.85→0.71. The phase is mostly kept, so this is **MSE damping of amplitude, not phase loss**. The rel-optimal fluctuation gain ⟨f_p,f_y⟩/⟨f_p,f_p⟩ is 0.95–1.02 on FINAL and 1.01–1.14 on SELECT: in rel terms the model is roughly calibrated, but its variance is too small.
- **Region.** The deficit is worst near the body and at the wake edges (ratio 0.15–0.40), but 65% of the target KE sits in the mid/downstream wake core, where the ratio is 0.63–0.76. The tke error is therefore driven by the wake core.
- **Masking.** PIV dropout contributes nothing to the tke error.

## 2. Phase-consistent extrapolation: dropped

The input autocorrelation (spatial mean, SELECT) is 1, .87, .59, .23, −.13, −.39, −.54, −.59 (lag 7), −.46, −.19, and **≈0 from lag 13 on**. The oscillation in these windows has a period of about 14 frames and decorrelates within about one period. By the time we reach forecast leads there is no phase memory left in the input, so DMD or FFT extrapolation of the input carries no skill there. This agrees with §12.6 (DMD blend closed).

## 3. Candidate: mean-preserving fluctuation gain, with the band kept on the stock centre

`P' = P̄ + a·(P − P̄)`, where P̄ is the prediction's own 20-frame mean per pixel. After scaling, P' is re-centred so that its temporal mean equals P̄ exactly. Consequences:
- **mvpe is invariant by construction**, because the probes use temporal means.
- KE scales by a².
- rel pays `(a−1)²‖f_p‖² − 2(a−1)⟨f_p, y−p⟩`.

The **SPS bounds stay `P_stock ± h_v8`**. The README only requires finite bounds with lower ≤ upper; the point prediction does not have to lie inside them.

Two gain variants:
- **G**: a global constant `a`.
- **I** (input-anchored): `a_w = clip((ΣKE(X)/ΣKE(P))^(γ/2), 1, 1.6)` per window. It uses only the input and the prediction.

Candidates were selected on SELECT and scored on FINAL once. Pre-registration is in `PREREG.txt`. Composite = partial weights .476/.086/.095/.243 applied to rel/tke/mvpe/sps.

| candidate | set | Δrel | Δtke | Δmvpe | Δsps | Δcomp |
|---|---|---|---|---|---|---|
| G a=1.15, band on stock P (pre-reg primary) | SELECT | −0.009 | +2.183 | 0 | +0.584 | +0.325 |
| | **FINAL** | **−0.069** | **+1.376** | **0.000** | **+0.275** | **+0.152** |
| I γ=0.5, band on stock P (pre-reg secondary) | SELECT | −0.016 | +2.200 | 0 | +0.582 | +0.323 |
| | **FINAL** | **−0.053** | **+1.619** | **0.000** | **+0.391** | **+0.209** |
| G a=1.15, band re-centred on P' | FINAL | −0.069 | +1.376 | 0 | **−0.687** | −0.082 |

Other results:
- The lead-dependent gain was no better than G at γ=0.5 and worse at γ=1.
- Regional gains were no better than G or I.
- Gains around 1.3 overshoot on both sets. The tke optimum is at a≈1.15, well below full energy matching (≈1.3), because the per-pixel L2 penalises boosting noise in the shape.

**Per trajectory on FINAL**, the benefit is set by `vr` = std(pred fluct)/std(input fluct):

| trajectories | vr | G a=1.15 | I γ=0.5 |
|---|---|---|---|
| 8 of 10 (AoA 10/15, one AoA 20) | 0.76–0.81 | Δtke +1.4 to +2.9, Δrel −0.01 to −0.06 (one AoA 20: −0.18) | about the same |
| 19050_20, 21600_20 (AoA 20) | 0.92 | Δtke −1.6 to −1.8 | Δtke about −0.2 to 0 |

I γ=0.5 is nearly neutral on the high-vr trajectories because it adapts per window. **Live vr is 0.697 (§12.2), below every local trajectory**, so the live flow falls in the regime where the boost helps most. I γ=0.5 would apply a≈1.2 there.

## 4. Why this is not a closed direction

- **C02 corrector.** The corrector was a learned additive field that moved the temporal mean, costing mvpe −0.19 locally. Its bands were also centred on the corrected field. From the ledger, v4 vs v8 live was rel −0.148 (−0.070 comp), tke +1.70 (+0.146), and **sps −0.789 (−0.192)**, so **the live loss came mostly from SPS**. The last row of the table above reproduces that SPS loss exactly when the band moves with P'. This candidate differs in three ways:
  - mvpe cannot move;
  - the band is not moved;
  - the tke:rel trade is 20–30:1, against the corrector's 12:1 locally and 11.5:1 live.
- **λ (§12.7, ELITE §7 "λ=1 optimal").** Only the shrink side is documented, and ELITE §2 says the λ-shrink rested on a wrong assumption. There is no script or table for λ>1 with the band kept fixed.
- **C01/C03.** These were fine-tunes. Nothing here is trained.

## 5. Verdict

- **Deterministic, 1–2 constants, zero training, zero GPU, negligible compute.** Locally it is a real and consistent win (8/10 trajectories), but it is **not rel-neutral** (rel −0.05 to −0.07) and it is **+1.4 to +1.6 tke on FINAL, not "several points"**. The local ceiling for any amplitude-only fix is +2.8 (oracle scalar). The rest of the gap to the top teams at ~79.5 is shape and noise that a post-hoc gain cannot reach. The 20-sample tke target caps near about 80 (§6.2).
- **Main live risk.** The holdout is memorized. On unseen data the model's fluctuation–truth correlation is lower, and the rel-optimal gain could fall below 1, which would raise the rel cost. The corrector's local→live transfer was about 63% on rel and tke. The favourable live vr (0.697) points the other way on tke.
- **Recommendation.** If a slot is free (26/27 Sep), ship **I γ=0.5, band on stock P**. It is a small edit to v8's `predict()`: compute `a_w` from the input and the prediction, rescale only `prediction`, and leave `lower`/`upper` on the stock prediction. Run `src/evaluation/preflight.py` first. It is safe to submit (mvpe fixed, bounded gain ≤1.6, no crash surface) and worth a slot only if nothing better is ready. Expect a modest positive, not a jump to the top 10.
