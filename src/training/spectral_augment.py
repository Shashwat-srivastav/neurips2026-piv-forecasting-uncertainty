#!/usr/bin/env python3
"""Spectral (radial-band) amplitude augmentation, matched to the live/local
gap measured AoA-15-only (CAMPAIGN_REPORT.md 13.5).

THE BUG THIS FIXES: the measured table is a POWER ratio (mean |FFT|^2 per
radial bin). Multiplying FFT COEFFICIENTS directly by a power ratio squares
the intended effect -- amplitude scales as sqrt(power). This was caught before
any training code was written; the correct multiplier is:

    amplitude_gain_b = sqrt(live_power_b / local_power_b)

AoA-15 gains (13.5), power -> amplitude:
    SPEC_u  bin0-3  power [1.878,1.602,1.419,1.381] -> amp [1.371,1.266,1.191,1.175]
    SPEC_v  bin0-3  power [1.751,1.380,1.458,1.483] -> amp [1.323,1.175,1.207,1.218]

PROCEDURE (per the spec):
  1. Inpaint masked cells (exactly-zero, from PIV dropout) before the FFT --
     otherwise the mask's hard edge injects high-frequency content that has
     nothing to do with turbulence and gets amplified along with everything
     else at that scale.
  2. rfft2 of the FLUCTUATION field (input minus its own temporal mean) --
     gains were measured on fluctuation spectra, not the raw field.
  3. Multiply each radial bin by its (real, positive) amplitude gain --
     phase is preserved automatically since the multiplier is real.
  4. irfft2, add the mean back.
  5. Restore the ORIGINAL masked cells to exactly zero.

VERIFICATION GATE: recompute the 4 radial power bins on the augmented pool and
require each within +/-15% of the live target before this is used for
training. Do not skip this -- it is what catches a sign, indexing or
inpainting error before it costs a training run.
"""
from __future__ import annotations

import numpy as np

# POOL-SPECIFIC amplitude gains (18 Aug v2): the AoA-15 POOLED average gain
# (kept below as AMP_GAIN_U_POOLED_AVG for reference) OVERSHOOTS when applied
# to the actual training pool -- the 11 most live-similar AoA-15 trajectories
# are already closer to live than the AoA-15 average, so they need LESS
# correction. Verified live on the actual training pool (live_similar_pool.npy,
# 473 windows, 11 trajectories): local SPEC_u [880.4,89.3,23.4,11.8], SPEC_v
# [240.7,12.8,3.1,1.5] -> these gains, not the pooled-average ones.
AMP_GAIN_U = np.array([1.295, 1.203, 1.132, 1.114], dtype=np.float32)
AMP_GAIN_V = np.array([1.236, 1.142, 1.159, 1.157], dtype=np.float32)

# AoA-15 POOLED AVERAGE gains (13.5) -- do NOT use directly on the live-similar
# training pool; kept only for reference / for augmenting OTHER, more-distant
# pools where the pooled average is the better estimate.
AMP_GAIN_U_POOLED_AVG = np.array([1.371, 1.266, 1.191, 1.175], dtype=np.float32)
AMP_GAIN_V_POOLED_AVG = np.array([1.323, 1.175, 1.207, 1.218], dtype=np.float32)

LIVE_POWER_U = np.array([1477.5, 129.3, 30.0, 14.7], dtype=np.float32)
LIVE_POWER_V = np.array([367.9, 16.7, 4.2, 2.0], dtype=np.float32)


def _radial_bins(h, w):
    """Same 4-bin radial partition as the diagnostic: edges at
    0/.25/.5/.75/1.0 of the max radius, on an rfft2 grid of shape (h, w)."""
    kh = np.minimum(np.arange(h), h - np.arange(h))
    kw = np.arange(w)
    rad = np.sqrt(kh[:, None].astype(float) ** 2 + kw[None, :].astype(float) ** 2)
    rmax = rad.max() + 1e-6
    edges = [0, rmax * 0.25, rmax * 0.5, rmax * 0.75, rmax + 1]
    return rad, edges


def _inpaint(frame, mask):
    """Fill masked (True) cells with the mean of their valid orthogonal
    neighbours, iterated until stable or a cap -- a simple, fast diffusion
    inpaint. Cheap: masked fraction is ~9-11% of a 32x64 grid."""
    out = frame.copy()
    m = mask.copy()
    for _ in range(8):
        if not m.any():
            break
        acc = np.zeros_like(out); cnt = np.zeros_like(out)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            shifted = np.roll(out, (dy, dx), axis=(0, 1))
            valid_src = np.roll(~m, (dy, dx), axis=(0, 1))
            acc += np.where(valid_src, shifted, 0.0)
            cnt += valid_src.astype(out.dtype)
        fillable = m & (cnt > 0)
        out[fillable] = acc[fillable] / cnt[fillable]
        m = m & ~fillable
    if m.any():                       # fully-isolated leftovers: global mean
        out[m] = out[~mask].mean() if (~mask).any() else 0.0
    return out


def spectral_boost(window, gain_u=AMP_GAIN_U, gain_v=AMP_GAIN_V):
    """window: (T, H, W, >=2) float32, channels [u, v, ...]. Returns a new
    array, same shape/dtype, with the input-only spectral boost applied.
    Target's untouched -- this function is never called on the target."""
    T, H, W = window.shape[:3]
    out = window.copy()
    mean = window[..., :2].mean(axis=0, keepdims=True)
    fluct = window[..., :2] - mean
    mask = (window[..., 0] == 0.0) & (window[..., 1] == 0.0)   # (T,H,W)

    for c, gain in ((0, gain_u), (1, gain_v)):
        ch = fluct[..., c].copy()
        for t in range(T):
            if mask[t].any():
                ch[t] = _inpaint(ch[t], mask[t])
        F = np.fft.rfft2(ch, axes=(1, 2))
        rad, edges = _radial_bins(F.shape[1], F.shape[2])
        g = np.ones_like(rad, dtype=np.float32)
        for j in range(4):
            g[(rad >= edges[j]) & (rad < edges[j + 1])] = gain[j]
        F = F * g[None, :, :]
        boosted = np.fft.irfft2(F, s=(H, W), axes=(1, 2))
        boosted = boosted + mean[0, ..., c]
        boosted[mask] = 0.0             # restore the original mask exactly
        out[..., c] = boosted
    return out


def verify_gate(windows, target_power_u=LIVE_POWER_U, target_power_v=LIVE_POWER_V,
                tol=0.15):
    """windows: list/array of (T,H,W,>=2) ALREADY-AUGMENTED arrays. Recomputes
    the 4 radial power bins and checks each is within `tol` of the live
    target. Returns (ok: bool, measured_u, measured_v, ratios_u, ratios_v)."""
    Pu_acc = np.zeros(4); Pv_acc = np.zeros(4); n = 0
    for w in windows:
        fl = w[..., :2] - w[..., :2].mean(axis=0, keepdims=True)
        Fu = np.fft.rfft2(fl[..., 0], axes=(1, 2))   # H,W are axes 1,2 of (T,H,W,2)
        Fv = np.fft.rfft2(fl[..., 1], axes=(1, 2))
        Pu = (np.abs(Fu) ** 2).mean(0); Pv = (np.abs(Fv) ** 2).mean(0)
        rad, edges = _radial_bins(Pu.shape[0], Pu.shape[1])
        for j in range(4):
            m = (rad >= edges[j]) & (rad < edges[j + 1])
            Pu_acc[j] += float(Pu[m].sum()); Pv_acc[j] += float(Pv[m].sum())
        n += 1
    Pu_acc /= max(n, 1); Pv_acc /= max(n, 1)
    ru = target_power_u / np.clip(Pu_acc, 1e-9, None)
    rv = target_power_v / np.clip(Pv_acc, 1e-9, None)
    ok = bool(np.all(np.abs(ru - 1) <= tol) and np.all(np.abs(rv - 1) <= tol))
    return ok, Pu_acc, Pv_acc, ru, rv


if __name__ == "__main__":
    # Self-test on real data: apply the boost to real AoA-15 windows, verify
    # the gate, and print the before/after so a sign or indexing error is
    # visible immediately rather than discovered after a training run.
    import glob, os, h5py

    ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
    files = sorted(glob.glob(os.path.join(ROOT, "*_15.h5")))[:4]
    raw, boosted = [], []
    for p in files:
        with h5py.File(p) as f:
            u = np.asarray(f["u"][:20, ::2, ::2], np.float32)
            v = np.asarray(f["v"][:20, ::2, ::2], np.float32)
        w = np.stack([u, v, np.zeros_like(u)], -1)
        raw.append(w)
        boosted.append(spectral_boost(w))

    print("BEFORE augmentation (should NOT already match live):")
    ok0, pu0, pv0, ru0, rv0 = verify_gate(raw)
    print(f"  measured SPEC_u {pu0.round(1).tolist()}  gain-to-live {ru0.round(3).tolist()}")
    print(f"  measured SPEC_v {pv0.round(1).tolist()}  gain-to-live {rv0.round(3).tolist()}")

    print("\nAFTER augmentation (gate: each ratio within +/-15% of 1.0):")
    ok1, pu1, pv1, ru1, rv1 = verify_gate(boosted)
    print(f"  measured SPEC_u {pu1.round(1).tolist()}  gain-to-live {ru1.round(3).tolist()}")
    print(f"  measured SPEC_v {pv1.round(1).tolist()}  gain-to-live {rv1.round(3).tolist()}")
    print(f"\nmasked cells preserved exactly: "
          f"{all(bool((( (b[...,0]==0)&(b[...,1]==0) ) == ((r[...,0]==0)&(r[...,1]==0))).all()) for b,r in zip(boosted,raw))}")
    print(f"GATE: {'PASS' if ok1 else 'FAIL'}")
