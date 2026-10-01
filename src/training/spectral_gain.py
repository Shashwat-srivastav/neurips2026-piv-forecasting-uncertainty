#!/usr/bin/env python3
"""Corrected spectral augmentation (masterplan v11 sec.3): scale the input
window's fluctuation field per radial frequency bin by sqrt(power_ratio), not
the power_ratio itself -- the measured table is a POWER ratio, and FFT
amplitudes scale as sqrt(power). Applied to the INPUT only; target untouched,
same restriction as augment_input() in finetune_fno_kaggle.py.

Same radial-bin definition as local_ref_diag.py / the live Step-1 diagnostic,
so the verification gate below is checking against the same quantity the
gain table was measured in.
"""
import numpy as np

# masterplan v11 sec.3 table -- POWER ratios, live/local, 4 radial bins.
SPEC_U_POWER_GAIN = np.array([1.878, 1.602, 1.419, 1.381])
SPEC_V_POWER_GAIN = SPEC_U_POWER_GAIN          # "same for v" per the plan

AMP_GAIN_U = np.sqrt(SPEC_U_POWER_GAIN)
AMP_GAIN_V = np.sqrt(SPEC_V_POWER_GAIN)

LIVE_SPEC_U = np.array([1477.5, 129.3, 30.0, 14.7])
LIVE_SPEC_V = np.array([367.9, 16.7, 4.2, 2.0])


def radial_bin_index(h, w_rfft):
    """(h, w_rfft) int array, values in {0,1,2,3} -- same edges as the live
    diagnostic and local_ref_diag.py."""
    kh = np.minimum(np.arange(h), h - np.arange(h))
    kw = np.arange(w_rfft)
    rad = np.sqrt(kh[:, None].astype(float) ** 2 + kw[None, :].astype(float) ** 2)
    rmax = rad.max() + 1e-6
    edges = [0, rmax * 0.25, rmax * 0.5, rmax * 0.75, rmax + 1]
    idx = np.zeros(rad.shape, dtype=np.int64)
    for j in range(4):
        idx[(rad >= edges[j]) & (rad < edges[j + 1])] = j
    return idx


def _inpaint_nearest(frame, valid):
    """Fill invalid cells with the value at the nearest valid cell, per frame.
    A single connected occlusion blob (per sec.13.2) makes this cheap and
    well-posed -- no extrapolation past the blob boundary."""
    if valid.all():
        return frame
    from scipy.ndimage import distance_transform_edt
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    return frame[iy, ix]


def apply_spectral_gain(x, amp_gain_u=AMP_GAIN_U, amp_gain_v=AMP_GAIN_V):
    """x: (T, H, W, >=2) float32, channels [u, v, ...]. Returns a copy with
    the fluctuation field's radial spectrum scaled by amp_gain (amplitude,
    not power), masked cells restored exactly, phase preserved (real
    positive per-bin multiplier only)."""
    x = x.copy()
    u, v = x[..., 0], x[..., 1]
    T, H, W = u.shape
    valid = (u != 0.0) | (v != 0.0)            # (T, H, W)

    ui = np.stack([_inpaint_nearest(u[t], valid[t]) for t in range(T)])
    vi = np.stack([_inpaint_nearest(v[t], valid[t]) for t in range(T)])

    mu, mv = ui.mean(0, keepdims=True), vi.mean(0, keepdims=True)
    flu, flv = ui - mu, vi - mv

    Fu = np.fft.rfft2(flu, axes=(1, 2))
    Fv = np.fft.rfft2(flv, axes=(1, 2))
    bidx = radial_bin_index(*Fu.shape[1:])
    gu = amp_gain_u[bidx][None]                 # (1, H, Wf) broadcast over T
    gv = amp_gain_v[bidx][None]
    Fu *= gu
    Fv *= gv

    flu_aug = np.fft.irfft2(Fu, s=(H, W), axes=(1, 2))
    flv_aug = np.fft.irfft2(Fv, s=(H, W), axes=(1, 2))
    u_aug = (flu_aug + mu).astype(np.float32)
    v_aug = (flv_aug + mv).astype(np.float32)

    u_aug = np.where(valid, u_aug, u)           # restore masked cells exactly
    v_aug = np.where(valid, v_aug, v)
    x[..., 0], x[..., 1] = u_aug, v_aug
    return x


def measure_radial_power(windows):
    """windows: iterable of (T,H,W,>=2) float32 arrays. Returns (SPEC_u,
    SPEC_v) mean 4-bin power, identical definition to local_ref_diag.py."""
    su, sv, n = np.zeros(4), np.zeros(4), 0
    for x in windows:
        u, v = x[..., 0], x[..., 1]
        flu, flv = u - u.mean(0, keepdims=True), v - v.mean(0, keepdims=True)
        Fu = np.fft.rfft2(flu, axes=(1, 2))
        Fv = np.fft.rfft2(flv, axes=(1, 2))
        Pu, Pv = (np.abs(Fu) ** 2).mean(0), (np.abs(Fv) ** 2).mean(0)
        bidx = radial_bin_index(*Pu.shape)
        for j in range(4):
            su[j] += Pu[bidx == j].sum()
            sv[j] += Pv[bidx == j].sum()
        n += 1
    return su / max(n, 1), sv / max(n, 1)


def verify_gate(windows, tol=0.15, live_u=LIVE_SPEC_U, live_v=LIVE_SPEC_V):
    """Post-augmentation verification gate (sec.3): each of the 4 radial bins
    must match live within +-tol on BOTH channels. Returns (passed, report)."""
    su, sv = measure_radial_power(windows)
    ru, rv = su / live_u, sv / live_v
    ok_u = np.all(np.abs(ru - 1.0) <= tol)
    ok_v = np.all(np.abs(rv - 1.0) <= tol)
    report = (f"SPEC_u aug/live ratio per bin: {ru.round(3).tolist()}  "
              f"({'PASS' if ok_u else 'FAIL'}, tol {tol})\n"
              f"SPEC_v aug/live ratio per bin: {rv.round(3).tolist()}  "
              f"({'PASS' if ok_v else 'FAIL'}, tol {tol})")
    return bool(ok_u and ok_v), report


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, W = 32, 64
    x = np.zeros((20, H, W, 3), np.float32)
    x[..., 0] = 0.15 + 0.01 * rng.standard_normal((20, H, W))
    x[..., 1] = -0.0005 + 0.002 * rng.standard_normal((20, H, W))
    x[5:8, 10:14, 20:26, :2] = 0.0             # a fake occlusion blob
    y = apply_spectral_gain(x)
    assert y.shape == x.shape
    assert np.array_equal(y[5:8, 10:14, 20:26, :2], x[5:8, 10:14, 20:26, :2]), \
        "masked cells must be restored exactly"
    print("apply_spectral_gain smoke test: OK, masked cells preserved")
    ok, rep = verify_gate([apply_spectral_gain(x) for _ in range(20)])
    print(rep)
