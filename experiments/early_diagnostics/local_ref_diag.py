#!/usr/bin/env python3
"""Local reference for the four Step-1 live diagnostics, on real train_real
data, using IDENTICAL bin definitions -- so gains/mismatches are measured
directly, not guessed. This is the check Step 4 needs before building any
augmentation: which of {spectrum, noise correlation, mask structure} actually
differ from live, and which are already matched?

Pure numpy on raw .h5 windows -- no model forward needed, since all four
diagnostics are properties of the INPUT DATA's fluctuation field, not of any
prediction (confirmed: _step1_diagnostics computes everything from xb alone).
"""
import os, glob
import numpy as np, h5py

ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
EXCLUDE = {"7575_0"}
T_IN, SUB = 20, 2

# regime pool: cluster-1 trajectories (AoA 10/15/20, Re 16.5k-26.7k), the same
# pool used throughout this session for the honest local/live comparison.
POOL = ['21600_15','22875_15','24150_15','20325_15','24150_10','19050_15',
        '20325_20','22875_10','19050_20','25425_15','25425_10','21600_10',
        '17775_20','26700_10','21600_20','16500_20','26700_15','24150_20','25425_20']


def _mask_components(mask2d):
    ys, xs = np.nonzero(mask2d)
    n = len(ys)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    idx = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy, dx in ((0, 1), (1, 0)):
            j = idx.get((int(y) + dy, int(x) + dx))
            if j is not None:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    return np.unique(roots, return_counts=True)[1]


specs_u, specs_v, sighists, acs, maskhists = [], [], [], [], []
n_windows = 0
for t in POOL:
    p = os.path.join(ROOT, t + ".h5")
    if not os.path.exists(p) or t in EXCLUDE:
        continue
    with h5py.File(p) as f:
        u = np.asarray(f["u"][:, ::SUB, ::SUB], np.float32)
        v = np.asarray(f["v"][:, ::SUB, ::SUB], np.float32)
    T = u.shape[0]
    for i in range(0, T - T_IN + 1, T_IN):     # non-overlapping windows
        wu, wv = u[i:i+T_IN], v[i:i+T_IN]
        flu = wu - wu.mean(0, keepdims=True)
        flv = wv - wv.mean(0, keepdims=True)
        n_windows += 1

        # 1. radial spectrum, identical bin logic to the live diagnostic
        Fu = np.fft.rfft2(flu, axes=(1, 2))
        Fv = np.fft.rfft2(flv, axes=(1, 2))
        Pu = (np.abs(Fu) ** 2).mean(0)
        Pv = (np.abs(Fv) ** 2).mean(0)
        h, wr = Pu.shape
        kh = np.minimum(np.arange(h), h - np.arange(h))
        kw = np.arange(wr)
        rad = np.sqrt(kh[:, None].astype(float) ** 2 + kw[None, :].astype(float) ** 2)
        rmax = rad.max() + 1e-6
        edges = [0, rmax * 0.25, rmax * 0.5, rmax * 0.75, rmax + 1]
        specs_u.append([float(Pu[(rad >= edges[j]) & (rad < edges[j+1])].sum()) for j in range(4)])
        specs_v.append([float(Pv[(rad >= edges[j]) & (rad < edges[j+1])].sum()) for j in range(4)])

        # 2. sigma_local histogram, u channel, SAME edges as live
        sl = np.sqrt((flu ** 2).mean(0))
        edges_s = [0.0, 0.003, 0.008, 0.016, 0.028, 0.045, 1.0]
        hist, _ = np.histogram(sl.ravel(), bins=edges_s)
        sighists.append(hist / max(hist.sum(), 1))

        # 3. mask connected components, last frame, SAME 5 bins
        mask2d = (wu[-1] == 0) & (wv[-1] == 0)
        sizes = _mask_components(mask2d)
        cc = np.histogram(sizes, bins=(1, 2, 4, 8, 16, 100000))[0] if sizes.size else np.zeros(5, int)
        maskhists.append(cc)

        # 4. autocorrelation lags 1-3, u channel
        var = (flu ** 2).mean(0).clip(min=1e-9)
        row = []
        for lag in (1, 2, 3):
            cov = (flu[:-lag] * flu[lag:]).mean(0)
            row.append(float((cov / var).mean()))
        acs.append(row)

specs_u = np.array(specs_u); specs_v = np.array(specs_v)
sighists = np.array(sighists); acs = np.array(acs)
maskhists = np.array(maskhists)

print(f"local reference: {n_windows} windows from {len(POOL)} cluster-1 trajectories\n")

LIVE_SPEC_U = np.array([1477.5, 129.3, 30.0, 14.7])
LIVE_SPEC_V = np.array([367.9, 16.7, 4.2, 2.0])
LIVE_SIGHIST = np.array([0.252, 0.281, 0.151, 0.121, 0.097, 0.098])
LIVE_MASKCC = np.array([0.82, 0.18, 0.12, 0.18, 1.18])
LIVE_AC = np.array([0.686, 0.417, 0.146])

print("=== 1. RADIAL SPECTRUM ===")
lu, lv = specs_u.mean(0), specs_v.mean(0)
print(f"local  SPEC_u {lu.round(1).tolist()}   live {LIVE_SPEC_U.tolist()}")
print(f"local  SPEC_v {lv.round(1).tolist()}   live {LIVE_SPEC_V.tolist()}")
print(f"per-bin gain (live/local), u: {(LIVE_SPEC_U/lu).round(3).tolist()}")
print(f"per-bin gain (live/local), v: {(LIVE_SPEC_V/lv).round(3).tolist()}")

print("\n=== 2. SIGMA_LOCAL HISTOGRAM ===")
ls = sighists.mean(0)
print(f"local  {ls.round(3).tolist()}")
print(f"live   {LIVE_SIGHIST.tolist()}")
print(f"per-bin gain (live/local): {(LIVE_SIGHIST/np.clip(ls,1e-6,None)).round(3).tolist()}")

print("\n=== 3. MASK CONNECTED-COMPONENTS ===")
lm = maskhists.mean(0)
print(f"local  mean {lm.round(3).tolist()}")
print(f"live   mean {LIVE_MASKCC.tolist()}")
print(f"local frac. w/ a 16+ blob: {(maskhists[:,4]>0).mean():.3f}  (live: 1.00)")
print(f"local frac. w/ ANY sub-16 fragment: {(maskhists[:,:4].sum(1)>0).mean():.3f}  (live: 0.47)")

print("\n=== 4. AUTOCORRELATION ===")
la = acs.mean(0)
print(f"local  lags 1/2/3: {la.round(3).tolist()}")
print(f"live   lags 1/2/3: {LIVE_AC.tolist()}")
print(f"local AR(1)-implied from lag1={la[0]:.3f}: lag2={la[0]**2:.3f} lag3={la[0]**3:.3f}")
print(f"local decay ratio vs AR(1) prediction: lag2 {la[1]/la[0]**2:.3f}  lag3 {la[2]/la[0]**3:.3f}")
