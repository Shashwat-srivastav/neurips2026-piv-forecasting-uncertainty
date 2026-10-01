#!/usr/bin/env python3
"""Full live/local ratio table, AoA-15 ONLY (all 17 trajectories, descriptive
-- not a fitting pool, so no holdout exclusion needed here). Same methodology
and bin definitions as the live diagnostic, computed on 2026-08-18."""
import os, glob
import numpy as np, h5py

ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
EXCLUDE = {"7575_0"}
T_IN, SUB = 20, 2


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
                if ri != rj: parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    return np.unique(roots, return_counts=True)[1]


files = sorted(f for f in glob.glob(os.path.join(ROOT, "*_15.h5"))
               if os.path.splitext(os.path.basename(f))[0] not in EXCLUDE)
print(f"AoA-15 trajectories: {len(files)}")

specs_u, specs_v, sighists, acs, maskhists = [], [], [], [], []
n_windows = 0
for p in files:
    with h5py.File(p) as f:
        u = np.asarray(f["u"][:, ::SUB, ::SUB], np.float32)
        v = np.asarray(f["v"][:, ::SUB, ::SUB], np.float32)
    T = u.shape[0]
    for i in range(0, T - T_IN + 1, T_IN):
        wu, wv = u[i:i+T_IN], v[i:i+T_IN]
        flu = wu - wu.mean(0, keepdims=True)
        flv = wv - wv.mean(0, keepdims=True)
        n_windows += 1

        Fu = np.fft.rfft2(flu, axes=(1, 2)); Fv = np.fft.rfft2(flv, axes=(1, 2))
        Pu = (np.abs(Fu) ** 2).mean(0); Pv = (np.abs(Fv) ** 2).mean(0)
        h, wr = Pu.shape
        kh = np.minimum(np.arange(h), h - np.arange(h)); kw = np.arange(wr)
        rad = np.sqrt(kh[:, None].astype(float) ** 2 + kw[None, :].astype(float) ** 2)
        rmax = rad.max() + 1e-6
        edges = [0, rmax * 0.25, rmax * 0.5, rmax * 0.75, rmax + 1]
        specs_u.append([float(Pu[(rad >= edges[j]) & (rad < edges[j+1])].sum()) for j in range(4)])
        specs_v.append([float(Pv[(rad >= edges[j]) & (rad < edges[j+1])].sum()) for j in range(4)])

        sl = np.sqrt((flu ** 2).mean(0))
        edges_s = [0.0, 0.003, 0.008, 0.016, 0.028, 0.045, 1.0]
        hist, _ = np.histogram(sl.ravel(), bins=edges_s)
        sighists.append(hist / max(hist.sum(), 1))

        mask2d = (wu[-1] == 0) & (wv[-1] == 0)
        sizes = _mask_components(mask2d)
        cc = np.histogram(sizes, bins=(1, 2, 4, 8, 16, 100000))[0] if sizes.size else np.zeros(5, int)
        maskhists.append(cc)

        var = (flu ** 2).mean(0).clip(min=1e-9)
        row = []
        for lag in (1, 2, 3):
            cov = (flu[:-lag] * flu[lag:]).mean(0)
            row.append(float((cov / var).mean()))
        acs.append(row)

specs_u = np.array(specs_u); specs_v = np.array(specs_v)
sighists = np.array(sighists); acs = np.array(acs); maskhists = np.array(maskhists)
print(f"total AoA-15 windows: {n_windows}\n")

LIVE_SPEC_U = np.array([1477.5, 129.3, 30.0, 14.7])
LIVE_SPEC_V = np.array([367.9, 16.7, 4.2, 2.0])
LIVE_SIGHIST = np.array([0.252, 0.281, 0.151, 0.121, 0.097, 0.098])
LIVE_MASKCC = np.array([0.82, 0.18, 0.12, 0.18, 1.18])
LIVE_AC = np.array([0.686, 0.417, 0.146])

lu, lv, ls, lm, la = (specs_u.mean(0), specs_v.mean(0), sighists.mean(0),
                      maskhists.mean(0), acs.mean(0))

print("=" * 78)
print("FULL LIVE/LOCAL RATIO TABLE -- AoA-15 ONLY, all 17 trajectories")
print("=" * 78)
print(f"\n{'bin':<28}{'local':>12}{'live':>12}{'gain(live/local)':>20}")
names_spec = [f"SPEC_u bin{i}" for i in range(4)]
for nm, l, v in zip(names_spec, lu, LIVE_SPEC_U):
    print(f"{nm:<28}{l:12.1f}{v:12.1f}{v/l:20.3f}")
names_spec_v = [f"SPEC_v bin{i}" for i in range(4)]
for nm, l, v in zip(names_spec_v, lv, LIVE_SPEC_V):
    print(f"{nm:<28}{l:12.1f}{v:12.1f}{v/l:20.3f}")
edges_lbl = ["[0,.003)", "[.003,.008)", "[.008,.016)", "[.016,.028)", "[.028,.045)", "[.045,inf)"]
for nm, l, v in zip(edges_lbl, ls, LIVE_SIGHIST):
    print(f"{'SIGHIST_u '+nm:<28}{l:12.3f}{v:12.3f}{v/max(l,1e-6):20.3f}")
cc_lbl = ["size1", "size2", "size4-7", "size8-15", "size16+"]
for nm, l, v in zip(cc_lbl, lm, LIVE_MASKCC):
    print(f"{'MASKCC '+nm:<28}{l:12.3f}{v:12.3f}{v/max(l,1e-6):20.3f}")
for i, (l, v) in enumerate(zip(la, LIVE_AC), start=1):
    print(f"{'AC lag'+str(i):<28}{l:12.3f}{v:12.3f}{v/l:20.3f}")

print(f"\nlocal frac. w/ 16+ blob: {(maskhists[:,4]>0).mean():.3f}  (live: 1.00)")
print(f"local frac. w/ ANY sub-16 fragment: {(maskhists[:,:4].sum(1)>0).mean():.3f}  (live: 0.47)")
