#!/usr/bin/env python3
"""GATE 2, AoA-15 ONLY. The previous run scanned AoA {10,15,20} (the
"regime-plausible" set from k-means clustering). Live AoA is DIRECTLY known to
be 15, not inferred -- so training-pool selection should not import AoA 10/20
windows on an assumption when the exact AoA is on record."""
import os, glob
import numpy as np, h5py

ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
EXCLUDE = {"7575_0"}
T_IN, T_OUT, SUB = 20, 20, 2
LIVE = np.array([0.21810, 0.15039, 0.02324, 0.01571, 0.00821, 0.01452])

c = np.load("data/derived/cluster_centroids.npy", allow_pickle=True).item()
mu, sd = c["mu"], c["sd"]

files = sorted(f for f in glob.glob(os.path.join(ROOT, "*_15.h5"))
               if os.path.splitext(os.path.basename(f))[0] not in EXCLUDE)
print(f"AoA-15 trajectories: {len(files)}")

rows = []
for p in files:
    nm = os.path.splitext(os.path.basename(p))[0]
    with h5py.File(p) as f:
        u = np.asarray(f["u"][:, ::SUB, ::SUB], np.float32)
        v = np.asarray(f["v"][:, ::SUB, ::SUB], np.float32)
    T = u.shape[0]
    for i in range(0, T - (T_IN + T_OUT) + 1, T_IN):
        wu, wv = u[i:i + T_IN], v[i:i + T_IN]
        fl = wu - wu.mean(0, keepdims=True)
        feat = np.array([
            wu.mean(), wu.std(), wv.std(),
            np.sqrt((fl ** 2).mean(0)).mean(),
            np.sqrt(((wv - wv.mean(0, keepdims=True)) ** 2).mean(0)).mean(),
            np.abs(np.diff(wu, axis=0)).mean(),
        ])
        rows.append((nm, i, feat))

F = np.stack([r[2] for r in rows])
Z = (F - mu) / sd
zl = (LIVE - mu) / sd
dist = np.linalg.norm(Z - zl, axis=1)
order = np.argsort(dist)

print(f"total AoA-15 windows: {len(rows)}")
for topk in (300, 500, 1000):
    sel = order[:topk]
    trajs = set(rows[i][0] for i in sel)
    counts = {}
    for i in sel:
        counts[rows[i][0]] = counts.get(rows[i][0], 0) + 1
    print(f"\ntop-{topk}: {len(trajs)} distinct trajectories, "
          f"max share {max(counts.values())/topk*100:.1f}%")
    for nm, cnt in sorted(counts.items(), key=lambda x: -x[1])[:8]:
        print(f"    {nm:12} {cnt:4d}  dist "
              f"{dist[[i for i in sel if rows[i][0]==nm]].min():.2f}-"
              f"{dist[[i for i in sel if rows[i][0]==nm]].max():.2f}")

np.save("data/derived/window_similarity_aoa15.npy",
        {"rows": [(r[0], r[1]) for r in rows], "dist": dist, "order": order},
        allow_pickle=True)
print("\nsaved window_similarity_aoa15.npy")
