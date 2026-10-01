#!/usr/bin/env python3
"""GATE 2: multivariate live-similarity window selection, replacing the
blunt two-threshold filter (u_std>0.14 AND framediff>0.012), which kept 659
windows, 29-of-31 from a SINGLE trajectory (26700_15) -- not a real
multi-trajectory selection, just a proxy for "the one extreme trajectory".

An AND-threshold on 2 correlated marginals throws away everything that is
close-but-not-quite on EITHER axis and ignores their correlation entirely.
This ranks every window by Mahalanobis-style distance to the live fingerprint
in the SAME 6-D standardized space already used for the regime clustering
(u_mean, u_std, v_std, sl_u, sl_v, dfrm), and takes a smooth top-K/percentile
cut instead of a hard AND.
"""
import os, glob
import numpy as np, h5py

ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
EXCLUDE = {"7575_0"}
T_IN, T_OUT, SUB = 20, 20, 2
LIVE = np.array([0.21810, 0.15039, 0.02324, 0.01571, 0.00821, 0.01452])  # u_mean,u_std,v_std,sl_u,sl_v,dfrm

c = np.load("data/derived/cluster_centroids.npy", allow_pickle=True).item()
mu, sd = c["mu"], c["sd"]
print(f"reusing the clustering's own standardization: mu={mu.round(4).tolist()} "
      f"sd={sd.round(4).tolist()}")

files = sorted(f for f in glob.glob(os.path.join(ROOT, "*_15.h5"))
               if os.path.splitext(os.path.basename(f))[0] not in EXCLUDE)
files += sorted(f for f in glob.glob(os.path.join(ROOT, "*_10.h5")) + glob.glob(os.path.join(ROOT, "*_20.h5"))
                if os.path.splitext(os.path.basename(f))[0] not in EXCLUDE)
print(f"scanning {len(files)} AoA-{{10,15,20}} trajectories (the regime-plausible AoA set)")

rows = []   # (traj, start, feature vector)
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
            np.sqrt((fl ** 2).mean(0)).mean(),                        # sl_u
            np.sqrt(((wv - wv.mean(0, keepdims=True)) ** 2).mean(0)).mean(),  # sl_v
            np.abs(np.diff(wu, axis=0)).mean(),                       # dfrm
        ])
        rows.append((nm, i, feat))

F = np.stack([r[2] for r in rows])
Z = (F - mu) / sd
zl = (LIVE - mu) / sd
dist = np.linalg.norm(Z - zl, axis=1)
order = np.argsort(dist)

print(f"\ntotal windows scanned: {len(rows)}")

# Old two-threshold selection, for direct comparison
U_STD_MIN, FRAMEDIFF_MIN = 0.14, 0.012
old_kept = [(r[0], r[1]) for r in rows if r[2][1] > U_STD_MIN and r[2][5] > FRAMEDIFF_MIN]
print(f"OLD two-threshold selection: {len(old_kept)} windows, "
      f"{len(set(t for t,_ in old_kept))} distinct trajectories")

for topk in (500, 1000, 2000, 5000):
    sel = order[:topk]
    trajs = set(rows[i][0] for i in sel)
    counts = {}
    for i in sel:
        counts[rows[i][0]] = counts.get(rows[i][0], 0) + 1
    top_traj_frac = max(counts.values()) / topk
    print(f"\nNEW top-{topk} by live-distance: {len(trajs)} distinct trajectories, "
          f"max single-trajectory share {top_traj_frac*100:.1f}%")
    for nm, cnt in sorted(counts.items(), key=lambda x: -x[1])[:6]:
        print(f"    {nm:12} {cnt:5d} windows  (dist range "
              f"{dist[[i for i in sel if rows[i][0]==nm]].min():.2f}-"
              f"{dist[[i for i in sel if rows[i][0]==nm]].max():.2f})")

np.save("data/derived/window_similarity.npy",
        {"rows": [(r[0], r[1]) for r in rows], "dist": dist, "order": order},
        allow_pickle=True)
print("\nsaved window_similarity.npy (trajectory, start, distance, rank order)")
