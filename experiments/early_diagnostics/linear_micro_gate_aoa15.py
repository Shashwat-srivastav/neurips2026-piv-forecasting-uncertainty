#!/usr/bin/env python3
"""GATE 3 RERUN: AoA-15-only selection (window_similarity_aoa15.npy), and a
LIVE-SIMILARITY-RANKED holdout instead of an arbitrary stride sample -- the
most live-similar windows within the 5 held-out trajectories, not just some
of them at a fixed interval. Streaming (fixed the eager np.stack OOM once
already this session); the fitted feature/model design is otherwise identical
to the first Gate 3 run, so this isolates exactly the two changes requested."""
import os
import numpy as np, h5py, torch

KIT = os.environ.get("RPDE_KIT_DIR", "")
ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
import sys; sys.path.insert(0, KIT)
from load_baseline import load_baseline

MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32)
SI = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32)
ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
HOLDOUT = {"21600_15", "22875_15", "20325_15", "24150_15", "19050_15"}
T_IN = T_OUT = 20

d = np.load("data/derived/window_similarity_aoa15.npy", allow_pickle=True).item()
rows, order, dist = d["rows"], d["order"], d["dist"]

train_idx = [i for i in order if rows[i][0] not in HOLDOUT][:500]
print(f"TRAIN (AoA-15, live-similarity top-500, holdout excluded): "
      f"{len(train_idx)} windows, {len(set(rows[i][0] for i in train_idx))} trajectories",
      flush=True)

# live-similarity-RANKED holdout: the 60 CLOSEST-to-live windows within the
# held-out trajectories, not a stride subsample -- the harder, more relevant
# test (does correction help where it matters most).
ho_all = [i for i in order if rows[i][0] in HOLDOUT]
ho_idx = ho_all[:60]
print(f"HOLDOUT (live-similarity top-60 within held-out trajectories): "
      f"{len(ho_idx)} windows, dist range "
      f"{dist[ho_idx].min():.2f}-{dist[ho_idx].max():.2f}", flush=True)

model, _ = load_baseline(os.path.join(KIT, "sim_real_fno_fp16.pth"), device="cpu")
model = model.float().eval()

_cache = {}
def load_traj(nm):
    if nm not in _cache:
        with h5py.File(os.path.join(ROOT, nm + ".h5")) as f:
            u = np.asarray(f["u"][:, ::2, ::2], np.float32)
            v = np.asarray(f["v"][:, ::2, ::2], np.float32)
        _cache[nm] = np.stack([u, v, np.zeros_like(u)], -1)
    return _cache[nm]


def feat_batch(X, P):
    xc = X[..., :2]
    fl = xc - xc.mean(1, keepdims=True)
    sl = np.sqrt((fl ** 2).mean(1))
    fd = np.abs(np.diff(xc, axis=1)).mean(1)
    mask = ((xc[:, -1, ..., 0] == 0) & (xc[:, -1, ..., 1] == 0)).astype(np.float32)
    n, t, h, w, _ = P.shape
    F = np.empty((n, t, h, w, 8), np.float32)
    F[..., 0] = P[..., 0]; F[..., 1] = P[..., 1]
    F[..., 2] = sl[:, None, ..., 0]; F[..., 3] = sl[:, None, ..., 1]
    F[..., 4] = fd[:, None, ..., 0]; F[..., 5] = fd[:, None, ..., 1]
    F[..., 6] = mask[:, None]
    F[..., 7] = 1.0
    return F.reshape(-1, 8)


def run_pool(idx_list, chunk=16, model_batch=4, tag="pool"):
    A = np.zeros((8, 8)); B = np.zeros((8, 2))
    r_sum, n_elem = 0.0, 0
    for c0 in range(0, len(idx_list), chunk):
        chunk_idx = idx_list[c0:c0 + chunk]
        Xs, Ys = [], []
        for i in chunk_idx:
            nm, s = rows[i]
            fr = load_traj(nm)
            Xs.append(fr[s:s + T_IN]); Ys.append(fr[s + T_IN:s + T_IN + T_OUT])
        X = np.stack(Xs); Y = np.stack(Ys)
        P = []
        with torch.no_grad():
            for i in range(0, len(X), model_batch):
                xb = torch.from_numpy(X[i:i + model_batch])
                yb = model((xb - MI) / SI) * ST + MT
                q = yb.numpy().copy(); q[..., 2] = 0.0
                P.append(q)
        P = np.concatenate(P)
        F = feat_batch(X, P)
        R = (Y[..., :2] - P[..., :2]).reshape(-1, 2)
        A += F.T @ F; B += F.T @ R
        r_sum += np.linalg.norm(R, axis=1).sum(); n_elem += len(R)
        if (c0 // chunk) % 5 == 0:
            print(f"  [{tag}] {c0+len(chunk_idx)}/{len(idx_list)} windows", flush=True)
    return A, B, r_sum / max(n_elem, 1), n_elem


print("\nstreaming TRAIN accumulation ...", flush=True)
A, B, _, n_tr = run_pool(train_idx, tag="train")
lam = 1e-3
W = np.linalg.solve(A + lam * np.eye(8), B)
print(f"\nfitted on {n_tr:,} elements. weights:\n{W.round(5)}")

print("\nstreaming HOLDOUT (live-similarity ranked) eval ...", flush=True)
A_h, B_h, r0, n_ho = run_pool(ho_idx, chunk=10, tag="holdout")

# recompute residual-after using the SAME streaming pass structure
r1_sum, n_check = 0.0, 0
for c0 in range(0, len(ho_idx), 10):
    chunk_idx = ho_idx[c0:c0 + 10]
    Xs, Ys = [], []
    for i in chunk_idx:
        nm, s = rows[i]
        fr = load_traj(nm)
        Xs.append(fr[s:s + T_IN]); Ys.append(fr[s + T_IN:s + T_IN + T_OUT])
    X = np.stack(Xs); Y = np.stack(Ys)
    P = []
    with torch.no_grad():
        for i in range(0, len(X), 4):
            xb = torch.from_numpy(X[i:i + 4])
            yb = model((xb - MI) / SI) * ST + MT
            q = yb.numpy().copy(); q[..., 2] = 0.0
            P.append(q)
    P = np.concatenate(P)
    F = feat_batch(X, P)
    R = (Y[..., :2] - P[..., :2]).reshape(-1, 2)
    pred = F @ W
    r1_sum += np.linalg.norm(R - pred, axis=1).sum(); n_check += len(R)

r1 = r1_sum / n_check
print(f"\nHOLDOUT (live-similarity top-60) residual norm: before {r0:.5f}  after {r1:.5f}")
print(f"reduction: {(1 - r1/r0)*100:+.3f}%")
print(f"\nGATE: {'SIGNAL FOUND' if r1 < r0 * 0.98 else 'NO SIGNAL'}")
