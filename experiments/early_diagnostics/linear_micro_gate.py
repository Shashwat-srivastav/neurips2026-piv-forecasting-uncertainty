#!/usr/bin/env python3
"""GATE 3: linear corrector micro-gate, STREAMING (v2).

v1 crashed: np.stack on 1903 windows tried to allocate 892 MiB for X and
another 892 MiB for Y at once -- the identical eager-materialization bug
already fixed once this session in the Windows() dataset class. Ridge
regression's closed form only needs A = sum(F^T F) (8x8) and B = sum(F^T R)
(8x2) -- tiny, accumulable in a streaming pass. Never holds more than one
small batch of windows in memory.
"""
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

d = np.load("data/derived/window_similarity.npy", allow_pickle=True).item()
rows, order = d["rows"], d["order"]
train_idx = [i for i in order[:3000] if rows[i][0] not in HOLDOUT][:2000]
print(f"training windows: {len(train_idx)}  "
      f"trajectories: {len(set(rows[i][0] for i in train_idx))}", flush=True)

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
    """9 features per (n,t,h,w) -> return (n*t*h*w, 8), residual (n*t*h*w, 2)."""
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


def accumulate(idx_list, chunk=16, model_batch=4):
    A = np.zeros((8, 8)); B = np.zeros((8, 2)); n_elem = 0
    r_before = 0.0
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
        r_before += np.linalg.norm(R, axis=1).sum()
        n_elem += len(R)
        if (c0 // chunk) % 5 == 0:
            print(f"  {c0+len(chunk_idx)}/{len(idx_list)} windows processed", flush=True)
    return A, B, n_elem, r_before / n_elem


print("streaming TRAIN accumulation (ridge sufficient statistics only) ...", flush=True)
A, B, n_tr, _ = accumulate(train_idx)
lam = 1e-3
W = np.linalg.solve(A + lam * np.eye(8), B)
print(f"\nfitted on {n_tr:,} elements. weights (rows=feature, cols=u,v):\n{W.round(5)}")

ho_idx = [i for i, r in enumerate(rows) if r[0] in HOLDOUT][::6][:200]
print(f"\nstreaming HOLDOUT eval ({len(ho_idx)} windows, never in train_idx) ...", flush=True)
r0_sum = r1_sum = n_ho = 0.0
for c0 in range(0, len(ho_idx), 16):
    chunk_idx = ho_idx[c0:c0 + 16]
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
    r0_sum += np.linalg.norm(R, axis=1).sum()
    r1_sum += np.linalg.norm(R - pred, axis=1).sum()
    n_ho += len(R)

r0, r1 = r0_sum / n_ho, r1_sum / n_ho
print(f"\nHOLDOUT residual norm, before: {r0:.5f}   after linear correction: {r1:.5f}")
print(f"reduction: {(1 - r1/r0)*100:+.3f}%")
print(f"\nGATE: {'SIGNAL FOUND -- proceed to full head' if r1 < r0 * 0.98 else 'NO SIGNAL -- do not build the full nonlinear head on this feature set'}")
