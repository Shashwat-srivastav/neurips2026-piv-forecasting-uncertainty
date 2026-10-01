#!/usr/bin/env python3
"""Score ONE trajectory with the frozen sim_real_fno_fp16 checkpoint and print
one JSON line. Meant to be invoked as a fresh subprocess per trajectory (see
holdout_hardness_distribution.py's rewrite) -- this machine runs at ~0.2-0.3 GB
free routinely, and a single long-lived process doing 14 sequential model
loads + forward passes segfaulted 4 times in a row (varying WHERE it crashed
each time) even with explicit gc.collect() between iterations, consistent
with C-level heap fragmentation across the 14 alloc/free cycles rather than
raw memory exhaustion. A fresh process per trajectory returns its entire heap
to the OS on exit, eliminating that class of failure at the cost of ~2.5s of
repeated model-loading per trajectory.

Usage: python score_one_trajectory.py <traj_name>
Prints exactly one line: {"name": ..., "re": ..., "aoa": ..., "n": ...,
"rel": ..., "tke": ..., "mvpe": ..., "sum": ...}
"""
import json
import os
import sys

import h5py
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'src', d) for d in ('training', 'evaluation', 'splits')]
KIT = os.path.join(REPO, "realpde_t1_starting_kit_v9", "realpde_t1_starting_kit_v9")
ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
CKPT = os.path.join(HERE, ".dist_eta0", "sim_real_fno_fp16.pth")

if len(sys.argv) != 2:
    sys.exit("usage: python score_one_trajectory.py <traj_name>")
NAME = sys.argv[1]

sys.path.insert(0, KIT)
sp = __import__("importlib.util", fromlist=["spec_from_file_location"])
spec = sp.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = sp.module_from_spec(spec)
spec.loader.exec_module(sc)
from load_baseline import load_baseline                      # noqa: E402

MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], np.float32)
STD_IN = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], np.float32)
STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
BATCH = 5

print("[trace] loading model...", file=sys.stderr, flush=True)
model, _ = load_baseline(CKPT, device="cpu")
model = model.eval()
print("[trace] model loaded", file=sys.stderr, flush=True)
mi, si, mt, st = (torch.from_numpy(z) for z in (MEAN_IN, STD_IN, MEAN_TGT, STD_TGT))

print(f"[trace] opening {NAME}.h5", file=sys.stderr, flush=True)
with h5py.File(os.path.join(ROOT, NAME + ".h5")) as f:
    u = np.asarray(f["u"][:, ::2, ::2], np.float32)
    v = np.asarray(f["v"][:, ::2, ::2], np.float32)
    re, aoa = float(f["re"][()]), float(f["aoa"][()])
print(f"[trace] h5 loaded, u.shape={u.shape}", file=sys.stderr, flush=True)
fr = np.stack([u, v, np.zeros_like(u)], -1)
X, Y = [], []
for i in range(0, len(fr) - 40 + 1, 20):
    X.append(fr[i:i + 20]); Y.append(fr[i + 20:i + 40])
X, Y = np.stack(X), np.stack(Y)
print(f"[trace] windows built: {X.shape}", file=sys.stderr, flush=True)

P = []
with torch.no_grad():
    for i in range(0, len(X), BATCH):
        chunk = X[i:i + BATCH]
        got = chunk.shape[0]
        if got < BATCH:
            chunk = np.concatenate(
                [chunk, np.repeat(chunk[-1:], BATCH - got, axis=0)], axis=0)
        xb = torch.from_numpy(np.ascontiguousarray(chunk))
        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"[trace] batch {i}: forward pass...", file=sys.stderr, flush=True)
        yb = model((xb - mi) / si) * st + mt
        yb[..., 2] = 0.0
        P.append(yb.float().numpy()[:got])
        print(f"[trace] batch {i}: done", file=sys.stderr, flush=True)
P = np.concatenate(P)
print("[trace] all batches done, computing metrics", file=sys.stderr, flush=True)

rel = sc.score_error(float(sc.rel_l2_per_sample(P, Y, 2).mean()))
tke = sc.score_error(float(sc.tke_rel_l2_per_sample(P, Y, 2).mean()))
mvpe = sc.score_error(float(sc.mvpe_rel_l2_per_sample(P, Y).mean()))
print(json.dumps(dict(name=NAME, re=re, aoa=aoa, n=len(X), rel=rel, tke=tke,
                       mvpe=mvpe, sum=rel + tke + mvpe)))
