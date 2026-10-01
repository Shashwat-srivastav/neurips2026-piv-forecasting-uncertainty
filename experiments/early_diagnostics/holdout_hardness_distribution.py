#!/usr/bin/env python3
"""Where does live actually sit against the per-TRAJECTORY distribution of the
14-trajectory holdout, not just the pooled average?

score_holdout_v12_3.py answered "does the frozen checkpoint generalize, on
average, to genuinely unseen data" (yes: 96.3/80.7/97.2 pooled). It never
asked the sharper question: is live simply the hardest single draw from the
SAME spread this model already shows across individual unseen trajectories,
or is it an outlier even against that spread's low end? That's a directly
testable, per-trajectory question -- no new data, no new training, just a
finer breakdown of what's already been measured.
"""
import gc
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

model, meta = load_baseline(CKPT, device="cpu")
model = model.eval()
mi, si, mt, st = (torch.from_numpy(z) for z in (MEAN_IN, STD_IN, MEAN_TGT, STD_TGT))
split = json.load(open(os.path.join(REPO, "data/splits/holdout_v12_3.json")))
names = split["select"] + split["final"]


def load_windows(name, stride=20):
    with h5py.File(os.path.join(ROOT, name + ".h5")) as f:
        u = np.asarray(f["u"][:, ::2, ::2], np.float32)
        v = np.asarray(f["v"][:, ::2, ::2], np.float32)
        re, aoa = float(f["re"][()]), float(f["aoa"][()])
    fr = np.stack([u, v, np.zeros_like(u)], -1)
    X, Y = [], []
    for i in range(0, len(fr) - 40 + 1, stride):
        X.append(fr[i:i + 20]); Y.append(fr[i + 20:i + 40])
    return np.stack(X), np.stack(Y), re, aoa


@torch.no_grad()
def score_one(name):
    X, Y, re, aoa = load_windows(name)
    P = []
    for i in range(0, len(X), BATCH):
        chunk = X[i:i + BATCH]
        got = chunk.shape[0]
        if got < BATCH:
            chunk = np.concatenate(
                [chunk, np.repeat(chunk[-1:], BATCH - got, axis=0)], axis=0)
        xb = torch.from_numpy(np.ascontiguousarray(chunk))
        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
        yb = model((xb - mi) / si) * st + mt
        yb[..., 2] = 0.0
        P.append(yb.float().numpy()[:got])
    P = np.concatenate(P)
    rel = sc.score_error(float(sc.rel_l2_per_sample(P, Y, 2).mean()))
    tke = sc.score_error(float(sc.tke_rel_l2_per_sample(P, Y, 2).mean()))
    mvpe = sc.score_error(float(sc.mvpe_rel_l2_per_sample(P, Y).mean()))
    out = dict(name=name, re=re, aoa=aoa, n=len(X), rel=rel, tke=tke, mvpe=mvpe,
               sum=rel + tke + mvpe)
    del X, Y, P
    return out


print(f"scoring each of {len(names)} holdout trajectories individually "
      f"(same frozen sim_real_fno_fp16 as live)...\n", flush=True)
# Explicit gc.collect() + a printed per-trajectory line between iterations --
# this machine runs at ~0.2 GB free routinely, and 14 sequential loads/
# forward-passes with no cleanup accumulated enough fragmentation to
# segfault twice before this was added. Printing as it goes also means a
# crash loses only the remaining trajectories, not everything computed so far.
rows = []
for nm in names:
    r = score_one(nm)
    rows.append(r)
    print(f"  {r['name']:>10}  re {r['re']:.0f}  aoa {r['aoa']:.0f}  "
          f"n {r['n']}  rel {r['rel']:.3f}  tke {r['tke']:.3f}  "
          f"mvpe {r['mvpe']:.3f}  sum {r['sum']:.3f}", flush=True)
    gc.collect()
rows.sort(key=lambda r: r["sum"])

live = dict(name="LIVE", rel=94.168148, tke=74.025874, mvpe=92.836233)
live["sum"] = live["rel"] + live["tke"] + live["mvpe"]

print(f"{'trajectory':>12}{'re':>7}{'aoa':>5}{'n':>5}{'rel':>9}{'tke':>9}{'mvpe':>9}{'sum':>10}")
print("-" * 66)
inserted = False
for r in rows:
    if not inserted and live["sum"] < r["sum"]:
        print(f"{'>> LIVE':>12}{'':>7}{'':>5}{'':>5}{live['rel']:9.3f}"
              f"{live['tke']:9.3f}{live['mvpe']:9.3f}{live['sum']:10.3f}   <-- rank")
        inserted = True
    print(f"{r['name']:>12}{r['re']:7.0f}{r['aoa']:5.0f}{r['n']:5d}"
          f"{r['rel']:9.3f}{r['tke']:9.3f}{r['mvpe']:9.3f}{r['sum']:10.3f}")
if not inserted:
    print(f"{'>> LIVE':>12}{'':>7}{'':>5}{'':>5}{live['rel']:9.3f}"
          f"{live['tke']:9.3f}{live['mvpe']:9.3f}{live['sum']:10.3f}   <-- rank (hardest of all)")

sums = np.array([r["sum"] for r in rows])
print(f"\n14-trajectory sum distribution: min {sums.min():.3f}  "
      f"p10 {np.percentile(sums,10):.3f}  median {np.median(sums):.3f}  "
      f"max {sums.max():.3f}  std {sums.std():.3f}")
z = (live["sum"] - sums.mean()) / sums.std()
print(f"live sum {live['sum']:.3f}  is {abs(z):.2f} std devs "
      f"{'BELOW' if z < 0 else 'above'} the holdout mean "
      f"({sums.mean():.3f} +/- {sums.std():.3f})")
print(f"live rank: {sum(1 for s in sums if s < live['sum'])} of {len(sums)} "
      f"holdout trajectories score WORSE than live's aggregate")
