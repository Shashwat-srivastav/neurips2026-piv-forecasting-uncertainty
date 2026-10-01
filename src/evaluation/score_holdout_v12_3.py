#!/usr/bin/env python3
"""Score the FROZEN, currently-shipped checkpoint (sim_real_fno_fp16, exactly
what's in submission_fno_eta0_v8.zip) on holdout_v12_3.json's 14-trajectory
holdout -- nothing trained, nothing fitted, this is the number 12.3 asked for:
does the incumbent's honest generalization gap on data it has genuinely never
seen match the live gap (94.168 / 74.026 / 92.836)?
"""
import json
import os

import h5py
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'src', d) for d in ('training', 'evaluation', 'splits')]
KIT = os.path.join(REPO, "realpde_t1_starting_kit_v9", "realpde_t1_starting_kit_v9")
ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
CKPT = os.path.join(HERE, ".dist_eta0", "sim_real_fno_fp16.pth")

import sys
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

split = json.load(open(os.path.join(REPO, "data/splits/holdout_v12_3.json")))
names = split["select"] + split["final"]
print(f"scoring on {len(names)} holdout trajectories "
      f"({len(split['select'])} select + {len(split['final'])} final)\n",
      flush=True)

dev = "cpu"
model, meta = load_baseline(CKPT, device=dev)
model = model.to(dev).eval()
print(f"loaded {os.path.basename(CKPT)}  {meta}", flush=True)
mi, si, mt, st = (torch.from_numpy(z) for z in (MEAN_IN, STD_IN, MEAN_TGT, STD_TGT))


def load_windows(names, stride=20):
    X, Y = [], []
    per_traj = {}
    for nm in names:
        with h5py.File(os.path.join(ROOT, nm + ".h5")) as f:
            u = np.asarray(f["u"][:, ::2, ::2], np.float32)
            v = np.asarray(f["v"][:, ::2, ::2], np.float32)
        fr = np.stack([u, v, np.zeros_like(u)], -1)
        n0 = len(X)
        for i in range(0, len(fr) - 40 + 1, stride):
            X.append(fr[i:i + 20]); Y.append(fr[i + 20:i + 40])
        per_traj[nm] = len(X) - n0
    return np.stack(X), np.stack(Y), per_traj


@torch.no_grad()
def score(names, tag, stride=20):
    X, Y, per_traj = load_windows(names, stride)
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
    print(f"{tag:16} windows {len(X):5d} ({len(names)} traj)   "
          f"rel {rel:8.3f}  tke {tke:8.3f}  mvpe {mvpe:8.3f}", flush=True)
    return dict(rel=rel, tke=tke, mvpe=mvpe, n=len(X))


live = dict(rel=94.168148, tke=74.025874, mvpe=92.836233)
print(f"{'LIVE (board)':16} {'':16}   "
      f"rel {live['rel']:8.3f}  tke {live['tke']:8.3f}  mvpe {live['mvpe']:8.3f}\n")

full = score(names, "HOLDOUT (14 traj)")
sel = score(split["select"], "  -- select (4)")
fin = score(split["final"], "  -- final (10, single-touch)")

print("\ngap vs live (positive = holdout beats live, i.e. a real generalisation gap):")
for tag, s in (("holdout(14)", full), ("final(10)", fin)):
    print(f"  {tag:12} rel {s['rel']-live['rel']:+7.3f}  "
          f"tke {s['tke']-live['tke']:+7.3f}  mvpe {s['mvpe']-live['mvpe']:+7.3f}")

json.dump({"live": live, "holdout_14": full, "select_4": sel, "final_10": fin},
          open(os.path.join(REPO, "results", "holdout_v12_3_stock_scores.json"), "w"), indent=1)
print("\nwrote holdout_v12_3_stock_scores.json")
