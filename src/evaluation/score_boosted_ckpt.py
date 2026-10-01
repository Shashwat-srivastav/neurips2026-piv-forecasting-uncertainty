#!/usr/bin/env python3
"""Score a boosted FNO ensemble (frozen base + boost_operator_kaggle.py's
boost_stages.pth) on holdout_v12_3.json's 14-trajectory holdout, at stride=20
-- the SAME official protocol as score_finetuned_ckpt.py and
holdout_v12_3_stock_scores.json.

WHY THIS EXISTS: boost_operator_kaggle.py's own printed FINAL numbers are
computed at stride=2 (dense, overlapping windows), which is not comparable to
the stride=20 numbers used everywhere else in this campaign. Confirmed 27 Aug:
the plain fine-tune's stride=2 internal FINAL delta (+2.374 sum) and its
stride=20 official rescore (+1.223 sum, TRUE-composite -0.10) disagreed on the
SIGN of whether to submit. Never trust a corrector's own internal number for a
submit decision -- always rescore here first.

Usage:
    python score_boosted_ckpt.py <path-to-boost_stages.pth> [<base-ckpt.pth>]

If <base-ckpt.pth> is omitted, defaults to .dist_eta0/sim_real_fno_fp16.pth
(the currently-shipping frozen base, matching what boost_operator_kaggle.py's
own `build(cfg["kit"], cfg["ckpt"], dev)` loads by default).
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

if len(sys.argv) < 2:
    sys.exit(__doc__)
BOOST_PATH = sys.argv[1]
BASE_PATH = (sys.argv[2] if len(sys.argv) > 2
             else os.path.join(HERE, ".dist_eta0", "sim_real_fno_fp16.pth"))
if not os.path.exists(BOOST_PATH):
    sys.exit(f"not found: {BOOST_PATH}")
if not os.path.exists(BASE_PATH):
    sys.exit(f"not found: {BASE_PATH}")

sys.path.insert(0, KIT)
sys.path.insert(0, HERE)
import importlib.util as _iu
spec = _iu.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = _iu.module_from_spec(spec)
spec.loader.exec_module(sc)
from load_baseline import load_baseline                      # noqa: E402
import boost_operator_kaggle as B                              # noqa: E402

MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], np.float32)
STD_IN = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], np.float32)
STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
BATCH = 5

split = json.load(open(os.path.join(REPO, "data/splits/holdout_v12_3.json")))
names = split["select"] + split["final"]
print(f"base:  {BASE_PATH}  ({os.path.getsize(BASE_PATH)/1e6:.1f} MB)")
print(f"boost: {BOOST_PATH}  ({os.path.getsize(BOOST_PATH)/1e6:.1f} MB)")
print(f"scoring on {len(names)} holdout trajectories "
      f"({len(split['select'])} select + {len(split['final'])} final)\n",
      flush=True)

dev = "cpu"
base, meta = load_baseline(BASE_PATH, device=dev)
base = base.to(dev).eval()
print(f"loaded base  meta {meta}", flush=True)

ckpt = torch.load(BOOST_PATH, map_location=dev)
stages = []
for s in ckpt["stages"]:
    corr = B.TinyOperator(width=ckpt["width"], modes=(ckpt["modes"], ckpt["modes"]))
    corr.load_state_dict(s["state"])
    corr.to(dev).eval()
    stages.append((corr, s["eta"]))
print(f"loaded {len(stages)} accepted stage(s), etas {[e for _, e in stages]}\n",
      flush=True)

mi, si, mt, st = (torch.from_numpy(z) for z in (MEAN_IN, STD_IN, MEAN_TGT, STD_TGT))


def load_windows(names, stride=20):
    X, Y = [], []
    for nm in names:
        with h5py.File(os.path.join(ROOT, nm + ".h5")) as f:
            u = np.asarray(f["u"][:, ::2, ::2], np.float32)
            v = np.asarray(f["v"][:, ::2, ::2], np.float32)
        fr = np.stack([u, v, np.zeros_like(u)], -1)
        for i in range(0, len(fr) - 40 + 1, stride):
            X.append(fr[i:i + 20]); Y.append(fr[i + 20:i + 40])
    return np.stack(X), np.stack(Y)


@torch.no_grad()
def score(names, tag, stride=20):
    X, Y = load_windows(names, stride)
    P = []
    for i in range(0, len(X), BATCH):
        chunk = X[i:i + BATCH]
        got = chunk.shape[0]
        if got < BATCH:
            chunk = np.concatenate(
                [chunk, np.repeat(chunk[-1:], BATCH - got, axis=0)], axis=0)
        xb = torch.from_numpy(np.ascontiguousarray(chunk))
        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
        yb = base((xb - mi) / si) * st + mt
        for corr, eta in stages:
            d = corr(xb)
            yb = yb + eta * torch.cat([d[..., :2], torch.zeros_like(d[..., 2:])], -1)
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
stock_path = os.path.join(REPO, "results", "holdout_v12_3_stock_scores.json")
stock = json.load(open(stock_path)) if os.path.exists(stock_path) else None

print(f"{'LIVE (board)':16} {'':16}   "
      f"rel {live['rel']:8.3f}  tke {live['tke']:8.3f}  mvpe {live['mvpe']:8.3f}\n")

full = score(names, "THIS BOOST (14)")
fin = score(split["final"], "  -- final (10)")

TRUE_W = dict(rel=0.649, tke=0.131, mvpe=0.160)   # CAMPAIGN_REPORT.md sec.14.1

if stock:
    s = stock["final_10"]
    print(f"\n{'stock final(10)':16} {'':16}   "
          f"rel {s['rel']:8.3f}  tke {s['tke']:8.3f}  mvpe {s['mvpe']:8.3f}"
          f"   (sim_real_fno_fp16, from 22 Aug)")
    d = {k: fin[k] - s[k] for k in ("rel", "tke", "mvpe")}
    true_comp = sum(TRUE_W[k] * d[k] for k in TRUE_W)
    print("\ndelta vs the frozen stock checkpoint on the SAME untouched final(10):")
    print(f"  rel {d['rel']:+7.3f}  tke {d['tke']:+7.3f}  mvpe {d['mvpe']:+7.3f}"
          f"  naive-sum {sum(d.values()):+7.3f}  TRUE-composite {true_comp:+7.3f}")
    print("\n(TRUE-composite uses the 22 Aug weights (rel .649/tke .131/mvpe .160),")
    print(" not an equal-weight sum. This is still a LOCAL comparison; trained-model")
    print(" local deltas transfer live at a small, unpredictable fraction of their")
    print(" size and have flipped sign before. Only a positive TRUE-composite")
    print(" justifies spending a live submission slot on this.)")
else:
    print("\n(no holdout_v12_3_stock_scores.json found -- run score_holdout_v12_3.py "
          "first to get the stock comparison baseline)")
