#!/usr/bin/env python3
"""Score a fine-tuned CNO checkpoint on holdout_v12_3.json, at the official
stride=20 protocol, against the FAIR baseline.

CORRECTED METHODOLOGY (27 Aug): earlier scripts this session
(score_finetuned_ckpt.py, score_boosted_ckpt.py, score_unet_ckpt.py) compared
clean-start candidates against `holdout_v12_3_stock_scores.json`'s frozen
sim_real_fno score. That baseline is contaminated -- sim_real_fno was trained
by the organizers on all of train_real, so its "holdout" score is a
memorization artifact (confirmed empirically 27 Aug: frozen sim_fno, which
genuinely never touched real data, scores rel 79.7/tke 66.9/mvpe 75.2 on this
same holdout -- a real domain-shift penalty -- while sim_real_fno scores
96.4/81.0/97.1, HIGHER than its own live score, which is the signature of
memorization, not generalization).

This script instead compares against `holdout_v12_3_simcno_frozen_scores.json`
-- the frozen, never-fine-tuned sim_cno checkpoint on the SAME holdout,
computed 27 Aug (rel 78.145/tke 67.145/mvpe 74.743 on final-10). Both sides of
this comparison are equally "clean" (neither has seen the holdout), so it is
a fair ablation: did fine-tuning help, relative to not fine-tuning, starting
from the same checkpoint. It does NOT tell you whether the result beats FNO
live -- nothing local can. Only a live submission answers that.

Usage:
    python score_cno_ckpt.py <path-to-checkpoint.pth>
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

if len(sys.argv) != 2:
    sys.exit(__doc__)
CKPT = sys.argv[1]
if not os.path.exists(CKPT):
    sys.exit(f"not found: {CKPT}")

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
print(f"checkpoint: {CKPT}  ({os.path.getsize(CKPT)/1e6:.1f} MB)")
print(f"scoring on {len(names)} holdout trajectories "
      f"({len(split['select'])} select + {len(split['final'])} final)\n",
      flush=True)

dev = "cpu"
model, meta = load_baseline(CKPT, device=dev)
model = model.to(dev).eval()
print(f"loaded  meta {meta}", flush=True)
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
fair_path = os.path.join(REPO, "results", "holdout_v12_3_simcno_frozen_scores.json")
fair = json.load(open(fair_path)) if os.path.exists(fair_path) else None
contaminated_path = os.path.join(REPO, "results", "holdout_v12_3_stock_scores.json")
contaminated = json.load(open(contaminated_path)) if os.path.exists(contaminated_path) else None

print(f"{'LIVE (board)':16} {'':16}   "
      f"rel {live['rel']:8.3f}  tke {live['tke']:8.3f}  mvpe {live['mvpe']:8.3f}\n")

full = score(names, "THIS CKPT (14)")
fin = score(split["final"], "  -- final (10)")

TRUE_W = dict(rel=0.649, tke=0.131, mvpe=0.160)   # CAMPAIGN_REPORT.md sec.14.1

if fair:
    s = fair["final_10"]
    print(f"\n{'FAIR baseline':16} {'':16}   "
          f"rel {s['rel']:8.3f}  tke {s['tke']:8.3f}  mvpe {s['mvpe']:8.3f}"
          f"   (frozen sim_cno, never touched real data, 27 Aug)")
    d = {k: fin[k] - s[k] for k in ("rel", "tke", "mvpe")}
    true_comp = sum(TRUE_W[k] * d[k] for k in TRUE_W)
    print("\ndelta vs the FAIR (equally-clean) baseline on the SAME untouched final(10):")
    print(f"  rel {d['rel']:+7.3f}  tke {d['tke']:+7.3f}  mvpe {d['mvpe']:+7.3f}"
          f"  naive-sum {sum(d.values()):+7.3f}  TRUE-composite {true_comp:+7.3f}")
    print("\nThis answers ONE question: did fine-tuning help, relative to the frozen")
    print("checkpoint it started from. A positive number here means real training")
    print("signal, not memorization -- both sides of this comparison are equally clean.")
else:
    print("\n(no holdout_v12_3_simcno_frozen_scores.json found -- that's the fair")
    print(" baseline; without it there is nothing honest to compare against locally)")

if contaminated:
    s = contaminated["final_10"]
    print(f"\n{'reference only':16} {'':16}   "
          f"rel {s['rel']:8.3f}  tke {s['tke']:8.3f}  mvpe {s['mvpe']:8.3f}"
          f"   (frozen sim_real_fno -- CONTAMINATED, likely memorized this")
    print(f"{'':34}holdout; shown for context only, NOT a valid comparison target)")

print("\n(Everything above is LOCAL. Per the 27 Aug finding, there is currently no")
print(" known-clean local proxy for live performance -- only a live submission")
print(" tells you whether this beats the current best (78.140964). Local numbers")
print(" here answer only 'did training help vs not training', nothing more.)")
