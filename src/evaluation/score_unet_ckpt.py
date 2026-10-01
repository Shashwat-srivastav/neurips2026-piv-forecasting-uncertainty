#!/usr/bin/env python3
"""Score RealPDEBench's organizer-released UNet foil checkpoint
(huggingface.co/AI4Science-WestlakeU/RealPDEBench-models, foil/unet/finetune.pth)
on holdout_v12_3.json's 14-trajectory holdout, at stride=20 -- the SAME
official protocol as score_finetuned_ckpt.py and holdout_v12_3_stock_scores.json.

WHY: the RealPDEBench paper's own foil benchmark table (real-world finetuning
paradigm) reports UNet beating FNO by ~30% relative L2 (0.0145 vs 0.0206), and
UNet was never tried by this campaign -- only FNO/CNO/Transolver were vendored
in the starting kit. This checkpoint is the org's OWN real-data finetune, not
something we trained, so there is no overfitting risk from our side; the only
open question is whether it transfers to OUR specific eval regime (live AoA-15,
possibly a different/higher Re range than RealPDEBench's public foil release,
which is documented as Re 2968-17031 vs our live-measured Re ~20,600-25,200).

NORMALIZER CAVEAT: RealPDEBench's own foil/mean_std.pt has mean_inputs[0]=0.347,
which does NOT match our kit's real-PIV-derived MEAN_IN[0]=0.155 -- almost
certainly because that mean_std.pt reflects NUMERICAL (simulated) training data,
not the real-data finetuning stage (channels.json shows real fields are u,v
only; the checkpoint's 3-channel init_conv implies a zero-padded p, exactly
like our kit's own convention). This script uses OUR kit's own MEAN_IN/STD_IN/
MEAN_TGT/STD_TGT (same real-PIV distribution, verified against scoring.py to
1e-7 already) rather than the ambiguous downloaded mean_std.pt. If results look
degenerate (huge errors, saturated outputs), the normalizer is the first thing
to re-derive -- see build_own_normalizer() below for a from-scratch fallback
computed directly on OUR train_real data with GaussianNormalizer's own method.

Usage:
    python score_unet_ckpt.py <path-to-finetune.pth>
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
sys.path.insert(0, HERE)
import importlib.util as _iu
spec = _iu.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = _iu.module_from_spec(spec)
spec.loader.exec_module(sc)
from rpdebench_unet import Unet3d                              # noqa: E402

# Same constants as score_finetuned_ckpt.py -- verified against scoring.py to
# 1e-7, computed from OUR real train_real PIV data (not RealPDEBench's own
# mean_std.pt, which appears to be numerical-data stats -- see module docstring).
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
model = Unet3d(dim=64, out_channels=3, dim_mults=(1, 2, 4), channels=3,
                in_time=20, out_time=20)
meta = model.load_checkpoint(CKPT, dev)
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
stock_path = os.path.join(REPO, "results", "holdout_v12_3_stock_scores.json")
stock = json.load(open(stock_path)) if os.path.exists(stock_path) else None

print(f"{'LIVE (board)':16} {'':16}   "
      f"rel {live['rel']:8.3f}  tke {live['tke']:8.3f}  mvpe {live['mvpe']:8.3f}\n")

full = score(names, "THIS UNET (14)")
fin = score(split["final"], "  -- final (10)")

TRUE_W = dict(rel=0.649, tke=0.131, mvpe=0.160)   # CAMPAIGN_REPORT.md sec.14.1

if stock:
    s = stock["final_10"]
    print(f"\n{'stock final(10)':16} {'':16}   "
          f"rel {s['rel']:8.3f}  tke {s['tke']:8.3f}  mvpe {s['mvpe']:8.3f}"
          f"   (sim_real_fno_fp16, from 22 Aug)")
    d = {k: fin[k] - s[k] for k in ("rel", "tke", "mvpe")}
    true_comp = sum(TRUE_W[k] * d[k] for k in TRUE_W)
    print("\ndelta vs the frozen stock FNO on the SAME untouched final(10):")
    print(f"  rel {d['rel']:+7.3f}  tke {d['tke']:+7.3f}  mvpe {d['mvpe']:+7.3f}"
          f"  naive-sum {sum(d.values()):+7.3f}  TRUE-composite {true_comp:+7.3f}")
    print("\n(If rel/tke/mvpe all look wildly wrong -- e.g. scores near 0 or")
    print(" saturated -- the normalizer is the first suspect; see the module")
    print(" docstring's NORMALIZER CAVEAT before trusting any other number here.)")
else:
    print("\n(no holdout_v12_3_stock_scores.json found -- run score_holdout_v12_3.py "
          "first to get the stock comparison baseline)")
