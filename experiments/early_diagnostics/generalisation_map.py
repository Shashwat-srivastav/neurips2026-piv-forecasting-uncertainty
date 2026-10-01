#!/usr/bin/env python3
"""Is the live 12.5-point loss ORDINARY generalisation, or is the eval flow anomalous?

This is the measurement that decides whether 80 is reachable by modelling at all,
and it needs no training and no submission slot. The FNO is frozen: 81 forward
passes over the whole training set.

The frozen FNO scores, on held-out AoA-15 trajectories, rel 96.06 / tke 80.29 /
mvpe 97.20 -- and on the live board 94.17 / 74.03 / 92.84. A 12.5-point loss.
Every attempt to close it by modelling has failed. The open question is which of
these is true:

  A) ORDINARY. Per-trajectory scores vary a lot anyway, and the live numbers sit
     inside that spread. Then the eval flow is simply a hard one, the gap is
     normal generalisation, and a better-generalising model can close it.

  B) ANOMALOUS. Every training trajectory scores far better than live, including
     ones the FNO never saw. Then the eval data differs from train_real in some
     way the model cannot learn from train_real, and NO model fitted on this data
     will close it. That would stop weeks of wasted work.

Distinguishing them: score the FNO on all 81, and ask where the live numbers fall
in that distribution. If live sits at the 5th percentile of trajectories the model
was TRAINED on, that is anomaly, not generalisation.

Reports the score against Re, AoA, and distance from the live flow fingerprint,
so any decay with distance is visible and can be extrapolated to the eval point.
"""
from __future__ import annotations

import importlib.util
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
CKPT = ("C:/Users/Shashwat/Downloads/realpde_t2_starting_kit_v3/"
        "realpde_t2_starting_kit_v3/NeurIPS 2026 RealPDECompetition/"
        "baseline_checkpoints/sim_real_ft/sim_real_fno_fp16.pth")
sys.path.insert(0, KIT)
sys.path.insert(0, HERE)

sp = importlib.util.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(sc)
from finetune_fno_kaggle import (  # noqa: E402
    MEAN_IN, STD_IN, MEAN_TGT, STD_TGT, build, forward_physical)

ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
TI = TO = 20
NWIN = 10                     # windows per trajectory: enough for the statistic
EXCLUDE = {"7575_0"}          # byte-identical duplicate of 6300_0
LIVE = dict(rel=94.168150, tke=74.025866, mvpe=92.836278)
FING = dict(u_mean=0.21810, u_std=0.15039, v_std=0.02323,
            sl_u=0.01626, sl_v=0.00731, dfrm=0.01448)


def load(traj):
    with h5py.File(os.path.join(ROOT, traj + ".h5"), "r") as f:
        u = np.asarray(f["u"][:, ::2, ::2], np.float32)
        v = np.asarray(f["v"][:, ::2, ::2], np.float32)
        re, aoa = float(f["re"][()]), float(f["aoa"][()])
    fr = np.stack([u, v, np.zeros_like(u)], -1)
    n = len(fr) - (TI + TO) + 1
    st = np.linspace(0, max(n - 1, 0), min(NWIN, max(n, 1))).astype(int)
    X = np.stack([fr[i:i + TI] for i in st])
    Y = np.stack([fr[i + TI:i + TI + TO] for i in st])
    return X, Y, re, aoa


def fingerprint(X):
    fl = X[..., :2] - X[..., :2].mean(1, keepdims=True)
    sl = np.sqrt((fl * fl).mean(1))
    xs = np.ascontiguousarray(X[:, ::4, ::4, ::4, :2])
    d1 = np.abs(xs[:, 1:] - xs[:, :-1])
    mk = (xs[:, :-1] != 0.0) & (xs[:, 1:] != 0.0)
    return dict(u_mean=float(X[..., 0].mean()), u_std=float(X[..., 0].std()),
                v_std=float(X[..., 1].std()), sl_u=float(sl[..., 0].mean()),
                sl_v=float(sl[..., 1].mean()),
                dfrm=float(d1[mk].mean()) if mk.any() else np.nan)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = build(KIT, CKPT, dev)
    model = model.to(dev).float().eval()
    mi, si, mt, st = (torch.from_numpy(z).to(dev)
                      for z in (MEAN_IN, STD_IN, MEAN_TGT, STD_TGT))

    names = sorted(f[:-3] for f in os.listdir(ROOT)
                   if f.endswith(".h5") and f[:-3] not in EXCLUDE)
    print(f"scoring the FROZEN FNO on {len(names)} trajectories, "
          f"{NWIN} windows each, on {dev}\n", flush=True)

    rows = []
    for k, t in enumerate(names):
        X, Y, re, aoa = load(t)
        with torch.no_grad():
            xb = torch.from_numpy(np.ascontiguousarray(X)).to(dev)
            p = forward_physical(model, xb, mi, si, mt, st).cpu().numpy()
        p = p.astype(np.float32)
        p[..., 2] = 0.0
        r = dict(rel=sc.score_error(float(sc.rel_l2_per_sample(p, Y, 2).mean())),
                 tke=sc.score_error(float(sc.tke_rel_l2_per_sample(p, Y, 2).mean())),
                 mvpe=sc.score_error(float(sc.mvpe_rel_l2_per_sample(p, Y).mean())))
        fp = fingerprint(X)
        d = float(np.mean([abs(np.log(max(fp[q], 1e-12) / FING[q])) for q in FING]))
        rows.append(dict(traj=t, re=re, aoa=aoa, dist=d, **r,
                         sum=r["rel"] + r["tke"] + r["mvpe"]))
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(names)}", flush=True)

    json.dump(rows, open(os.path.join(REPO, "results", "generalisation_map.json"), "w"), indent=1)
    A = {k: np.array([r[k] for r in rows]) for k in
         ("re", "aoa", "dist", "rel", "tke", "mvpe", "sum")}

    print("\nDISTRIBUTION over the 81 training trajectories, vs the LIVE score")
    print(f"  {'metric':6}{'min':>9}{'p5':>9}{'p25':>9}{'median':>9}{'max':>9}"
          f"{'LIVE':>10}{'percentile':>12}")
    for m in ("rel", "tke", "mvpe"):
        v = A[m]
        pc = 100.0 * float((v < LIVE[m]).mean())
        print(f"  {m:6}{v.min():9.2f}{np.percentile(v,5):9.2f}"
              f"{np.percentile(v,25):9.2f}{np.median(v):9.2f}{v.max():9.2f}"
              f"{LIVE[m]:10.2f}{pc:11.1f}%")

    print("\n  ^ 'percentile' = share of TRAINING trajectories scoring BELOW live.")
    print("    near 0% on all three  -> ANOMALY: live is worse than anything the")
    print("                             model was fitted on")
    print("    20-50%               -> ORDINARY: the eval flow is just a hard one")

    print("\nDOES THE SCORE DECAY WITH DISTANCE FROM THE LIVE FINGERPRINT?")
    o = np.argsort(A["dist"])
    for lab, sel in (("closest 10", o[:10]), ("middle 10", o[35:45]),
                     ("furthest 10", o[-10:])):
        print(f"  {lab:12} dist {A['dist'][sel].mean():5.2f}   "
              f"rel {A['rel'][sel].mean():6.2f}  tke {A['tke'][sel].mean():6.2f}  "
              f"mvpe {A['mvpe'][sel].mean():6.2f}")
    c = np.corrcoef(A["dist"], A["sum"])[0, 1]
    print(f"  correlation(distance, sum) = {c:+.3f}"
          f"   {'-- score DOES decay with distance' if c < -0.3 else '-- no decay'}")

    print("\nWORST 8 TRAJECTORIES (do any score as badly as live?)")
    w = np.argsort(A["sum"])[:8]
    print(f"  {'traj':>11}{'Re':>7}{'AoA':>5}{'rel':>8}{'tke':>8}{'mvpe':>8}{'sum':>9}")
    for i in w:
        r = rows[i]
        print(f"  {r['traj']:>11}{r['re']:7.0f}{r['aoa']:5.0f}"
              f"{r['rel']:8.2f}{r['tke']:8.2f}{r['mvpe']:8.2f}{r['sum']:9.2f}")
    print(f"  {'LIVE':>11}{'?':>7}{'15':>5}{LIVE['rel']:8.2f}{LIVE['tke']:8.2f}"
          f"{LIVE['mvpe']:8.2f}{sum(LIVE.values()):9.2f}")


if __name__ == "__main__":
    main()
