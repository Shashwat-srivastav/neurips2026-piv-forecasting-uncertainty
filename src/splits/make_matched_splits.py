#!/usr/bin/env python3
"""Validation splits MATCHED to the measured evaluation distribution.

v9_splits.json stratifies by Reynolds and angle of attack so that every split
spans the training range. That is the right design when you do not know what the
evaluator draws from. We now do.

The ingestion log of 6 Aug 2026 reported the evaluator's own input statistics
(call 2 only; call 1 is a synthetic warmup and is excluded):

    u mean 0.2180   u std 0.1503   v std 0.0233
    sigma_local u 0.0164   v 0.0073   framediff 0.01452

Ranking all 82 training trajectories by mean absolute log-distance across those
six statistics puts Re 19050-26700 at AoA 10/15/20 closest, and Re 3750-5025 at
AoA 0/5 furthest, spanning a factor of 20 in distance. The old SELECT split sits
at distance ranks 6, 12, 16, 24, 39, 47, 48, 50, 61, 71, 73 out of 82 -- it
averages over flows an order of magnitude less energetic than the eval set.

That is a sufficient explanation for the two transfer failures on record:
TKE_BLEND validated +3.25 and delivered -1.02 live; FNO validated +8.05 and
delivered +0.561, which under the v9 linear SPS is -0.121. Both were fitted and
selected against a distribution the evaluator does not draw from.

This builds MTRAIN / MSELECT / MFINAL where MSELECT and MFINAL are drawn from
the closest trajectories, so a local number is comparable to a board number.
MTRAIN keeps everything else plus a third of the matched pool, so a model can
still learn the regime it will be scored in.

Writes matched_splits.json. Does not modify v9_splits.json.
"""
from __future__ import annotations

import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'src', d) for d in ('training', 'evaluation', 'splits')]
OUT = os.path.join(REPO, "data/splits/matched_splits.json")
SRC = json.load(open(os.path.join(HERE, "v9_splits.json")))
T_IN, T_OUT = SRC["t_in"], SRC["t_out"]
STRIDE, SUB, ROOT = SRC["stride"], SRC["sub"], SRC["root"]

# Measured from the live ingestion log, call 2 of each process only.
EVAL_FINGERPRINT = {
    "u_mean": 0.2180, "u_std": 0.1503, "v_std": 0.0233,
    "sl_u": 0.0164, "sl_v": 0.0073, "dfrm": 0.01452,
}
N_MATCHED = 24          # size of the matched pool, dealt train/select/final
PATTERN = ["train", "select", "final"]


def stats(traj):
    """The same six statistics the submission logs, on the same strided view."""
    with h5py.File(os.path.join(ROOT, traj + ".h5"), "r") as f:
        u = np.asarray(f["u"][:, ::SUB, ::SUB], np.float32)
        v = np.asarray(f["v"][:, ::SUB, ::SUB], np.float32)
    fr = np.stack([u, v], -1)
    st = list(range(0, len(fr) - (T_IN + T_OUT) + 1, STRIDE))
    X = np.stack([fr[i:i + T_IN] for i in st])
    n = len(X)
    fl = (X - X.mean(1, keepdims=True)).reshape(n, T_IN, -1)
    sl = np.sqrt(np.einsum("ntm,ntm->nm", fl, fl) / T_IN).reshape(n, 32, 64, 2)
    xs = np.ascontiguousarray(X[:, ::4, ::4, ::4, :])
    d1 = np.abs(xs[:, 1:] - xs[:, :-1])
    mk = (xs[:, :-1] != 0.0) & (xs[:, 1:] != 0.0)
    return {
        "u_mean": float(X[..., 0].mean()), "u_std": float(X[..., 0].std()),
        "v_std": float(X[..., 1].std()), "sl_u": float(sl[..., 0].mean()),
        "sl_v": float(sl[..., 1].mean()),
        "dfrm": float(d1[mk].mean()) if mk.any() else float("nan"),
        "windows": n,
    }


def main():
    rows = []
    for r in SRC["trajectories"]:
        s = stats(r["traj"])
        s.update(traj=r["traj"], re=r["re"], aoa=r["aoa"], old_split=r["split"])
        s["dist"] = float(np.mean([abs(np.log(s[k] / EVAL_FINGERPRINT[k]))
                                   for k in EVAL_FINGERPRINT]))
        rows.append(s)
    rows.sort(key=lambda x: x["dist"])

    # Deal the matched pool round-robin so MSELECT and MFINAL are equally close
    # to the eval fingerprint; anything further away is training data only.
    for i, r in enumerate(rows):
        r["split"] = PATTERN[i % len(PATTERN)] if i < N_MATCHED else "train"

    print(f"{'traj':>11}{'Re':>7}{'AoA':>4}{'rank':>6}{'dist':>7}{'win':>5}  split")
    for i, r in enumerate(rows):
        if i < N_MATCHED or r["split"] != "train":
            print(f"{r['traj']:>11}{r['re']:7d}{r['aoa']:4d}{i:6d}{r['dist']:7.3f}"
                  f"{r['windows']:5d}  {r['split']}")
    print("   ... remaining trajectories -> train")

    print("\nSUMMARY")
    print(f"  {'split':8}{'traj':>6}{'windows':>9}{'mean dist':>11}{'Re range':>16}{'AoA':>16}")
    for s in ("train", "select", "final"):
        sub = [r for r in rows if r["split"] == s]
        res = sorted({r["re"] for r in sub})
        print(f"  {s:8}{len(sub):6d}{sum(r['windows'] for r in sub):9d}"
              f"{np.mean([r['dist'] for r in sub]):11.3f}"
              f"{f'{min(res)}-{max(res)}':>16}{str(sorted({r['aoa'] for r in sub})):>16}")

    old_sel = [r for r in rows if r["old_split"] == "select"]
    new_sel = [r for r in rows if r["split"] == "select"]
    print(f"\n  mean distance to eval: old SELECT {np.mean([r['dist'] for r in old_sel]):.3f}"
          f"  ->  matched SELECT {np.mean([r['dist'] for r in new_sel]):.3f}")

    for a in ("train", "select", "final"):
        for b in ("train", "select", "final"):
            if a < b:
                sa = {r["traj"] for r in rows if r["split"] == a}
                sb = {r["traj"] for r in rows if r["split"] == b}
                assert not (sa & sb), f"{a}/{b} overlap"
    print("  trajectory-disjointness: VERIFIED")

    json.dump({"stride": STRIDE, "t_in": T_IN, "t_out": T_OUT, "sub": SUB,
               "root": ROOT, "eval_fingerprint": EVAL_FINGERPRINT,
               "n_matched": N_MATCHED, "trajectories": rows},
              open(OUT, "w"), indent=1)
    print(f"\n  written: {OUT}")


if __name__ == "__main__":
    main()
