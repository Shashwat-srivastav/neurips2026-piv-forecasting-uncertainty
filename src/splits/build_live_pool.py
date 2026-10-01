#!/usr/bin/env python3
"""Masterplan v11 sec.2.1: rank AoA-15 trajectories by live-distance (using
the already-computed window_similarity_aoa15.npy), select the 11 most
live-similar as the training pool, holdout and 7575_0 excluded, and verify
the pool's average radial spectrum is within +-30% of live before any
training touches it.

Per-trajectory distance = mean over that trajectory's stride-20 anchor
windows in window_similarity_aoa15.npy (same windows the Gate-3 rerun used),
not the single closest window -- one lucky window should not qualify an
otherwise-distant trajectory.
"""
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spectral_gain import LIVE_SPEC_U, LIVE_SPEC_V, measure_radial_power

ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
# Union of TWO different holdout definitions in this codebase, which
# disagree: linear_micro_gate_aoa15.py's 5-trajectory HOLDOUT omits 25425_15,
# but finetune_fno_kaggle.py's FINAL (the set "touched exactly once" for
# honest reporting) INCLUDES it. Training on 25425_15 here would silently
# contaminate that script's FINAL confirmation. Caught by
# finetune_fno_kaggle.py's --pool-file guard before this pool was ever used.
HOLDOUT = {"21600_15", "22875_15", "20325_15", "24150_15", "19050_15", "25425_15"}
# 7575_0 is a documented byte-duplicate of 6300_0 at AoA 0 -- irrelevant to
# this AoA-15-only pool, kept only so the exclusion list matches the plan's
# wording. 7575_15 has NO documented duplicate issue and is a legitimate
# candidate.
EXCLUDE = {"7575_0"}
N_POOL = 11
TOL = 0.30

d = np.load("data/derived/window_similarity_aoa15.npy", allow_pickle=True).item()
rows, dist = d["rows"], d["dist"]

per_traj = {}
for (nm, _s), dv in zip(rows, dist):
    if nm in HOLDOUT or nm in EXCLUDE:
        continue
    per_traj.setdefault(nm, []).append(dv)

ranked = sorted(per_traj, key=lambda nm: np.mean(per_traj[nm]))
pool = ranked[:N_POOL]

print(f"{len(per_traj)} candidate trajectories after excluding "
      f"{len(HOLDOUT)} holdout + {EXCLUDE & set(per_traj)} exclude")
print("ranked by mean live-distance (closest first):")
for nm in ranked:
    mark = " <- selected" if nm in pool else ""
    print(f"  {nm:10s} mean dist {np.mean(per_traj[nm]):.3f}"
          f"  ({len(per_traj[nm])} windows){mark}")

print(f"\nSELECTED POOL ({N_POOL} trajectories): {pool}")

windows = []
for nm in pool:
    p = os.path.join(ROOT, nm + ".h5")
    with h5py.File(p) as f:
        u = np.asarray(f["u"][:, ::2, ::2], np.float32)
        v = np.asarray(f["v"][:, ::2, ::2], np.float32)
    T = u.shape[0]
    for i in range(0, T - 20 + 1, 20):          # non-overlapping, matches local_ref_diag.py
        windows.append(np.stack([u[i:i + 20], v[i:i + 20], np.zeros_like(u[i:i + 20])], -1))

print(f"\n{len(windows)} non-overlapping 20-frame windows drawn from the pool "
      f"for spectrum verification")

su, sv = measure_radial_power(windows)
ru, rv = su / LIVE_SPEC_U, sv / LIVE_SPEC_V
ok = bool(np.all(np.abs(ru - 1.0) <= TOL) and np.all(np.abs(rv - 1.0) <= TOL))
print(f"pool SPEC_u {su.round(1).tolist()}  live {LIVE_SPEC_U.tolist()}  "
      f"ratio {ru.round(3).tolist()}")
print(f"pool SPEC_v {sv.round(1).tolist()}  live {LIVE_SPEC_V.tolist()}  "
      f"ratio {rv.round(3).tolist()}")
print(f"\nGATE (+-{TOL:.0%}): {'PASS' if ok else 'FAIL'}")

out = {"pool": pool, "holdout": sorted(HOLDOUT), "spec_u_ratio": ru.tolist(),
       "spec_v_ratio": rv.tolist(), "gate_pass": ok}
with open("data/splits/live_pool.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("\nwrote live_pool.json")
if not ok:
    print("STOP per sec.2.1/sec.3: pool spectrum gate failed, do not train on it "
          "without revisiting the selection.")
