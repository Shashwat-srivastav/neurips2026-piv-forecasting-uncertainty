#!/usr/bin/env python3
"""Build the holdout the campaign report specified at 12.3 and never ran:

    "The only valid local experiment: hold out >=12 trajectories completely,
    train on the rest, compare against 94.168 / 74.026 / 92.836."

Every prior "local held-out" figure in this campaign used sim_real_ft, which
the organizers fine-tuned on ALL 81 usable train_real trajectories -- so every
one of those numbers was a TRAINING score, not a holdout score, which is why
local deltas have repeatedly failed to transfer live (12.3).

SELECTION RULE: the 14 trajectories with the SMALLEST distance to the live
fingerprint in matched_splits.json's standardized 6-D space (u_mean, u_std,
v_std, sl_u, sl_v, dfrm). This is deliberately the HARDEST version of the
test, not an arbitrary one: these are precisely the trajectories most similar
to live, so if the checkpoint's generalization gap is real, it should show up
here MOST clearly, not least. It also reproduces the existing 4-trajectory
SELECT/FINAL set exactly (they are already the top 4 by construction), so nothing
downstream that references those names breaks.

ROLE SPLIT, honestly caveated:
  SELECT (4) = 22875_15, 24150_15, 21600_15, 25425_15 -- the original
    SELECT/FINAL quad. These were used earlier this campaign to fit the SPS
    BAND multiplier table (v8's header: "fitted on the held-out SELECT pool,
    evaluated on an untouched FINAL pool"). That does not touch model
    weights or rel/tke/mvpe -- bands only reshape lower/upper -- so they stay
    valid for judging model accuracy, just not pristine enough to be the
    "touched once" confirmation set. Used here for epoch-selection only.
  FINAL (10) = every other trajectory in the top-14. Not used ANYWHERE in
    this campaign before now -- true single-touch confirmation set.
"""
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'src', d) for d in ('training', 'evaluation', 'splits')]
ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
EXCLUDE = {"7575_0"}          # byte-identical duplicate of 6300_0

d = json.load(open(os.path.join(REPO, "data/splits/matched_splits.json")))
rows = [r for r in d["trajectories"] if r["traj"] not in EXCLUDE]
rows.sort(key=lambda r: r["dist"])

HOLDOUT_N = 14
# The Re LADDER TOPS OUT at 26700 -- any trajectory at that Re can only ever
# be an extrapolation holdout (no training neighbour exists above it), no
# matter which selection rule picks it. Keep the ceiling Re IN TRAINING so it
# can bracket near-top holdout members from above, and backfill the holdout
# from the next-nearest-by-distance candidates instead. This is a real
# structural fact about the data, not a tuning choice -- min N with margin.
RE_CEILING = max(r["re"] for r in rows)
ranked = [r for r in rows if r["re"] != RE_CEILING]
ceiling_rows = [r for r in rows if r["re"] == RE_CEILING]
holdout_rows = ranked[:HOLDOUT_N]
train_rows = ranked[HOLDOUT_N:] + ceiling_rows
print(f"Re ceiling {RE_CEILING:.0f}: {len(ceiling_rows)} trajectories kept in "
      f"training unconditionally so near-top holdout members stay bracketed.\n")

SELECT = ["22875_15", "24150_15", "21600_15", "25425_15"]
holdout_names = [r["traj"] for r in holdout_rows]
assert set(SELECT) <= set(holdout_names), \
    "the original SELECT/FINAL quad is not inside the new top-14 -- rule changed"
FINAL = [n for n in holdout_names if n not in SELECT]

print(f"usable trajectories: {len(rows)}  (82 - {len(EXCLUDE)} duplicate)")
print(f"holdout: {len(holdout_names)}   train: {len(train_rows)}\n")
print(f"{'role':>7} {'traj':>10} {'re':>7} {'aoa':>4} {'dist':>7}  note")
for r in holdout_rows:
    role = "SELECT" if r["traj"] in SELECT else "FINAL"
    edge = "  <- edge Re, extrapolation not interpolation" \
        if r["re"] in (3750, 26700) else ""
    print(f"{role:>7} {r['traj']:>10} {r['re']:>7.0f} {r['aoa']:>4.0f} "
          f"{r['dist']:>7.3f}{edge}")

# Sanity: confirm every held-out trajectory actually has a real file, and that
# its Re sits BETWEEN two training Re values at the same AoA where possible
# (mirrors the live task's own interpolation geometry -- see data-locations
# memory: eval Re 15225 is bracketed 1275 on each side, never an edge case).
train_re_by_aoa = {}
for r in train_rows:
    train_re_by_aoa.setdefault(r["aoa"], set()).add(r["re"])

print("\nbracketing check (does training retain a neighbour on each side?):")
n_bracketed = 0
for r in holdout_rows:
    res = sorted(train_re_by_aoa.get(r["aoa"], set()))
    below = [x for x in res if x < r["re"]]
    above = [x for x in res if x > r["re"]]
    ok = bool(below) and bool(above)
    n_bracketed += ok
    print(f"  {r['traj']:>10}  nearest below {max(below) if below else None}  "
          f"above {min(above) if above else None}  {'OK' if ok else 'EDGE'}")
print(f"{n_bracketed}/{len(holdout_rows)} bracketed on both sides at the same AoA")

for nm in holdout_names:
    p = os.path.join(ROOT, nm + ".h5")
    assert os.path.exists(p), f"missing file for holdout trajectory {nm}: {p}"
    with h5py.File(p) as f:
        assert f["u"].shape[0] > 40, f"{nm} too short ({f['u'].shape[0]} frames)"

out = dict(select=SELECT, final=FINAL,
           holdout=holdout_names,
           train=[r["traj"] for r in train_rows],
           n_bracketed=int(n_bracketed))
json.dump(out, open(os.path.join(REPO, "data/splits/holdout_v12_3.json"), "w"), indent=1)
print(f"\nwrote holdout_v12_3.json  "
      f"(select={len(SELECT)} final={len(FINAL)} train={len(train_rows)})")
