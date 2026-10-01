"""Per-window compliance test for a build (copied into submissions/v17_per_window and submissions/v19_per_window).
usage: python test_independence.py        (run from anywhere; CPU is fine)

1. INDEPENDENCE: for real FINAL windows, predict() on a batch of 5 (and on shuffled / mixed /
   short / 10-window calls) must give, for every window, exactly what predict() gives on that
   window ALONE: max |diff| < 1e-6 on prediction, lower and upper.
2. TORCH vs NUMPY: predict() vs the numpy pipeline (experiments/per_window_rebuild/blend19.py, per-window rank bins)
   on 40 FINAL windows."""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PW = os.path.abspath(os.path.join(HERE, "..", "per_window_rebuild"))
sys.path.insert(0, HERE)
import numpy as np, torch
import submission as S
import _blend_tables as TB
sys.path.insert(1, PW)
from blend19 import prep, apply, width_of, boost  # noqa: E402  (per-window numpy mirror)

TOL = 1e-6
D = prep("final", None, 1)
X = D["X"]
x3 = np.concatenate([X, np.zeros(X.shape[:-1] + (1,), np.float32)], -1)

def maxdiff(o, ref, rows):
    return max(float(np.abs(o[k] - ref[k][rows]).max()) for k in ("prediction", "lower", "upper"))

# ---- 1. independence ----
n1 = 20
single = {k: [] for k in ("prediction", "lower", "upper")}
for i in range(n1):
    o = S.predict(x3[i:i + 1])
    for k in single: single[k].append(o[k][0])
single = {k: np.stack(v) for k, v in single.items()}
rng = np.random.default_rng(0)
worst = 0.0; lines = []
cases = [("batch5 [%d:%d]" % (i, i + 5), np.arange(i, i + 5)) for i in range(0, n1, 5)]
cases += [("shuffled5", rng.permutation(n1)[:5]), ("mixed5", np.array([0, 19, 7, 12, 3])),
          ("short3 (padded)", np.array([4, 9, 14])), ("batch10 (2 chunks)", np.arange(10)),
          ("dup5", np.array([6, 6, 1, 6, 2]))]
for name, rows in cases:
    o = S.predict(x3[rows])
    d = maxdiff(o, single, rows); worst = max(worst, d)
    lines.append(f"  {name:20s} rows {list(rows)}  max|batched - alone| = {d:.3e}")
print("INDEPENDENCE (prediction/lower/upper vs each window predicted alone):")
print("\n".join(lines))
print(f"  worst {worst:.3e}  -> {'PASS' if worst < TOL else 'FAIL'} (tol {TOL:g})")
# contrast: how much the SHIPPED 5-window binning moved the same windows is reported by eval, not here.

# ---- 2. torch vs numpy ----
n = 40
for k in ("X", "Y", "P", "Em", "El", "valid", "bins", "p1b", "h8"): D[k] = D[k][:n]
out = S.predict(x3[:n])
m = S._state["model"]; mi, si, mt, st = S._state["norm"]
with torch.no_grad():
    Praw = np.concatenate([(m((torch.from_numpy(x3[i:i + 5]) - mi) / si) * st + mt)[..., :2].numpy()
                           for i in range(0, n, 5)])
pt, bt, H = np.array(TB.PRED_TAB), np.array(TB.BAND_TAB), np.array(TB.WIDTH_TAB)
if pt.shape[-1] == 3:                                   # v19: numpy ridge L with the build's weights
    from ridge import predict_L
    from lib19 import feat_index
    rw = np.load(os.path.join(HERE, "ridge_w.npz"))
    Wp = rw["W"].reshape(16, -1, 2, 6, 20).transpose(2, 3, 0, 1, 4).reshape(2, 96, -1, 20)
    D["L"] = predict_L(X[:n], feat_index(int(rw["m"]), int(rw["k"])), Wp.astype(np.float64))
# bins as torch computes them (numpy vs torch sigma_local differ ~1e-8 -> rare edge flips)
xt = torch.from_numpy(X[:n]); flt = xt - xt.mean(1, keepdim=True); slt = (flt * flt).mean(1).sqrt()
tb = S._rank_idx(slt, torch.from_numpy(D["valid"])).numpy()
print(f"TORCH vs NUMPY (n={n}): sextile bin mismatches {int((tb != D['bins']).sum())} of {tb.size}")
pt1 = xt[:, 1:] - xt[:, :-1]; p1t = (pt1 * pt1).mean(1).sqrt()
tp = S._rank_idx(p1t, (xt != 0).any(1), H.shape[-1]).numpy()
print(f"  p1 decile bin mismatches {int((tp != D['p1b']).sum())} of {tp.size}")
for label, bins, p1b in (("numpy bins", D["bins"], D["p1b"]), ("torch bins", tb, tp)):
    D["bins"], D["p1b"] = bins, p1b
    if "L" in D and label == "torch bins":
        D["L"] = predict_L(X[:n], feat_index(int(rw["m"]), int(rw["k"])), Wp.astype(np.float64), bins=torch.from_numpy(tb))
    pred = boost(apply(D, Praw, pt), X[:n]); c = apply(D, Praw, bt); h = width_of(D, H)
    for k, ref in (("prediction", pred), ("lower", c - h), ("upper", c + h)):
        d = np.abs(out[k][..., :2] - ref)
        print(f"  [{label}] {k:10} max|torch-numpy| {d.max():.3e}  elements>1e-5: {(d > 1e-5).sum()} of {d.size}")
print("  finite", all(np.isfinite(out[k]).all() for k in out), "lo<=up", bool((out["lower"] <= out["upper"]).all()),
      "p max", float(np.abs(out["prediction"][..., 2]).max()))
sys.exit(0 if worst < TOL else 1)
