"""Numerically verify submissions/v17/submission.py predict() against the numpy evaluation pipeline."""
import sys, os, numpy as np, torch
import sys as _s; _s.dont_write_bytecode = True; _s.path.insert(0, r'../blend')
from fit import *
B17 = os.path.abspath(sys.argv[1])
sys.path.insert(0, B17)
import _blend_tables as TB
import submission as S
X, Y, P, _ = load("final")
if len(sys.argv) > 2: P = np.load(sys.argv[2])
n = 40
X = X[:n]
x3 = np.concatenate([X, np.zeros(X.shape[:-1] + (1,), np.float32)], -1)
out = S.predict(x3)
m, dev = S._state["model"], S._state["device"]
mi, si, mt, st = S._state["norm"]
with torch.no_grad():
    Praw = np.concatenate([(m((torch.from_numpy(x3[i:i+5]) - mi) / si) * st + mt)[..., :2].numpy() for i in range(0, n, 5)])
print("raw FNO vs cache P max diff", np.abs(Praw - P[:n]).max())
E, valid = estimators(X)
# rank-bin edge ties: numpy vs torch sigma_local differ by ~1e-8, which flips a handful of edge elements
xt = torch.from_numpy(X); flt = xt - xt.mean(1, keepdim=True); slt = (flt * flt).mean(1).sqrt()
it = np.concatenate([S._rank_idx(slt[i:i+5], torch.from_numpy(valid[i:i+5])).numpy() for i in range(0, n, 5)])
sl = sigma_local(X).astype(np.float32)
ib = rank_bins(sl, valid, 6)
print("sextile bin mismatches numpy vs torch:", int((it != ib).sum()), "of", it.size)
D = dict(E=E, valid=valid, bins={("sl", 6): it if os.environ.get("TBINS") else ib})
PT = np.array(TB.PRED_TAB); BT = np.array(TB.BAND_TAB)
pred = apply_ls(D, Praw, "both", "sl", 6, PT)
centre = apply_ls(D, Praw, "both", "sl", 6, BT)
if TB.WIDTH_TAB is None:
    h = np.broadcast_to(TB.BAND_G * v8_half(X), pred.shape)
else:
    W = np.array(TB.WIDTH_TAB)
    p1 = np.sqrt(((X[:, 1:] - X[:, :-1]) ** 2).mean(1)).astype(np.float32)
    b = rank_bins(p1, valid, W.shape[-1]); ch = np.arange(2)[None, None, None, :]
    h = np.stack([W[t // 5][ch, b] for t in range(20)], 1)
fin = boost(pred, X)
for k, ref in (("prediction", fin), ("lower", centre - h), ("upper", centre + h)):
    d = np.abs(out[k][..., :2] - ref)
    print(f"{k:10} max|torch-numpy| = {d.max():.3e}   elements >1e-5: {(d > 1e-5).sum()} of {d.size} ({(d > 1e-5).mean():.2e})   max over the rest {d[d <= 1e-5].max():.2e}")
print("p-channel max", np.abs(out["prediction"][..., 2]).max(), "bounds finite", np.isfinite(out["lower"]).all() and np.isfinite(out["upper"]).all(),
      "lower<=upper", bool((out["lower"] <= out["upper"]).all()))
