"""Verify experiments/v19_ridge_partner/build_v19/submission.py predict() against the numpy pipeline of blend19.py (L_con tables)."""
import sys, os, pickle, time
sys.dont_write_bytecode = True
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
from blend19 import prep, apply, width_of, boost
B19 = os.path.abspath(os.path.join(HERE, "build_v19")); sys.path.insert(0, B19)
import submission as S
n = 40
D = prep("final", np.load("L_sxb3_final.npy"))
for k in ("X", "Y", "P", "Em", "El", "valid", "bins", "p1b", "h8", "L"): D[k] = D[k][:n]
X = D["X"]; x3 = np.concatenate([X, np.zeros(X.shape[:-1] + (1,), np.float32)], -1)
out = S.predict(x3)
m = S._state["model"]; mi, si, mt, st = S._state["norm"]
with torch.no_grad():
    Praw = np.concatenate([(m((torch.from_numpy(x3[i:i+5]) - mi) / si) * st + mt)[..., :2].numpy() for i in range(0, n, 5)])
pt, bt, H = pickle.load(open("blend_sxb3.pkl", "rb"))["tabs"]["L_con"]
pred = boost(apply(D, Praw, pt), X); c = apply(D, Praw, bt); h = width_of(D, H)
for k, ref in (("prediction", pred), ("lower", c - h), ("upper", c + h)):
    d = np.abs(out[k][..., :2] - ref)
    print(f"{k:10} max|torch-numpy| {d.max():.3e}  elements>1e-5: {(d > 1e-5).sum()} of {d.size}")
print("finite", all(np.isfinite(out[k]).all() for k in out), "lo<=up", bool((out["lower"] <= out["upper"]).all()),
      "shape", out["prediction"].shape, "p max", np.abs(out["prediction"][..., 2]).max())
p5 = x3[:5]
for _ in range(3): S.predict(p5)
t = time.perf_counter()
for _ in range(10): S.predict(p5)
print("cpu ms/sample full predict", (time.perf_counter() - t) / 50 * 1000)
