"""Run submissions/build_exp1/submission.py on all FINAL windows: score it, and check per-window independence."""
import os, sys, numpy as np
sys.path.insert(0, "submissions/build_exp1"); sys.path.insert(1, "experiments/per_window_rebuild")
import submission as S, blend19 as B
d = np.load("experiments/interval_recipes/cache_final_s20.npz"); X, Y = d["X"], d["Y"]
x3 = np.concatenate([X, np.zeros(X.shape[:-1] + (1,), np.float32)], -1)
outs = [S.predict(x3[i:i + 5]) for i in range(0, len(x3), 5)]
O = {k: np.concatenate([o[k] for o in outs]) for k in ("prediction", "lower", "upper")}
assert all(np.isfinite(O[k]).all() for k in O) and (O["lower"] <= O["upper"]).all()
lo, up, pr = O["lower"][..., :2], O["upper"][..., :2], O["prediction"][..., :2]
z = B.score(pr, Y, (lo + up) / 2, (up - lo) / 2)
sub = lambda e: 100 / (1 + 0.5 * e)
print(f"BUILD on FINAL: rel {sub(z['rel']):.3f} tke {sub(z['tke']):.3f} mvpe {sub(z['mvpe']):.3f} sps {z['sps']:.2f}  (expect 94.155/79.178/95.504/37.70)")
w = 0.0
for rows in (np.arange(0, 5), np.array([7, 3, 11, 19, 2]), np.array([4, 9])):
    ob = S.predict(x3[rows])
    for j, r in enumerate(rows):
        oa = S.predict(x3[r:r + 1])
        w = max(w, max(float(np.abs(ob[k][j] - oa[k][0]).max()) for k in ob))
print(f"INDEPENDENCE worst |batched - alone| = {w:.2e} -> {'PASS' if w < 1e-6 else 'FAIL'}")
