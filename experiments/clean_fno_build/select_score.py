"""Run-1 build (raw + interval tables) scored per AoA on FINAL, vs v17 LIVE. Live eval set is AoA 15 only."""
import os, sys, numpy as np
sys.path.insert(0, "submissions/build_exp1"); sys.path.insert(1, "experiments/per_window_rebuild")
import submission as S, blend19 as B
d = np.load("experiments/interval_recipes/cache_select_s20.npz"); X, Y, T = d["X"], d["Y"], d["traj"]
x3 = np.concatenate([X, np.zeros(X.shape[:-1] + (1,), np.float32)], -1)
outs = [S.predict(x3[i:i + 5]) for i in range(0, len(x3), 5)]
O = {k: np.concatenate([o[k] for o in outs])[..., :2] for k in ("prediction", "lower", "upper")}
sub = lambda e: 100 / (1 + 0.5 * e)
W = dict(rel=.453, tke=.087, mvpe=.081, sps=.284); live = dict(rel=94.104, tke=75.368, mvpe=93.085, sps=37.287)
aoa = np.array([t.split("_")[1] for t in T])
for a in ("all",):
    i = np.arange(len(T)) if a == "all" else np.where(aoa == a)[0]
    lo, up, pr = O["lower"][i], O["upper"][i], O["prediction"][i]
    z = B.score(pr, Y[i], (lo + up) / 2, (up - lo) / 2)
    r = dict(rel=sub(z["rel"]), tke=sub(z["tke"]), mvpe=sub(z["mvpe"]), sps=z["sps"])
    print(f"AoA {a:>3} ({len(set(T[i]))} traj, {len(i)} win): rel {r['rel']:.3f} tke {r['tke']:.3f} mvpe {r['mvpe']:.3f} sps {r['sps']:.2f}"
          f" | vs v17 LIVE {sum(W[k]*(r[k]-live[k]) for k in W):+.3f}", flush=True)
print("trajectories:", sorted(set(T)))
