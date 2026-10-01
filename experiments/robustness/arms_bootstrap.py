"""Baseline arms and trajectory-cluster bootstrap CIs (raw forecasts, no intervals).

Arms on SELECT and FINAL: official FNO (P), our clean sim->real FNO Run 1 (Q, never saw either split), persistence
(last input frame) and the window mean. Scores use the official per-window errors mapped by 100/(1+0.5e) after
averaging over windows. The bootstrap resamples whole trajectories (B = 4000).
Also: the memorisation differential. drop = live - FINAL per subscore; D = drop_clean - drop_official
= (live_clean - live_official) - (FINAL_clean - FINAL_official). The live terms are single numbers; the CI comes from
the FINAL term. Output: arms_bootstrap.json
"""
import gc, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "estimator_blend"))
from base import sc, load   # noqa: E402

R1 = np.load(os.path.join(HERE, "..", "..", "report", "data", "run1_preds.npz"))
LIVE = {"official": (94.168, 74.026, 92.836), "clean": (93.728, 77.348, 92.378)}   # raw forecasts, live (v8 / exp1_sf)
B = 4000
rng = np.random.default_rng(0)
s_of = lambda e: 100.0 / (1.0 + 0.5 * e)   # noqa: E731
out = {}
for role in ("select", "final"):
    X, Y, P, traj = load(role)
    arms = {"official": P, "clean": R1[role].astype(np.float32),
            "persistence": np.repeat(X[:, -1:], 20, 1), "window_mean": np.repeat(X.mean(1, keepdims=True), 20, 1)}
    E = {}
    for name, A in arms.items():
        E[name] = np.stack([sc.rel_l2_per_sample(A, Y, 2), sc.tke_rel_l2_per_sample(A, Y, 2),
                            sc.mvpe_rel_l2_per_sample(A, Y)], 1).astype(np.float64)   # (n, 3)
    del X, Y, P, arms; gc.collect()
    names = list(dict.fromkeys(traj.tolist())); groups = [np.where(traj == n)[0] for n in names]
    point = {k: [round(float(s_of(v)), 3) for v in e.mean(0)] for k, e in E.items()}
    draws = {k: [] for k in E}
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for k, e in E.items():
            draws[k].append(s_of(e[idx].mean(0)))
    draws = {k: np.array(v) for k, v in draws.items()}   # (B, 3)
    ci = lambda a: [[round(float(x), 3) for x in np.percentile(a, 2.5, 0)], [round(float(x), 3) for x in np.percentile(a, 97.5, 0)]]  # noqa: E731
    r = dict(n_windows=int(len(traj)), n_traj=len(names), point=point, ci={k: ci(v) for k, v in draws.items()})
    d = draws["clean"] - draws["official"]
    r["clean_minus_official"] = dict(point=[round(a - b, 3) for a, b in zip(point["clean"], point["official"])], ci=ci(d),
                                     p_clean_better=[round(float(x), 3) for x in (d > 0).mean(0)])
    if role == "final":
        live_d = np.array(LIVE["clean"]) - np.array(LIVE["official"])
        D = live_d[None, :] - d                                        # drop_clean - drop_official
        r["memorisation_differential"] = dict(
            point=[round(float(a), 3) for a in live_d - (np.array(point["clean"]) - np.array(point["official"]))],
            ci=ci(D), p_positive=[round(float(x), 3) for x in (D > 0).mean(0)])
    out[role] = r
    print(role, json.dumps(r, indent=None)[:900], flush=True)
    del E, draws; gc.collect()
json.dump(out, open(os.path.join(HERE, "arms_bootstrap.json"), "w"), indent=1)
print("saved")
