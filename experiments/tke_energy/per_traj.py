"""Per-trajectory robustness of the two candidates on FINAL (rel/tke deltas; mvpe is invariant)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_rescale import C, sc, apply_gain, gain_input
X, Y, P, tr = C.load("final")
cands = {"G1.15": apply_gain(P, 1.15), "I0.5": apply_gain(P, gain_input(X, P, 0.5))}
a = gain_input(X, P, 0.5).ravel(); print(f"I0.5 per-window gain: median {np.median(a):.3f} p10 {np.percentile(a,10):.3f} p90 {np.percentile(a,90):.3f}")
fx = X - X.mean(1, keepdims=True); fp = P - P.mean(1, keepdims=True)
for t in sorted(set(tr)):
    m = tr == t
    vr = np.sqrt((fp[m]**2).mean() / (fx[m]**2).mean())
    r0 = sc.score_error(sc.rel_l2_per_sample(P[m], Y[m], 2).mean()); k0 = sc.score_error(sc.tke_rel_l2_per_sample(P[m], Y[m], 2).mean())
    s = f"{t:9} vr {vr:.3f}"
    for n, Pn in cands.items():
        r = sc.score_error(sc.rel_l2_per_sample(Pn[m], Y[m], 2).mean()); k = sc.score_error(sc.tke_rel_l2_per_sample(Pn[m], Y[m], 2).mean())
        s += f" | {n} drel {r-r0:+.3f} dtke {k-k0:+.3f}"
    print(s)
