"""Per-lead skill of the FNO fluctuation vs phase-consistent periodic persistence of the input.
Period T estimated per window from the input window (spatially averaged autocorr of v fluct), and
fixed T=22 as reference. Skill = pattern correlation of fluctuations about each series' window mean."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "interval_recipes"))
import common as C

def pcorr(a, b):
    a = a.reshape(a.shape[0], a.shape[1], -1); b = b.reshape(b.shape[0], b.shape[1], -1)
    return ((a * b).sum((0, 2)) / np.sqrt((a * a).sum((0, 2)) * (b * b).sum((0, 2))))

for role in ("select",):
    X, Y, P, _ = C.load(role)
    fy = Y - Y.mean(1, keepdims=True); fp = P - P.mean(1, keepdims=True)
    xm = X.mean(1, keepdims=True)
    print(role, "FNO fluct corr by lead :", np.round(pcorr(fp, fy), 2))
    # period from input: autocorr of v-fluct field over lags 8..19 -- 20 frames can't resolve 21.9 well
    fx = (X - xm)[..., 1].reshape(len(X), 20, -1)
    ac = np.array([[(fx[i, :20-l] * fx[i, l:]).sum() / (fx[i] ** 2).sum() * 20 / (20 - l) for l in range(20)] for i in range(len(X))])
    print("input autocorr (mean over windows) lags 0..19:", np.round(ac.mean(0), 2))
    for T in (20, 21, 22, 23):
        Q = np.empty_like(Y)
        for k in range(20):
            j = (20 + k - T)
            while j >= 20: j -= T
            Q[:, k] = X[:, j] if 0 <= j < 20 else X[:, (j % T) if (j % T) < 20 else 19]
        fq = Q - Q.mean(1, keepdims=True)
        print(f"periodic persistence T={T} corr by lead:", np.round(pcorr(fq, fy), 2))
