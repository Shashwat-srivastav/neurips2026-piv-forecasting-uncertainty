"""Centre shifting stacked on the best width recipe (ridge-logfeature lookup), under the proxy.
centre = P + a_g*(inputmean - P) + b_g*(lastframe - P), per channel and lead group g in {0, 1-4, 5-9, 10-19}."""
import os, numpy as np
from rlib import *

C = {k: np.load(os.path.join(D, f"{k[0]}_{k[1]}_centre.npz")) for k in NPZ}
def grp(t): return np.select([t == 0, t < 5, t < 10], [0, 1, 2], 3)
def fit_ab(e, cm, cl, g, mode):
    ab = np.zeros((4, 2))
    for j in range(4):
        m = g == j; A = np.stack([cm[m], cl[m]], 1).astype(np.float64)
        if mode == "l2":
            ab[j] = np.linalg.lstsq(A, e[m].astype(np.float64), rcond=None)[0]
        else:   # median-regression via IRLS (L1), 15 iters on subsample
            idx = np.random.default_rng(j).choice(m.sum(), min(m.sum(), 400000), replace=False)
            Aa, ee = A[idx], e[m][idx].astype(np.float64); w = np.ones(len(ee))
            for _ in range(15):
                sw = np.sqrt(w); coef = np.linalg.lstsq(Aa * sw[:, None], ee * sw, rcond=None)[0]
                w = 1 / np.maximum(np.abs(ee - Aa @ coef), 1e-4)
            ab[j] = coef
    return ab
