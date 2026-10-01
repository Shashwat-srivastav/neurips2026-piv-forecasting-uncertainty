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
def shifted(ab, cm, cl, g): return ab[g, 0] * cm + ab[g, 1] * cl
for tag, kf, ke in (("rel/rel", KR["select"], KR["final"]), ("T/T", KT["select"], KT["final"]), ("fit T, eval 2.0", KT["select"], 2.0)):
    print(f"### proxy {tag}: k_fit {kf:.3f} k_eval {ke:.3f}")
    for mode, kcoef in (("none", None), ("l2", "proxy"), ("l1", "proxy"), ("l1", "raw")):
        hs, aes = [], []
        for c in (0, 1):
            cs, cf = C[("select", c)], C[("final", c)]
            gs, gf = grp(dat[("select", c)]["t"]), grp(dat[("final", c)]["t"])
            es, ef = cs["e"] * kf, cf["e"] * ke
            if mode == "none":
                ab = np.zeros((4, 2))
            else:
                ab = fit_ab(es if kcoef == "proxy" else cs["e"], cs["cm"], cs["cl"], gs, mode)
            rs = np.abs(es - shifted(ab, cs["cm"], cs["cl"], gs)); rf = np.abs(ef - shifted(ab, cf["cm"], cf["cl"], gf))
            e_, h_ = lookup(dat[("select", c)]["ridge"], rs, 40)
            hs.append(h_[np.searchsorted(e_, dat[("final", c)]["ridge"])]); aes.append(rf)
            if tag == "rel/rel": print(f"   ch {c} {mode}/{kcoef} coefs (a,b) per lead grp:", np.round(ab, 3).tolist())
        T = Tall(hs, aes)
        print(f"  width=ridge lookup, centre {mode:4} coef-fit {kcoef}:  T {T:.4f} (u {Tsum(hs[0], aes[0])/len(aes[0]):.4f} v {Tsum(hs[1], aes[1])/len(aes[1]):.4f})", flush=True)
