"""Shared: v17 recipe refit with PER-WINDOW rank bins (batch=1) everywhere, incl. v8 band shape."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "estimator_blend"))
import base
from base import sc, SIG, sigma_local, estimators, d2rms, boost, score, delta, s_err, W8
from common import v8_half, fit_k, load
from fit import fit_ls, apply_ls
from fit_band import fit_band
KA = {"select": 1.534, "final": 1.665}; KT = {"select": 1.394, "final": 1.447}

def rank_bins1(F, valid, nb): return base.rank_bins(F, valid, nb, batch=1)

def v8_T(Y, P, h8, k):
    Yp = np.where(Y == 0, 0, P + k * (Y - P)); ins = np.abs(Yp - P) <= h8; s = Yp != 0
    return float(np.sum(np.where(ins, np.exp(-2 * np.broadcast_to(h8, P.shape) / SIG), 0), where=s) / s.sum())

def prep(X, Y, P, ka=None, kt=None, only=None):
    E, valid = estimators(X)
    D = dict(X=X, Y=Y, P=P, E=E, valid=valid)
    D["bins"] = {("sl", 6): rank_bins1(sigma_local(X).astype(np.float32), valid, 6)}
    p1 = np.sqrt(((X[:, 1:] - X[:, :-1]) ** 2).mean(1)).astype(np.float32)
    D["p1b"] = rank_bins1(p1, valid, 10)
    D["h8"] = v8_half(X, batch=1)
    if ka is None: ka = fit_k(Y, P, "rel")
    if kt is None:   # match v8 live T = 0.4914 (as experiments/interval_recipes/rlib.k_for_T)
        lo, hi = 0.8, 3.0
        for _ in range(30):
            m = .5 * (lo + hi)
            if v8_T(Y, P, D["h8"], m) > 0.4914: lo = m
            else: hi = m
        kt = .5 * (lo + hi)
    D["k"] = (ka, kt)
    tgt = lambda k: np.where(Y == 0.0, 0.0, P + k * (Y - P)).astype(np.float32)
    kb = base.LIVE_REL / float(sc.rel_l2_per_sample(P, Y, 2).mean())
    mk = dict(A=lambda: (tgt(ka), P), B=lambda: (Y, (Y + kb * (P - Y)).astype(np.float32)),
              AT=lambda: (tgt(kt), P), BT=lambda: (Y, (Y + kt * (P - Y)).astype(np.float32)))
    D["pr"] = {k: f() for k, f in mk.items() if only is None or k in only}
    return D

def opt_h(ae):
    e = np.sort(ae); m = len(e)
    return float(e[np.argmax(np.exp(-2 * e / SIG) * np.arange(1, m + 1) / m)]) if m else 0.01

def fit_width(D, btab, prx=("AT", "BT"), nb=10):
    errs = {}
    for pn in prx:
        Yt, Pt = D["pr"][pn]; c = apply_ls(D, Pt, "both", "sl", 6, btab)
        ae = np.abs(Yt - c); m_ = (Yt != 0) & D["valid"][:, None]
        for lg in range(4):
            for ch in (0, 1):
                a = ae[:, lg*5:(lg+1)*5, ..., ch]; m = m_[:, lg*5:(lg+1)*5, ..., ch]
                b = np.broadcast_to(D["p1b"][:, None, ..., ch], a.shape)
                for k in range(nb): errs.setdefault((lg, ch, k), []).append(a[m & (b == k)])
    H = np.zeros((4, 2, nb))
    for (lg, ch, k), v in errs.items(): H[lg, ch, k] = opt_h(np.concatenate(v))
    return H

def width_of(D, H):
    h = np.empty(D["X"].shape, np.float32); ch = np.arange(2)[None, None, None, :]
    for t in range(20): h[:, t] = H[t // 5][ch, D["p1b"]]
    return h

def fit_recipe(D):
    """v17 recipe: PRED ('both','sl',6) on proxy A; BAND centre on A+B at g=1 over v8 widths; p1 width on AT,BT."""
    pt = fit_ls(D, "both", "sl", 6, ["A"])
    bt = fit_band(D, D["h8"], "sl", 6, ["A", "B"], 1.0)
    return dict(pred=pt, band=bt, width=fit_width(D, bt))

def outputs(F, tabs, pn="AT"):
    Yt, Pt = F["pr"][pn]
    pred = apply_ls(F, Pt, "both", "sl", 6, tabs["pred"]); c = apply_ls(F, Pt, "both", "sl", 6, tabs["band"])
    return Yt, boost(pred, F["X"]), c, width_of(F, tabs["width"])

def score_sub(out, Yt, c, h, idx): return score(out[idx], Yt[idx], c[idx], h[idx])
