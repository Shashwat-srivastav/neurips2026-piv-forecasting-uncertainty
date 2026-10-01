"""v18 study helpers. Imports (read-only) experiments/estimator_blend + experiments/interval_recipes; never writes there."""
import os, sys
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "estimator_blend"))
sys.path.insert(0, os.path.join(HERE, "..", "interval_recipes"))
from base import load, sc, SIG, v8_half, sigma_local, estimators, rank_bins, boost, score, s_err, W8  # noqa
import pickle

KREL = {"select": 1.534, "final": 1.665}
KT = {"select": 1.394, "final": 1.447}
W5 = dict(rel=.454, tke=.087, mvpe=.080, sps=.284)
RANK, RHO = 6, 0.95


def filled(X, Em):
    nz = X != 0
    return np.where(nz, X, Em[:, None]).astype(np.float32)


def dmd_forecast(X, Em, rank=RANK, rho=RHO, horizon=20, batch=64):
    """Coherent rank-r DMD forecast of the (mask-filled) input fluctuation, u,v jointly. (n,20,H,W,2)."""
    n, t_in, h, w, c = X.shape
    out = np.empty((n, horizon, h, w, c), np.float32)
    for i in range(0, n, batch):
        Xf = filled(X[i:i+batch], Em[i:i+batch])
        flat = (Xf - Em[i:i+batch, None]).reshape(len(Xf), t_in, -1).astype(np.float64)
        fc = dmd_op(flat, horizon, rank, rho)
        out[i:i+batch] = np.matmul(fc, flat).reshape(len(Xf), horizon, h, w, c)
    return out


def dmd_op(flat, horizon, rank, rho):
    n, t_in, _ = flat.shape
    gram = np.matmul(flat, flat.swapaxes(1, 2))
    ev, V = np.linalg.eigh(gram)
    ev = ev[:, ::-1][:, :rank]; V = V[:, :, ::-1][:, :, :rank]
    s = np.sqrt(np.clip(ev, 0, None))
    keep = s > 1e-8 * np.maximum(s[:, :1], 1e-30)
    ss = np.where(keep, s, 1.0)
    coef = np.einsum("nr,ntr->nrt", s, V)
    A = np.einsum("nrt,nts->nrs", coef[:, :, 1:], np.linalg.pinv(coef[:, :, :-1]))
    lam, W = np.linalg.eig(A)
    mag = np.abs(lam)
    lam = np.where(mag > rho, lam / np.maximum(mag, 1e-12) * rho, lam)
    amp = np.einsum("nrs,ns->nr", np.linalg.pinv(W), coef[:, :, -1].astype(complex))
    steps = np.arange(1, horizon + 1)
    traj = np.einsum("nrs,nsk->nrk", W, amp[:, :, None] * lam[:, :, None] ** steps).real
    wts = np.where(keep[:, None, :], V / ss[:, None, :], 0.0)
    fc = np.einsum("nrk,ntr->nkt", traj, wts)
    fc[~np.isfinite(fc).all(axis=(1, 2))] = 0.0
    return fc


def lin_trend(X, Em):
    """Per-pixel LS line over the (filled) window: returns (end value - Em, slope per frame)."""
    Xf = filled(X, Em)
    t = np.arange(X.shape[1], dtype=np.float32); t = t - t.mean()
    slope = np.einsum("t,nthwc->nhwc", t, Xf) / float((t * t).sum())
    end = Em + slope * t[-1]
    return (end - Em).astype(np.float32), slope.astype(np.float32)


def prep(role, Pover=None):
    X, Y, P, traj = load(role)
    if Pover is not None:
        P = Pover
    E, valid = estimators(X)
    Em, El = E["mean"], E["last"]
    D = dict(X=X, Y=Y, P=P, Em=Em, El=El, valid=valid, role=role)
    Xf = filled(X, Em)
    D["El5"] = Xf[:, -5:].mean(1)
    D["Lend"], D["slope"] = lin_trend(X, Em)
    D["dmd"] = dmd_forecast(X, Em)
    D["sl"] = sigma_local(X).astype(np.float32)
    D["bins"] = rank_bins(D["sl"], valid, 6)
    p1 = np.sqrt(((X[:, 1:] - X[:, :-1]) ** 2).mean(1)).astype(np.float32)
    D["p1b"] = rank_bins(p1, valid, 10)
    D["h8"] = v8_half(X)
    return D


def targets(D, name):
    Y, P = D["Y"], D["P"]; role = D["role"]
    k = {"A": KREL[role], "AT": KT[role]}.get(name)
    if name in ("A", "AT"):
        return np.where(Y == 0.0, 0.0, P + k * (Y - P)).astype(np.float32), P
    if name in ("B", "BT"):
        k = KT[role] if name == "BT" else 0.12393 / float(sc.rel_l2_per_sample(P, Y, 2).mean())
        return Y, (Y + k * (P - Y)).astype(np.float32)
    if name == "R":
        return Y, P
    raise KeyError(name)


def regressors(D, Pt, t, names):
    """list of (n,H,W,2) regressors minus Em for lead t."""
    Em = D["Em"]; out = []
    for nm in names:
        if nm == "P": out.append(Pt[:, t] - Em)
        elif nm == "El": out.append(D["El"] - Em)
        elif nm == "El5": out.append(D["El5"] - Em)
        elif nm == "Lend": out.append(D["Lend"])
        elif nm == "slope": out.append(D["slope"])
        elif nm == "dmd": out.append(D["dmd"][:, t])
        else: raise KeyError(nm)
    return out


def fit_ls(D, names, prx, ridge=1e-6, nb=6):
    """tab (20,2,nb,len(names)) LS weights of Y'-Em on regressors, per (lead,ch,sextile)."""
    k = len(names); tab = np.zeros((20, 2, nb, k))
    for t in range(20):
        G = np.zeros((2, nb, k, k)); b = np.zeros((2, nb, k))
        for pn in prx:
            Yt, Pt = targets(D, pn)
            y = Yt[:, t]; m = (y != 0) & D["valid"]
            R = regressors(D, Pt, t, names); r = y - D["Em"]
            for c in (0, 1):
                mc = m[..., c]; g = D["bins"][..., c][mc]
                Rc = [q[..., c][mc].astype(np.float64) for q in R]; rc = r[..., c][mc].astype(np.float64)
                for i in range(k):
                    b[c, :, i] += np.bincount(g, Rc[i] * rc, nb)
                    for j in range(i, k):
                        G[c, :, i, j] += np.bincount(g, Rc[i] * Rc[j], nb)
                        G[c, :, j, i] = G[c, :, i, j]
        for c in (0, 1):
            for q in range(nb):
                A_ = G[c, q] + ridge * np.trace(G[c, q]) / k * np.eye(k)
                tab[t, c, q] = np.linalg.solve(A_, b[c, q])
    return tab


def apply_tab(D, Pt, names, tab):
    ch = np.arange(2)[None, None, None, :]
    out = np.empty_like(Pt)
    for t in range(20):
        R = regressors(D, Pt, t, names)
        o = D["Em"].copy()
        for i, q in enumerate(R):
            o = o + tab[t, ch, D["bins"], i] * q
        out[:, t] = np.where(D["valid"], o, Pt[:, t])
    return out.astype(np.float32)


def opt_h(ae):
    e = np.sort(ae); m = len(e)
    return float(e[np.argmax(np.exp(-2 * e / SIG) * np.arange(1, m + 1) / m)]) if m else 0.01


def fit_width(D, centre_fn, prx, nb=10):
    H = np.zeros((4, 2, nb)); errs = {}
    for pn in prx:
        Yt, Pt = targets(D, pn); c = centre_fn(Pt)
        ae = np.abs(Yt - c); scm = (Yt != 0) & D["valid"][:, None]
        for lg in range(4):
            for ch in (0, 1):
                a = ae[:, lg*5:(lg+1)*5, ..., ch]; m = scm[:, lg*5:(lg+1)*5, ..., ch]
                b = np.broadcast_to(D["p1b"][:, None, ..., ch], a.shape)
                for q in range(nb):
                    errs.setdefault((lg, ch, q), []).append(a[m & (b == q)])
    for key, v in errs.items(): H[key] = opt_h(np.concatenate(v))
    return H


def width_of(D, H):
    h = np.empty(D["X"].shape, np.float32); ch = np.arange(2)[None, None, None, :]
    for t in range(20): h[:, t] = H[t // 5][ch, D["p1b"]]
    return h


def delta(s, b):
    d = dict(rel=s_err(s["rel"]) - s_err(b["rel"]), tke=s_err(s["tke"]) - s_err(b["tke"]),
             mvpe=s_err(s["mvpe"]) - s_err(b["mvpe"]), sps=s["sps"] - b["sps"], T=s["T"] - b["T"])
    d["comp"] = sum(W5[k] * d[k] for k in W5)
    return d


def fmt(d):
    return f"rel {d['rel']:+.3f} tke {d['tke']:+.3f} mvpe {d['mvpe']:+.3f} sps {d['sps']:+.3f} T {d['T']:+.4f} comp {d['comp']:+.3f}"
