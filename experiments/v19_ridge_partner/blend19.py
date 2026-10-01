"""v17 recipe with L as a third blend partner. usage: blend19.py L_select.npy L_final.npy [tag]
pred = Em + a(P-Em) + b(El-Em) + c(L-Em) per (lead,ch,sl sextile) [LS fit on SELECT proxy A (k_rel), as v17]
centre = same form, grid max-T with v8 widths on SELECT proxies A,B (k_rel), as v17
width = (lead group, ch, p1 decile) T-optimal on SELECT (AT,BT), as v17; tke boost after.
Scored on FINAL under AT/BT (k=1.447) and A/B (k_rel)."""
import sys, os, pickle, itertools
sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v18_refits"))
from lib18 import load, sc, SIG, v8_half, sigma_local, estimators, rank_bins, boost, score, delta, fmt, opt_h, KREL, KT

def prep(role, L):
    X, Y, P, _ = load(role)
    E, valid = estimators(X)
    D = dict(X=X, Y=Y, P=P, Em=E["mean"], El=E["last"], valid=valid, role=role, L=L)
    D["bins"] = rank_bins(sigma_local(X).astype(np.float32), valid, 6)
    p1 = np.sqrt(((X[:, 1:] - X[:, :-1]) ** 2).mean(1)).astype(np.float32)
    D["p1b"] = rank_bins(p1, valid, 10)
    D["h8"] = v8_half(X)
    return D

def tg(D, pn):
    Y, P = D["Y"], D["P"]; r = D["role"]
    if pn in ("A", "AT"):
        k = KREL[r] if pn == "A" else KT[r]
        return np.where(Y == 0.0, 0.0, P + k * (Y - P)).astype(np.float32), P
    k = KT[r] if pn == "BT" else 0.12393 / float(sc.rel_l2_per_sample(P, Y, 2).mean())
    return Y, (Y + k * (P - Y)).astype(np.float32)

def regs(D, Pt, t, useL):
    R = [Pt[:, t] - D["Em"], D["El"] - D["Em"]]
    if useL: R.append(D["L"][:, t] - D["Em"])
    return R

def fit_pred(D, prx, useL, constrained=True):
    k = 3 if useL else 2
    tab = np.zeros((20, 2, 6, k))
    for t in range(20):
        G = np.zeros((2, 6, k, k)); b = np.zeros((2, 6, k))
        for pn in prx:
            Yt, Pt = tg(D, pn); y = Yt[:, t]; m = (y != 0) & D["valid"]
            R = regs(D, Pt, t, useL); r = y - D["Em"]
            for c in (0, 1):
                mc = m[..., c]; g = D["bins"][..., c][mc]
                Rc = [q[..., c][mc].astype(np.float64) for q in R]; rc = r[..., c][mc].astype(np.float64)
                for i in range(k):
                    b[c, :, i] += np.bincount(g, Rc[i] * rc, 6)
                    for j in range(k):
                        G[c, :, i, j] += np.bincount(g, Rc[i] * Rc[j], 6)
        for c in (0, 1):
            for q in range(6):
                A_, bb = G[c, q], b[c, q]
                if not constrained:
                    tab[t, c, q] = np.linalg.solve(A_ + 1e-9 * np.trace(A_) * np.eye(k), bb); continue
                f = lambda w: w @ A_ @ w - 2 * w @ bb
                gr = lambda w: 2 * A_ @ w - 2 * bb
                best = None
                for x0 in (np.full(k, 1.0 / (k + 1)), np.eye(k)[0] * .9, np.eye(k)[-1] * .9):
                    r_ = minimize(f, x0, jac=gr, method="SLSQP", bounds=[(0, 1)] * k,
                                  constraints=[dict(type="ineq", fun=lambda w: 1 - w.sum(), jac=lambda w: -np.ones(k))])
                    if best is None or r_.fun < best.fun: best = r_
                tab[t, c, q] = best.x
    return tab

def apply(D, Pt, tab):
    ch = np.arange(2)[None, None, None, :]; k = tab.shape[-1]
    out = np.empty_like(Pt)
    for t in range(20):
        R = regs(D, Pt, t, k == 3); o = D["Em"].copy()
        for i, q in enumerate(R): o = o + tab[t, ch, D["bins"], i] * q
        out[:, t] = np.where(D["valid"], o, Pt[:, t])
    return out.astype(np.float32)

def fit_band(D, prx, useL, step=0.1):
    n = int(round(1 / step))
    if useL:
        GRID = [(a * step, b * step, c * step) for a in range(n + 1) for b in range(n + 1 - a) for c in range(n + 1 - a - b)]
    else:
        GRID = [(a * step, b * step) for a in range(n + 1) for b in range(n + 1 - a)]
    GRID = np.array(GRID)
    h = D["h8"][:, 0]; pen = np.exp(-2 * h / SIG)
    tab = np.zeros((20, 2, 6, GRID.shape[1]))
    for t in range(20):
        acc = np.zeros((len(GRID), 2, 6))
        for pn in prx:
            Yt, Pt = tg(D, pn); y = Yt[:, t]; m = (y != 0) & D["valid"]
            R = regs(D, Pt, t, useL); r = y - D["Em"]
            for c in (0, 1):
                mc = m[..., c]; gi = D["bins"][..., c][mc]
                Rc = np.stack([q[..., c][mc] for q in R]); rc = r[..., c][mc]; hc = h[..., c][mc]; pc = pen[..., c][mc]
                for j, w in enumerate(GRID):
                    ins = np.abs(rc - w @ Rc) <= hc
                    acc[j, c] += np.bincount(gi, np.where(ins, pc, 0.0), 6)
        best = acc.argmax(0)
        for c in (0, 1):
            for q in range(6): tab[t, c, q] = GRID[best[c, q]]
    return tab

def fit_width(D, btab, prx, nb=10):
    H = np.zeros((4, 2, nb)); errs = {}
    for pn in prx:
        Yt, Pt = tg(D, pn); c = apply(D, Pt, btab)
        ae = np.abs(Yt - c); scm = (Yt != 0) & D["valid"][:, None]
        for lg in range(4):
            for ch in (0, 1):
                a = ae[:, lg*5:(lg+1)*5, ..., ch]; m = scm[:, lg*5:(lg+1)*5, ..., ch]
                b = np.broadcast_to(D["p1b"][:, None, ..., ch], a.shape)
                for q in range(nb): errs.setdefault((lg, ch, q), []).append(a[m & (b == q)])
    for key, v in errs.items(): H[key] = opt_h(np.concatenate(v))
    return H

def width_of(D, H):
    h = np.empty(D["X"].shape, np.float32); ch = np.arange(2)[None, None, None, :]
    for t in range(20): h[:, t] = H[t // 5][ch, D["p1b"]]
    return h

def evaluate(D, ptab, btab, H, prx=("AT", "BT", "A", "B")):
    out = {}
    for pn in prx:
        Yt, Pt = tg(D, pn)
        pred = boost(apply(D, Pt, ptab), D["X"]); c = apply(D, Pt, btab)
        out[pn] = score(pred, Yt, c, width_of(D, H))
    return out

if __name__ == "__main__":
    tag = sys.argv[3] if len(sys.argv) > 3 else "L"
    S = prep("select", np.load(sys.argv[1])); F = prep("final", np.load(sys.argv[2]))
    import importlib.util
    spec = importlib.util.spec_from_file_location("tb17", os.path.join(HERE, "..", "..", "submissions", "v17", "_blend_tables.py"))
    TB = importlib.util.module_from_spec(spec); spec.loader.exec_module(TB)
    v17 = evaluate(F, np.array(TB.PRED_TAB), np.array(TB.BAND_TAB), np.array(TB.WIDTH_TAB))
    res = {"v17": v17}
    cfgs = [("refit_v17", False, True), ("L_con", True, True), ("L_uncon_pred", True, False)]
    tabs = {}
    for name, useL, con in cfgs:
        pt = fit_pred(S, ("A",), useL, con); bt = fit_band(S, ("A", "B"), useL); H = fit_width(S, bt, ("AT", "BT"))
        tabs[name] = (pt, bt, H)
        res[name] = evaluate(F, pt, bt, H)
        print(name, flush=True)
        for pn in ("AT", "BT", "A", "B"):
            print(f"   {pn:2} [{fmt(delta(res[name][pn], v17[pn]))}]  abs sps {res[name][pn]['sps']:.2f} T {res[name][pn]['T']:.4f} (v17 {v17[pn]['sps']:.2f} T {v17[pn]['T']:.4f})", flush=True)
    pickle.dump(dict(res=res, tabs=tabs), open(f"blend_{tag}.pkl", "wb"))
