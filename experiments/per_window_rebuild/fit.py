"""Fit per-(lead,channel,bin) blend weights on SELECT. pred' = Em + a(P-Em) + b(El-Em)."""
from base import *

def prep(role, nb=6):
    X, Y, P, traj = load(role)
    E, valid = estimators(X)
    F = dict(sl=sigma_local(X).astype(np.float32), d2=d2rms(X))
    bins = {(f, n): rank_bins(F[f], valid, n) for f in F for n in (1, 3, 6)}
    pr, ks = proxies(role, Y, P)
    return dict(X=X, Y=Y, P=P, E=E, valid=valid, bins=bins, pr=pr, ks=ks)

def fit_ls(D, est, feat, nb, proxies_used, wmax=1.0):
    """est: 'mean' | 'last' | 'both'. Returns tab (20,2,nb,2): (alpha on P, beta on last-mean) wrt base."""
    base = D["E"]["last"] if est == "last" else D["E"]["mean"]
    bins = D["bins"][(feat, nb)]
    tab = np.zeros((20, 2, nb, 2))
    for t in range(20):
        S = np.zeros((2, nb, 3, 3))
        for pn in proxies_used:
            Yt, Pt = D["pr"][pn]
            y = Yt[:, t]; m = (y != 0) & D["valid"]
            a = Pt[:, t] - base; b = D["E"]["last"] - D["E"]["mean"]; r = y - base
            for c in (0, 1):
                mc = m[..., c]; g = bins[..., c][mc]
                A_, B_, R_ = a[..., c][mc].astype(np.float64), b[..., c][mc].astype(np.float64), r[..., c][mc].astype(np.float64)
                for i, (u, v) in enumerate([(A_, A_), (A_, B_), (B_, B_)]):
                    pass
                S[c, :, 0, 0] += np.bincount(g, A_ * A_, nb); S[c, :, 0, 1] += np.bincount(g, A_ * B_, nb)
                S[c, :, 1, 1] += np.bincount(g, B_ * B_, nb); S[c, :, 0, 2] += np.bincount(g, A_ * R_, nb)
                S[c, :, 1, 2] += np.bincount(g, B_ * R_, nb)
        for c in (0, 1):
            for k in range(nb):
                aa, ab, bb, ar, br = S[c, k, 0, 0], S[c, k, 0, 1], S[c, k, 1, 1], S[c, k, 0, 2], S[c, k, 1, 2]
                if est != "both":
                    tab[t, c, k] = (np.clip(ar / max(aa, 1e-12), 0, wmax), 0.0)
                    continue
                det = aa * bb - ab * ab
                al = (ar * bb - br * ab) / det; be = (aa * br - ab * ar) / det
                if al < 0 or be < 0 or al + be > 1:
                    # best feasible among the edges: beta=0, alpha=0, alpha+beta=1
                    cands = []
                    for al2, be2 in [(np.clip(ar / aa, 0, 1), 0.0), (0.0, np.clip(br / bb, 0, 1))]:
                        cands.append((al2, be2))
                    # alpha+beta=1 edge: pred = last + alpha(P-last)
                    d_ = aa - 2 * ab + bb; n_ = ar - br - ab + bb
                    al3 = np.clip(n_ / max(d_, 1e-12), 0, 1); cands.append((al3, 1 - al3))
                    sse = lambda p: p[0]**2*aa + 2*p[0]*p[1]*ab + p[1]**2*bb - 2*(p[0]*ar + p[1]*br)
                    al, be = min(cands, key=sse)
                tab[t, c, k] = (al, be)
    return tab

def apply_ls(D, Pt, est, feat, nb, tab):
    base = D["E"]["last"] if est == "last" else D["E"]["mean"]
    bins = D["bins"][(feat, nb)]
    ch = np.arange(2)[None, None, None, :]
    out = np.empty_like(Pt)
    dlm = D["E"]["last"] - D["E"]["mean"]
    for t in range(20):
        al = tab[t][ch, bins][..., 0].squeeze(0) if False else tab[t, ch, bins, 0]
        be = tab[t, ch, bins, 1]
        o = base + al * (Pt[:, t] - base) + be * dlm
        out[:, t] = np.where(D["valid"], o, Pt[:, t])
    return out.astype(np.float32)
