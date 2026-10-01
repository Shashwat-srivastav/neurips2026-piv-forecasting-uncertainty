"""Band-centre fit: per (lead,ch,bin) choose (alpha,beta) on a grid maximising T; centre = Em + a(P-Em) + b(El-Em)."""
from fit import *
GRID = [(a / 10, b / 10) for a in range(11) for b in range(11 - a)]

def fit_band(D, h8, feat, nb, proxies_used, g):
    bins = D["bins"][(feat, nb)]
    h = (g * h8[:, 0]).astype(np.float32)          # (n,H,W,2)
    pen = np.exp(-2 * h / SIG)
    dlm = D["E"]["last"] - D["E"]["mean"]
    tab = np.zeros((20, 2, nb, 2)); Ttot = 0.0; N = 0
    for t in range(20):
        acc = np.zeros((len(GRID), 2, nb))
        for pn in proxies_used:
            Yt, Pt = D["pr"][pn]
            y = Yt[:, t]; m = (y != 0)
            dP = Pt[:, t] - D["E"]["mean"]; r = y - D["E"]["mean"]
            for c in (0, 1):
                mc = m[..., c] & D["valid"][..., c]
                gi = bins[..., c][mc]; dPc = dP[..., c][mc]; dl = dlm[..., c][mc]; rc = r[..., c][mc]; hc = h[..., c][mc]; pc = pen[..., c][mc]
                for j, (a, b) in enumerate(GRID):
                    ins = np.abs(rc - a * dPc - b * dl) <= hc
                    acc[j, c] += np.bincount(gi, np.where(ins, pc, 0.0), nb)
        best = acc.argmax(0)
        for c in (0, 1):
            for k in range(nb):
                tab[t, c, k] = GRID[best[c, k]]
    return tab
