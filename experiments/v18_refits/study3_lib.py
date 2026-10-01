"""fit_centre extracted from study3.py."""
from lib18 import *


def fit_centre(D, names, grids, prx, h):
    """per (lead, ch, sextile) choose coefficient vector from grid maximising T (sum over prx)."""
    import itertools
    combos = [c for c in itertools.product(*grids) if sum(c[:2]) <= 1.0 + 1e-9]
    CM = np.array(combos, np.float32)                       # (G,k)
    tab = np.zeros((20, 2, 6, len(names)))
    for t in range(20):
        acc = np.zeros((len(CM), 2, 6))
        for pn in prx:
            Yt, Pt = targets(D, pn)
            y = Yt[:, t]; R = regressors(D, Pt, t, names); r = y - D["Em"]
            ht = h[:, t]; pen = np.exp(-2 * ht / SIG)
            for c in (0, 1):
                m = (y[..., c] != 0) & D["valid"][..., c]
                g = D["bins"][..., c][m]; Rc = np.stack([q[..., c][m] for q in R], 1); rc = r[..., c][m]; hc = ht[..., c][m]; pc = pen[..., c][m]
                for j in range(len(CM)):
                    ins = np.abs(rc - Rc @ CM[j]) <= hc
                    acc[j, c] += np.bincount(g, np.where(ins, pc, 0.0), 6)
        best = acc.argmax(0)
        for c in (0, 1):
            for q in range(6): tab[t, c, q] = CM[best[c, q]]
    return tab


