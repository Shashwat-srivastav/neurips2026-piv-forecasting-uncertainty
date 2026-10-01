"""Stage 3: band CENTRE refit by T-grid, using v17 widths during the fit, with/without a DMD term; width refit after."""
import os
from lib18 import *
import time, importlib.util
t0 = time.time()
spec = importlib.util.spec_from_file_location("tb17", os.path.join(HERE, "..", "..", "submissions", "v17", "_blend_tables.py"))
TB = importlib.util.module_from_spec(spec); spec.loader.exec_module(TB)
PT17, BT17, WT17 = np.array(TB.PRED_TAB), np.array(TB.BAND_TAB), np.array(TB.WIDTH_TAB)
PFT = os.environ.get("PFT")
S = prep("select", None if not PFT else np.load(os.path.join(HERE, "Pft_select.npy")))
F = prep("final", None if not PFT else np.load(os.path.join(HERE, "Pft_final.npy")))
TAG = "ft" if PFT else "stock"
for D in (S, F): D["h17"] = width_of(D, WT17)
print("prep", round(time.time() - t0), flush=True)


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


EVAL = ("AT", "A")
rows = {}
def run(name, D, pf, cf, h, base="v17"):
    for pn in EVAL:
        Yt, Pt = targets(D, pn)
        rows.setdefault((D["role"], name), {})[pn] = score(boost(pf(D, Pt), D["X"]), Yt, cf(D, Pt), h)
    b = rows.get((D["role"], base), rows[(D["role"], name)])
    print(f"[{D['role']}] {name}", flush=True)
    for pn in EVAL:
        s = rows[(D["role"], name)][pn]
        print(f"   {pn:2}[{fmt(delta(s, b[pn]))}]  abs sps {s['sps']:.2f} T {s['T']:.4f}", flush=True)

v17p = lambda D, Pt: apply_tab(D, Pt, ["P", "El"], PT17)
v17c = lambda D, Pt: apply_tab(D, Pt, ["P", "El"], BT17)
for D in (S, F): run("v17", D, v17p, v17c, D["h17"])
G11 = [i / 10 for i in range(11)]
CT = {}
for nm, names, grids, prx in (("PEl@h17 AT,BT", ["P", "El"], [G11, G11], ("AT", "BT")),
                              ("PEl@h17 A,B", ["P", "El"], [G11, G11], ("A", "B")),
                              ("PEldmd@h17 AT,BT", ["P", "El", "dmd"], [G11, G11, [-0.2, 0.0, 0.2, 0.4, 0.6]], ("AT", "BT"))):
    CT[nm] = tab = fit_centre(S, names, grids, prx, S["h17"])
    cf = (lambda names, tab: lambda D, Pt: apply_tab(D, Pt, names, tab))(names, tab)
    Hn = fit_width(S, (lambda cf: lambda Pt: cf(S, Pt))(cf), ("AT", "BT"))
    CT[nm + " H"] = Hn
    print("fit", nm, round(time.time() - t0), flush=True)
    run(f"centre {nm} (v17 width)", F, v17p, cf, F["h17"])
    run(f"centre {nm} + width refit", F, v17p, cf, width_of(F, Hn))
pickle.dump(CT, open(os.path.join(HERE, f"study3_{TAG}.pkl"), "wb"))
print("done", round(time.time() - t0))
