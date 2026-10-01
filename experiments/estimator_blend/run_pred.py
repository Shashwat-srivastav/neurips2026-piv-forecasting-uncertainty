"""Stage 1: prediction-only blend. rel on FINAL under both proxies, for many configs (cheap)."""
from fit import *
import itertools, pickle, time
t0 = time.time()
S = prep("select"); F = prep("final"); print("prep", time.time() - t0, "k S", S["ks"], "k F", F["ks"])
res = {}
for pn in "AB":
    Yt, Pt = F["pr"][pn]
    res[("v8", pn)] = float(sc.rel_l2_per_sample(Pt, Yt, 2).mean())
print("v8 rel A %.5f B %.5f" % (res[("v8", "A")], res[("v8", "B")]))
tabs = {}
for est, feat, nb, fitp in itertools.product(("mean", "last", "both"), ("d2", "sl"), (1, 3, 6), ("A", "B", "AB")):
    if nb == 1 and feat == "sl": continue
    tab = fit_ls(S, est, feat, nb, list(fitp)); tabs[(est, feat, nb, fitp)] = tab
    line = []
    for pn in "AB":
        Yt, Pt = F["pr"][pn]
        r = float(sc.rel_l2_per_sample(apply_ls(F, Pt, est, feat, nb, tab), Yt, 2).mean())
        line.append(s_err(r) - s_err(res[("v8", pn)]))
    print(f"{est:5} {feat} nb{nb} fit{fitp:2}  d_relscore A {line[0]:+.3f}  B {line[1]:+.3f}   alpha(t0,u,bins)={np.round(tab[0,0,:,0],2)} beta={np.round(tab[0,0,:,1],2)} alpha(t19,u)={np.round(tab[19,0,:,0],2)}", flush=True)
pickle.dump(tabs, open("tabs_pred.pkl", "wb"))
