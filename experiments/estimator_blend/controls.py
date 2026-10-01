"""Controls separating band-centre blend from pure band widening (g). FINAL, both proxies."""
from fit_band import *
import pickle
S = prep("select"); F = prep("final")
S["h8"] = v8_half(S["X"]); F["h8"] = v8_half(F["X"])
tabs = pickle.load(open("tabs_pred.pkl", "rb"))
import os
if os.path.exists("tabs_band_g1.pkl"):
    bt = pickle.load(open("tabs_band_g1.pkl", "rb"))
else:
    bt = {}
    for fitp in ("A", "B", "AB"):
        bt[fitp] = fit_band(S, S["h8"], "sl", 6, list(fitp), 1.0)
    pickle.dump(bt, open("tabs_band_g1.pkl", "wb"))
del S
rows = {}
def run(name, pk, bk, g, bst):
    for pn in "AB":
        Yt, Pt = F["pr"][pn]
        pred = Pt if pk is None else apply_ls(F, Pt, "both", "sl", 6, tabs[("both", "sl", 6, pk)])
        centre = Pt if bk is None else (pred if bk == "pred" else apply_ls(F, Pt, "both", "sl", 6, bt[bk]))
        out = boost(pred, F["X"]) if bst else pred
        rows.setdefault(name, {})[pn] = score(out, Yt, centre, g * F["h8"])
    b = rows["v8"]; s = rows[name]
    fmt = lambda d: " ".join(f"{k} {d[k]:+.3f}" for k in ("rel", "tke", "mvpe", "sps", "T", "comp"))
    print(f"{name:40} A[{fmt(delta(s['A'], b['A']))}]  B[{fmt(delta(s['B'], b['B']))}]", flush=True)
run("v8", None, None, 1.0, False)
for g in (1.1, 1.2): run(f"v8 g={g}", None, None, g, False)
for g in (1.1, 1.2): run(f"v16 g={g}", None, None, g, True)
run("v16", None, None, 1.0, True)
for fb in ("A", "B", "AB"):
    run(f"band{fb} g1", None, fb, 1.0, False)
    run(f"band{fb} g1 +boost", None, fb, 1.0, True)
    for pk in ("A", "AB"):
        run(f"pred{pk}+band{fb} g1 +boost", pk, fb, 1.0, True)
pickle.dump(rows, open("rows_controls.pkl", "wb"))
