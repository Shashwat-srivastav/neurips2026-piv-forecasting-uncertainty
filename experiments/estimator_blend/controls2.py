"""Controls + p1-rank width stack. FINAL evaluated under 4 proxies: A/B x (k_rel, k_T).
k_T from experiments/interval_recipes (v8 live T matched): SELECT 1.394, FINAL 1.447."""
from fit_band import *
import pickle
KT = {"select": 1.394, "final": 1.447}
def add_T(D, role):
    Y, P = D["Y"], D["P"]; k = KT[role]
    D["pr"]["AT"] = (np.where(Y == 0.0, 0.0, P + k * (Y - P)).astype(np.float32), P)
    D["pr"]["BT"] = (Y, (Y + k * (P - Y)).astype(np.float32))
S = prep("select"); F = prep("final"); add_T(S, "select"); add_T(F, "final")
S["h8"] = v8_half(S["X"]); F["h8"] = v8_half(F["X"])
tabs = pickle.load(open("tabs_pred.pkl", "rb")); bt = pickle.load(open("tabs_band_g1.pkl", "rb"))

def p1_bins(D, nb=10):
    X = D["X"]; p1 = np.sqrt(((X[:, 1:] - X[:, :-1]) ** 2).mean(1)).astype(np.float32)
    return rank_bins(p1, D["valid"], nb)
S["p1b"] = p1_bins(S); F["p1b"] = p1_bins(F)
def opt_h(ae):
    e = np.sort(ae); m = len(e)
    return float(e[np.argmax(np.exp(-2 * e / SIG) * np.arange(1, m + 1) / m)]) if m else 0.01
def centre_of(D, Pt, pk, bk):
    pred = Pt if pk is None else apply_ls(D, Pt, "both", "sl", 6, tabs[("both", "sl", 6, pk)])
    c = Pt if bk is None else apply_ls(D, Pt, "both", "sl", 6, bt[bk])
    return pred, c
def fit_width(pk, bk, prx, nb=10):
    """h table (4 lead groups, 2 ch, nb p1-rank bins) = T-optimal half-width of |Y - centre| on SELECT."""
    H = np.zeros((4, 2, nb)); errs = {}
    for pn in prx:
        Yt, Pt = S["pr"][pn]; _, c = centre_of(S, Pt, pk, bk)
        ae = np.abs(Yt - c); sc_ = (Yt != 0) & S["valid"][:, None]
        for lg in range(4):
            for ch in (0, 1):
                a = ae[:, lg*5:(lg+1)*5, ..., ch]; m = sc_[:, lg*5:(lg+1)*5, ..., ch]
                b = np.broadcast_to(S["p1b"][:, None, ..., ch], a.shape)
                for k in range(nb):
                    errs.setdefault((lg, ch, k), []).append(a[m & (b == k)])
    for (lg, ch, k), v in errs.items(): H[lg, ch, k] = opt_h(np.concatenate(v))
    return H
def width_of(D, H):
    h = np.empty(D["X"].shape, np.float32); ch = np.arange(2)[None, None, None, :]
    for t in range(20): h[:, t] = H[t // 5][ch, D["p1b"]]
    return h
rows = {}
def run(name, pk, bk, g=1.0, bst=True, H=None):
    for pn in ("A", "B", "AT", "BT"):
        Yt, Pt = F["pr"][pn]; pred, c = centre_of(F, Pt, pk, bk)
        h = g * F["h8"] if H is None else width_of(F, H)
        out = boost(pred, F["X"]) if bst else pred
        rows.setdefault(name, {})[pn] = score(out, Yt, c, h)
    b = rows["v8"]; s = rows[name]
    fmt = lambda d: f"rel {d['rel']:+.3f} tke {d['tke']:+.3f} mvpe {d['mvpe']:+.3f} sps {d['sps']:+.3f} T {d['T']:+.4f} comp {d['comp']:+.3f}"
    print(name, flush=True)
    for pn in ("A", "B", "AT", "BT"):
        print(f"   {pn:2}[{fmt(delta(s[pn], b[pn]))}]  abs sps {s[pn]['sps']:.2f} T {s[pn]['T']:.4f}", flush=True)
run("v8", None, None, bst=False)
run("v16", None, None)
run("v8 g=1.2 (width only)", None, None, g=1.2, bst=False)
for fb in ("A", "B", "AB"):
    run(f"band{fb}@g1 (centre only)", None, fb, bst=False)
run("predA+boost", "A", None)
run("predA+bandAB@g1+boost", "A", "AB")
run("predAB+bandAB@g1+boost", "AB", "AB")
run("predA+bandA@g1+boost", "A", "A")
HW = {}
for fitk in (("A", "B"), ("AT", "BT")):
    HW[fitk] = fit_width("A", "AB", fitk); print("p1 width table fit", fitk, "lead0 u", np.round(HW[fitk][0, 0], 4), "lead3 u", np.round(HW[fitk][3, 0], 4), flush=True)
    run(f"predA+bandAB@g1+boost+p1width{fitk}", "A", "AB", H=HW[fitk])
Hv8 = fit_width(None, None, ("AT", "BT"))
run("v8 centre + p1width(AT,BT) (width-only control)", None, None, bst=False, H=Hv8)
pickle.dump(dict(HW=HW, Hv8=Hv8), open("tabs_width.pkl", "wb"))
pickle.dump(rows, open("rows_controls2.pkl", "wb"))
