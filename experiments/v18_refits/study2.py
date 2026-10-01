"""Stage 2: (c) ridge-rank band width refit, (b) boost variants, (a) unconstrained PEl pred. Fit SELECT, eval FINAL (AT primary, A robustness)."""
import os
from lib18 import *
import ridge18 as R
import time, importlib.util
t0 = time.time()
spec = importlib.util.spec_from_file_location("tb17", os.path.join(HERE, "..", "..", "submissions", "v17", "_blend_tables.py"))
TB = importlib.util.module_from_spec(spec); spec.loader.exec_module(TB)
PT17, BT17, WT17 = np.array(TB.PRED_TAB), np.array(TB.BAND_TAB), np.array(TB.WIDTH_TAB)
PFT = os.environ.get("PFT")          # task 2: override P with fine-tuned preds
S = prep("select", None if not PFT else np.load(os.path.join(HERE, "Pft_select.npy")))
F = prep("final", None if not PFT else np.load(os.path.join(HERE, "Pft_final.npy")))
TAG = "ft" if PFT else "stock"
T1 = pickle.load(open(os.path.join(HERE, "tabs_study1.pkl"), "rb"))
PEL = fit_ls(S, ["P", "El"], ["A"]) if PFT else T1["TABS"]["PEl"]
print("prep", round(time.time() - t0), flush=True)


def boost_v(pred, X, gamma=0.25, cap=1.6, perlead=False, capl=None):
    xc = X[..., :2]; fl = xc - xc.mean(1, keepdims=True)
    e_in = (fl * fl).mean(1).sum((1, 2, 3))
    ym = pred.mean(1, keepdims=True); fp = pred - ym
    if perlead:
        e_pr = (fp * fp).sum((2, 3, 4))                        # (n,20)
        a = np.clip((e_in[:, None] / np.maximum(e_pr, 1e-12)) ** gamma, 1.0, cap)[:, :, None, None, None]
    else:
        e_pr = (fp * fp).mean(1).sum((1, 2, 3))
        a = np.clip((e_in / np.maximum(e_pr, 1e-12)) ** gamma, 1.0, cap)[:, None, None, None, None]
        if capl is not None:   # lead-dependent cap: linear from capl[0] (lead0) to capl[1] (lead19)
            cl = np.linspace(capl[0], capl[1], 20)[None, :, None, None, None]
            a = np.minimum(np.broadcast_to(a, (len(a), 20, 1, 1, 1)), cl)
    return (ym + a * fp).astype(np.float32)


def bins_width(D, H, b):
    h = np.empty(D["X"].shape, np.float32); ch = np.arange(2)[None, None, None, :]
    for t in range(20): h[:, t] = H[t // 5][ch, b[:, t]]
    return h


def fit_width_b(D, centre, prx, b, nb):
    H = np.zeros((4, 2, nb)); errs = {}
    for pn in prx:
        Yt, Pt = targets(D, pn); c = centre(D, Pt)
        ae = np.abs(Yt - c); scm = (Yt != 0) & D["valid"][:, None]
        for lg in range(4):
            for ch in (0, 1):
                a = ae[:, lg*5:(lg+1)*5, ..., ch]; m = scm[:, lg*5:(lg+1)*5, ..., ch]; bb = b[:, lg*5:(lg+1)*5, ..., ch]
                for q in range(nb): errs.setdefault((lg, ch, q), []).append(a[m & (bb == q)])
    for k_, v in errs.items(): H[k_] = opt_h(np.concatenate(v))
    return H


EVAL = ("AT", "A")
rows = {}
def run(name, D, pred_fn, centre_fn, h, boost_fn, base="v17", quiet=False):
    for pn in EVAL:
        Yt, Pt = targets(D, pn)
        pred = pred_fn(D, Pt); c = centre_fn(D, Pt)
        out = boost_fn(pred, D["X"]) if boost_fn else pred
        rows.setdefault((D["role"], name), {})[pn] = score(out, Yt, c, h)
    b = rows.get((D["role"], base), rows[(D["role"], name)])
    ds = {pn: delta(rows[(D["role"], name)][pn], b[pn]) for pn in EVAL}
    if not quiet:
        print(f"[{D['role']}] {name}", flush=True)
        for pn in EVAL:
            s = rows[(D["role"], name)][pn]
            print(f"   {pn:2}[{fmt(ds[pn])}]  abs rel {s['rel']:.4f} tke {s['tke']:.4f} mvpe {s['mvpe']:.4f} sps {s['sps']:.2f} T {s['T']:.4f}", flush=True)
    return ds

v17p = lambda D, Pt: apply_tab(D, Pt, ["P", "El"], PT17)
v17c = lambda D, Pt: apply_tab(D, Pt, ["P", "El"], BT17)
pel = lambda D, Pt: apply_tab(D, Pt, ["P", "El"], PEL)
for D in (S, F):
    D["h17"] = width_of(D, WT17)
    run("v17", D, v17p, v17c, D["h17"], boost)

# ---------- (c) ridge-rank width, centre = v17 band centre ----------
Yt, Pt = targets(S, "AT")
Wr = R.fit(S["X"], S["P"], Yt, v17c(S, Pt), S["valid"])
print("ridge fit", round(time.time() - t0), flush=True)
for D in (S, F): D["rs"] = R.score(D["X"], D["P"], Wr)
WID = {}
for nb in (10, 20, 40):
    for D in (S, F): D[f"rb{nb}"] = R.rank_bins_lg(D["rs"], D["valid"], nb)
    WID[nb] = fit_width_b(S, v17c, ("AT", "BT"), S[f"rb{nb}"], nb)
    for D in (S, F):
        run(f"v17 + ridge-rank width nb{nb}", D, v17p, v17c, bins_width(D, WID[nb], D[f"rb{nb}"]), boost)
# p1 refit control (same machinery)
for D in (S, F): D["p1b20"] = np.broadcast_to(rank_bins(np.sqrt(((D["X"][:, 1:] - D["X"][:, :-1]) ** 2).mean(1)).astype(np.float32), D["valid"], 20)[:, None], D["X"].shape)
WID["p1_20"] = fit_width_b(S, v17c, ("AT", "BT"), S["p1b20"], 20)
for D in (S, F): run("v17 + p1-rank width nb20 (control)", D, v17p, v17c, bins_width(D, WID["p1_20"], D["p1b20"]), boost)

# ---------- (b) boost grid on SELECT (v17 pred, v17 band) then FINAL for the top ----------
BG = {"g.25c1.6": dict(), "g.25c2.0": dict(cap=2.0), "g.35c1.6": dict(gamma=.35), "g.35c2.0": dict(gamma=.35, cap=2.0),
      "g.5c2.0": dict(gamma=.5, cap=2.0), "g.5c2.5": dict(gamma=.5, cap=2.5), "perlead g.25c1.6": dict(perlead=True),
      "perlead g.35c2.0": dict(perlead=True, gamma=.35, cap=2.0), "capl1.3-2.0": dict(cap=3.0, capl=(1.3, 2.0)),
      "g.35 capl1.3-2.2": dict(gamma=.35, cap=3.0, capl=(1.3, 2.2)), "capl1.6-2.2": dict(cap=3.0, capl=(1.6, 2.2))}
sel = {}
for pn_, pf in (("v17pred", v17p), ("PElpred", pel)):
    for bn, kw in BG.items():
        d = run(f"{pn_} boost {bn}", S, pf, v17c, S["h17"], (lambda kw: lambda p, X: boost_v(p, X, **kw))(kw), quiet=True)
        sel[(pn_, bn)] = d["AT"]["comp"]
        print(f"  SELECT {pn_} boost {bn:18} AT[{fmt(d['AT'])}]", flush=True)
top = sorted(sel, key=sel.get)[-4:]
for pn_, bn in top + [("PElpred", "g.25c1.6")]:
    pf = v17p if pn_ == "v17pred" else pel
    run(f"{pn_} boost {bn}", F, pf, v17c, F["h17"], (lambda kw: lambda p, X: boost_v(p, X, **kw))(BG[bn]))
pickle.dump(dict(Wr=Wr, WID=WID, rows=rows, sel=sel), open(os.path.join(HERE, f"study2_{TAG}.pkl"), "wb"))
print("done", round(time.time() - t0))
