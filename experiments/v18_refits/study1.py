"""Stage 1: more trivial estimators in the blend (pred and band centre), FINAL under AT (k=1.447) and A (k=1.665)."""
from lib18 import *
import time, importlib.util
t0 = time.time()
spec = importlib.util.spec_from_file_location("tb17", os.path.join(HERE, "..", "..", "submissions", "v17", "_blend_tables.py"))
TB = importlib.util.module_from_spec(spec); spec.loader.exec_module(TB)
PT17, BT17, WT17 = np.array(TB.PRED_TAB), np.array(TB.BAND_TAB), np.array(TB.WIDTH_TAB)
S = prep("select"); F = prep("final")
print("prep", round(time.time() - t0), flush=True)

# raw estimator quality on FINAL (true Y)
for nm, E in (("P", F["P"]), ("Em", np.broadcast_to(F["Em"][:, None], F["Y"].shape)), ("El", np.broadcast_to(F["El"][:, None], F["Y"].shape)),
              ("dmd", F["Em"][:, None] + F["dmd"])):
    E = np.ascontiguousarray(E)
    print(f"raw {nm:4} rel {sc.rel_l2_per_sample(E, F['Y'], 2).mean():.4f} tke {sc.tke_rel_l2_per_sample(E, F['Y'], 2).mean():.4f} mvpe {sc.mvpe_rel_l2_per_sample(E, F['Y']).mean():.4f}", flush=True)

EVAL = ("AT", "A")
rows = {}
def run(name, pred_fn, centre_fn, H, bst=True, boost_fn=None):
    for pn in EVAL:
        Yt, Pt = targets(F, pn)
        pred = pred_fn(F, Pt); c = centre_fn(F, Pt)
        h = F["h8"] if H is None else width_of(F, H)
        out = (boost_fn or boost)(pred, F["X"]) if bst else pred
        rows.setdefault(name, {})[pn] = score(out, Yt, c, h)
    b = rows["v17"] if "v17" in rows else rows[name]
    print(name, flush=True)
    for pn in EVAL:
        print(f"   {pn:2}[{fmt(delta(rows[name][pn], b[pn]))}]  abs sps {rows[name][pn]['sps']:.2f} T {rows[name][pn]['T']:.4f} rel {rows[name][pn]['rel']:.4f} tke {rows[name][pn]['tke']:.4f}", flush=True)

v17p = lambda D, Pt: apply_tab(D, Pt, ["P", "El"], PT17)
v17c = lambda D, Pt: apply_tab(D, Pt, ["P", "El"], BT17)
run("v17", v17p, v17c, WT17)
run("v8", lambda D, Pt: Pt, lambda D, Pt: Pt, None, bst=False)

SETS = {"PEl": ["P", "El"], "PEl+dmd": ["P", "El", "dmd"], "PEl+lin": ["P", "El", "Lend", "slope"],
        "PEl+dmd+lin": ["P", "El", "dmd", "Lend", "slope"], "all": ["P", "El", "El5", "dmd", "Lend", "slope"]}
TABS = {}
for sn, names in SETS.items():
    TABS[sn] = tab = fit_ls(S, names, ["A"])
    pf = (lambda names, tab: lambda D, Pt: apply_tab(D, Pt, names, tab))(names, tab)
    run(f"pred[{sn}] +v17 centre/width", pf, v17c, WT17)
    print("    lead0/lead19 u weights (sextile 0,5):", np.round(tab[0, 0, [0, 5]], 3).tolist(), np.round(tab[19, 0, [0, 5]], 3).tolist(), flush=True)
pickle.dump(dict(TABS=TABS, SETS=SETS), open(os.path.join(HERE, "tabs_study1.pkl"), "wb"))
print("done", round(time.time() - t0))
