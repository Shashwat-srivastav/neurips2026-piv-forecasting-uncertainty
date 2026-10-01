"""Task 2: fine-tuned FNO. Raw FINAL subscores (true Y) stock vs ft; then v17-style post-processing refit on ft residuals.
Deltas are vs stock v17 (stock P + v17 tables) under proxy AT/A -- the proxy k was calibrated on the STOCK model's live
behaviour and says nothing about how the fine-tuned model transfers live."""
from lib18 import *
import time, importlib.util
t0 = time.time()
spec = importlib.util.spec_from_file_location("tb17", os.path.join(HERE, "..", "..", "submissions", "v17", "_blend_tables.py"))
TB = importlib.util.module_from_spec(spec); spec.loader.exec_module(TB)
PT17, BT17, WT17 = np.array(TB.PRED_TAB), np.array(TB.BAND_TAB), np.array(TB.WIDTH_TAB)
from study3_lib import fit_centre
F = prep("final"); PstF = F["P"].copy()
Pft = np.load(os.path.join(HERE, "Pft_final.npy"))
for nm, P in (("stock", PstF), ("ft", Pft)):
    print(f"RAW FINAL {nm:5} rel {sc.rel_l2_per_sample(P, F['Y'], 2).mean():.4f} ({s_err(sc.rel_l2_per_sample(P, F['Y'], 2).mean()):.3f})"
          f"  tke {sc.tke_rel_l2_per_sample(P, F['Y'], 2).mean():.4f} ({s_err(sc.tke_rel_l2_per_sample(P, F['Y'], 2).mean()):.3f})"
          f"  mvpe {sc.mvpe_rel_l2_per_sample(P, F['Y']).mean():.4f} ({s_err(sc.mvpe_rel_l2_per_sample(P, F['Y']).mean()):.3f})", flush=True)
F["h17"] = width_of(F, WT17)
EVAL = ("AT", "A", "R")
v17p = lambda D, Pt: apply_tab(D, Pt, ["P", "El"], PT17)
v17c = lambda D, Pt: apply_tab(D, Pt, ["P", "El"], BT17)
rows = {}
def run(name, pf, cf, h):
    for pn in EVAL:
        Yt, Pt = targets(F, pn)
        rows.setdefault(name, {})[pn] = score(boost(pf(F, Pt), F["X"]), Yt, cf(F, Pt), h)
    b = rows["stock v17"]
    print(name, flush=True)
    for pn in EVAL:
        s = rows[name][pn]
        print(f"   {pn:2}[{fmt(delta(s, b[pn]))}]  abs rel {s_err(s['rel']):.3f} tke {s_err(s['tke']):.3f} mvpe {s_err(s['mvpe']):.3f} sps {s['sps']:.2f} T {s['T']:.4f}", flush=True)
run("stock v17", v17p, v17c, F["h17"])
PEL = pickle.load(open(os.path.join(HERE, "pel_tabs_stock.pkl"), "rb"))
run("stock v18 (PEl_A)", lambda D, Pt: apply_tab(D, Pt, ["P", "El"], PEL["A"]), v17c, F["h17"])
del F
# ---- fine-tuned: fit on SELECT with ft P ----
S = prep("select", np.load(os.path.join(HERE, "Pft_select.npy"))); F = prep("final", Pft); F["h17"] = width_of(F, WT17); S["h17"] = width_of(S, WT17)
run("ft + v17 tables", v17p, v17c, F["h17"])
run("ft + v18 stock tables", lambda D, Pt: apply_tab(D, Pt, ["P", "El"], PEL["A"]), v17c, F["h17"])
PA = fit_ls(S, ["P", "El"], ["A"])
pa = lambda D, Pt: apply_tab(D, Pt, ["P", "El"], PA)
run("ft + PEl_A refit", pa, v17c, F["h17"])
G11 = [i / 10 for i in range(11)]
CB = fit_centre(S, ["P", "El"], [G11, G11], ("AT", "BT"), S["h17"])
cb = lambda D, Pt: apply_tab(D, Pt, ["P", "El"], CB)
HW = fit_width(S, lambda Pt: cb(S, Pt), ("AT", "BT"))
run("ft + PEl_A refit + centre refit", pa, cb, F["h17"])
run("ft + PEl_A refit + centre refit + width refit", pa, cb, width_of(F, HW))
pickle.dump(dict(PRED=PA, BAND=CB, WIDTH=HW), open(os.path.join(HERE, "ft_tabs.pkl"), "wb"))
print("done", round(time.time() - t0))
