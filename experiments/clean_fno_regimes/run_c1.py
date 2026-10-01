"""C1: scale-free vs absolute interval widths, honest cross-fit (fit SELECT -> eval FINAL, fit FINAL -> eval SELECT).
usage: python experiments/clean_fno_regimes/run_c1.py <P_select.npy|official> <P_final.npy|official> <tag> [kY]
kY: proxy factor for models that saw SELECT/FINAL (Y' = P + k(Y-P)); 1 = honest."""
import os, sys, gc, pickle, importlib.util as iu
sys.dont_write_bytecode = True
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sflib as L

ps, pf, tag = sys.argv[1], sys.argv[2], sys.argv[3]
kY = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
OUT = os.path.dirname(os.path.abspath(__file__))


def get(role, p):
    F = L.load_split(role, p)
    if kY != 1.0:
        P = np.asarray(F["P"]); Y = F["Y"]
        F["Y"] = np.where(Y == 0.0, 0.0, P + kY * (Y - P)).astype(np.float32); del P, Y; gc.collect()
    return F


def fit(F):
    bt = L.fit_band(F)
    Ha, Hs = L.width_hists(F, bt)
    T = dict(bt=bt, abs=L.solve_abs(Ha)); T.update({k: L.solve_sf(Hs[k], F[k]) for k in L.SCALES})
    del Ha, Hs; gc.collect()
    return T


def report(F, T, label, extra=None):
    acc = L.acc_per_sample(F); subs = L.subsets(F)
    rows = dict(extra or {}); rows.update({m: (T["bt"], T[m], m) for m in ("abs",) + L.SCALES})
    res = {}
    for name, (bt, H, mode) in rows.items():
        band = L.band_per_sample(F, bt, H, mode)
        for sname, idx in subs.items():
            r = L.summarise(acc, band, idx); res[(name, sname)] = r
            print(f"  {label:26s} {name:12s} {sname:6s} ({len(set(F['traj'][idx]))} traj) {L.fmt(r)}", flush=True)
    return res


FS = get("select", ps); print("SELECT scales med s_p1", np.median(FS["s_p1"], 0), "s_sl", np.median(FS["s_sl"], 0), flush=True)
TS = fit(FS); print("fitted on SELECT", flush=True)
FF = get("final", pf); print("FINAL scales med s_p1", np.median(FF["s_p1"], 0), "s_sl", np.median(FF["s_sl"], 0), flush=True)
TF = fit(FF); print("fitted on FINAL", flush=True)
pickle.dump(dict(TS=TS, TF=TF), open(os.path.join(OUT, f"tables_{tag}.pkl"), "wb"))

extra = {}
if tag.startswith("run1"):   # exactness check: shipped exp1 tables (abs, SELECT-fit) on FINAL must give 37.70
    s = iu.spec_from_file_location("t", os.path.join(L.LAB, "build_exp1", "_blend_tables.py")); t = iu.module_from_spec(s); s.loader.exec_module(t)
    extra["shipped_abs"] = (np.array(t.BAND_TAB), np.array(t.WIDTH_TAB), "abs")
print("== FINAL, tables fitted on SELECT (honest cross-fit) ==", flush=True)
rF = report(FF, TS, "fitS->evalF", extra)
print("== FINAL, tables fitted on FINAL (in-sample, optimistic) ==", flush=True)
rFF = report(FF, TF, "fitF->evalF")
del FF; gc.collect()
print("== SELECT, tables fitted on FINAL (honest cross-fit, live-like) ==", flush=True)
rS = report(FS, TF, "fitF->evalS")
print("== SELECT, tables fitted on SELECT (in-sample, optimistic) ==", flush=True)
rSS = report(FS, TS, "fitS->evalS")
pickle.dump(dict(rF=rF, rFF=rFF, rS=rS, rSS=rSS), open(os.path.join(OUT, f"results_{tag}.pkl"), "wb"))
