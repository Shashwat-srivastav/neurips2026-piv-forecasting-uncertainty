"""Per-window compliance study. usage: eval_pw.py v17 | v19 L_old_sel L_old_fin L_new_sel L_new_fin
(a) v17 shipped (5-window bins); (b) v17 tables + per-window bins; (c) v17 recipe refit per-window;
(d) v19 recipe refit per-window (L from ridge with per-window classes). FINAL, proxies AT/BT/A/B.
Everything reuses blend19.py (pw copy); its "refit_v17" config reproduced the shipped v17 tables exactly."""
import sys, os, pickle, importlib.util
sys.dont_write_bytecode = True
from blend19 import *

def load_tb(path):
    spec = importlib.util.spec_from_file_location("tb" + str(abs(hash(path))), path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return np.array(m.PRED_TAB), np.array(m.BAND_TAB), np.array(m.WIDTH_TAB)

def refit(S, useL):
    pt = fit_pred(S, ("A",), useL, True); bt = fit_band(S, ("A", "B"), useL); H = fit_width(S, bt, ("AT", "BT"))
    return pt, bt, H

def show(name, r, ref):
    print(name, flush=True)
    for pn in ("AT", "BT", "A", "B"):
        print(f"   {pn:2} [{fmt(delta(r[pn], ref[pn]))}]  abs sps {r[pn]['sps']:.2f} T {r[pn]['T']:.4f} rel {r[pn]['rel']:.5f}", flush=True)

B17 = os.path.join(HERE, "..", "..", "submissions", "v17", "_blend_tables.py")
B19 = os.path.join(HERE, "..", "v19_ridge_partner", "build_v19", "_blend_tables.py")
out = {}
if sys.argv[1] == "v17":
    F5 = prep("final", None, 5); T17 = load_tb(B17)
    a = evaluate(F5, *T17); del F5
    F1 = prep("final", None, 1)
    b = evaluate(F1, *T17)
    S1 = prep("select", None, 1)
    tabs = refit(S1, False); c = evaluate(F1, *tabs)
    out = dict(a=a, b=b, c=c, tabs_c=tabs)
    for k in "abc": show(k, out[k], a)
    pickle.dump(out, open("eval_v17pw.pkl", "wb"))
else:
    a = pickle.load(open("eval_v17pw.pkl", "rb"))["a"]
    T19 = load_tb(B19)
    F5 = prep("final", np.load(os.path.join(HERE, "..", "v19_ridge_partner", "L_sxb3_final.npy")), 5)
    e = evaluate(F5, *T19); del F5
    res = dict(v19_shipped=e)
    for tag, ls, lf in (("old_ridge", sys.argv[2], sys.argv[3]), ("pw_ridge", sys.argv[4], sys.argv[5])):
        if not os.path.exists(lf): continue
        S1 = prep("select", np.load(ls), 1); F1 = prep("final", np.load(lf), 1)
        res[f"v19_pw_notrefit_{tag}"] = evaluate(F1, *T19)
        tabs = refit(S1, True); res[f"d_{tag}"] = evaluate(F1, *tabs); res[f"tabs_{tag}"] = tabs
        del S1, F1
    for k, v in res.items():
        if not k.startswith("tabs"): show(k, v, a)
    pickle.dump(res, open("eval_v19pw.pkl", "wb"))
