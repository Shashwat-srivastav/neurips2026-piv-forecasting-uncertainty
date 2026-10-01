"""Stage 4: per-pixel (KE-map) boost variants vs the v16/v17 global boost. v17 pred/centre/width fixed."""
import os
from lib18 import *
import time, importlib.util, torch, torch.nn.functional as Fn
t0 = time.time()
spec = importlib.util.spec_from_file_location("tb17", os.path.join(HERE, "..", "..", "submissions", "v17", "_blend_tables.py"))
TB = importlib.util.module_from_spec(spec); spec.loader.exec_module(TB)
PT17, BT17, WT17 = np.array(TB.PRED_TAB), np.array(TB.BAND_TAB), np.array(TB.WIDTH_TAB)
PFT = os.environ.get("PFT")
S = prep("select", None if not PFT else np.load(os.path.join(HERE, "Pft_select.npy")))
F = prep("final", None if not PFT else np.load(os.path.join(HERE, "Pft_final.npy")))
for D in (S, F): D["h17"] = width_of(D, WT17)
v17p = lambda D, Pt: apply_tab(D, Pt, ["P", "El"], PT17)
v17c = lambda D, Pt: apply_tab(D, Pt, ["P", "El"], BT17)


def blur(a, k):
    if k <= 1: return a
    t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(a, -1, 1)))
    return np.moveaxis(Fn.avg_pool2d(t, k, 1, k // 2, count_include_pad=False).numpy(), 1, -1)


def boost_px(pred, X, g=1.0, cap=1.6, k=3, glob=True, gamma=0.5):
    """global v16 boost (optional) then per-pixel top-up of the u,v fluctuation towards g^2 * blurred input variance."""
    out = boost(pred, X) if glob else pred
    xc = X; fl = xc - xc.mean(1, keepdims=True)
    vin = blur((fl * fl).mean(1), k)                         # (n,H,W,2)
    ym = out.mean(1, keepdims=True); fp = out - ym
    vpr = blur((fp * fp).mean(1), k)
    a = np.clip((g * g * vin / np.maximum(vpr, 1e-12)) ** gamma, 1.0, cap).astype(np.float32)
    return (ym + a[:, None] * fp).astype(np.float32)


EVAL = ("AT", "A")
rows = {}
def run(name, D, bf, quiet=False):
    for pn in EVAL:
        Yt, Pt = targets(D, pn)
        rows.setdefault((D["role"], name), {})[pn] = score(bf(v17p(D, Pt), D["X"]), Yt, v17c(D, Pt), D["h17"])
    b = rows.get((D["role"], "v17"), rows[(D["role"], name)])
    ds = {pn: delta(rows[(D["role"], name)][pn], b[pn]) for pn in EVAL}
    print(f"[{D['role']}] {name:34} " + "  ".join(f"{pn}[{fmt(ds[pn])}]" for pn in EVAL), flush=True)
    return ds

for D in (S, F): run("v17", D, boost)
PEL = {pn: fit_ls(S, ["P", "El"], [pn]) for pn in ("A", "R", "AT")}
pickle.dump(PEL, open(os.path.join(HERE, f"pel_tabs_{'ft' if PFT else 'stock'}.pkl"), "wb"))
PF = {"v17": v17p}
for pn, tab in PEL.items():
    PF["PEl_" + pn] = (lambda tab: lambda D, Pt: apply_tab(D, Pt, ["P", "El"], tab))(tab)
def run2(name, D, pf, bf):
    for pn in EVAL:
        Yt, Pt = targets(D, pn)
        rows.setdefault((D["role"], name), {})[pn] = score(bf(pf(D, Pt), D["X"]), Yt, v17c(D, Pt), D["h17"])
    b = rows[(D["role"], "v17")]
    ds = {pn: delta(rows[(D["role"], name)][pn], b[pn]) for pn in EVAL}
    print(f"[{D['role']}] {name:34} " + "  ".join(f"{pn}[{fmt(ds[pn])}]" for pn in EVAL), flush=True)
    return ds
for nm in ("PEl_A", "PEl_R", "PEl_AT"):
    for D in (S, F): run2(nm + " +v16boost", D, PF[nm], boost)
V = {}
for g in (0.9, 1.0):
    for cap in (1.3, 1.6):
        for k in (1, 3):
            for glob in (True, False):
                V[f"g{g} cap{cap} k{k} glob{int(glob)}"] = dict(g=g, cap=cap, k=k, glob=glob)
sel = {}
for pfn in ("v17", "PEl_R"):
    for nm, kw in V.items():
        sel[(pfn, nm)] = run2(pfn + " " + nm, S, PF[pfn], (lambda kw: lambda p, X: boost_px(p, X, **kw))(kw))["AT"]["comp"]
top = sorted(sel, key=sel.get)[-4:]
print("top on SELECT:", top, flush=True)
for pfn, nm in top:
    run2(pfn + " " + nm, F, PF[pfn], (lambda kw: lambda p, X: boost_px(p, X, **kw))(V[nm]))
pickle.dump(dict(sel=sel, rows=rows), open(os.path.join(HERE, f"study4_{'ft' if PFT else 'stock'}.pkl"), "wb"))
print("done", round(time.time() - t0))
