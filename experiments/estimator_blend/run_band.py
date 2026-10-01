"""Stage 2: full official-scorer evaluation on FINAL under proxies A and B."""
from fit_band import *
import pickle, time, json
t0 = time.time()
S = prep("select"); F = prep("final")
S["h8"] = v8_half(S["X"]); F["h8"] = v8_half(F["X"])
tabs = pickle.load(open("tabs_pred.pkl", "rb"))
print("prep", round(time.time() - t0))

# --- global band factor + band centre fit on SELECT (per fit-proxy) ---
def sel_T(D, tab, feat, nb, g, pn):
    Yt, Pt = D["pr"][pn]
    c = apply_ls(D, Pt, "both", feat, nb, tab)
    h = g * D["h8"]; inside = np.abs(Yt - c) <= h; sc_ = Yt != 0
    return float(np.sum(np.where(inside, np.exp(-2 * h / SIG), 0), where=sc_) / sc_.sum())

band = {}
for feat, nb in (("sl", 6), ("d2", 6)):
    for fitp in ("A", "B", "AB"):
        best = None
        for g in (0.8, 0.9, 1.0, 1.1, 1.2):
            tab = fit_band(S, S["h8"], feat, nb, list(fitp), g)
            Ts = [sel_T(S, tab, feat, nb, g, pn) for pn in fitp]
            m = float(np.mean(Ts))
            print(f"band fit {feat}{nb} {fitp:2} g={g}: SELECT T {m:.4f}", flush=True)
            if best is None or m > best[0]: best = (m, g, tab)
        band[(feat, nb, fitp)] = best
        print("  best g", best[1], "tab t0 u", best[2][0, 0], "t19 u", best[2][19, 0], flush=True)
pickle.dump(band, open("tabs_band.pkl", "wb"))

# --- candidates on FINAL ---
def run(name, pred_cfg, band_cfg, use_boost, rows):
    for pn in "AB":
        Yt, Pt = F["pr"][pn]
        pred = Pt if pred_cfg is None else apply_ls(F, Pt, "both", pred_cfg[0], pred_cfg[1], tabs[("both",) + pred_cfg])
        if band_cfg is None: centre, g = Pt, 1.0
        elif band_cfg == "pred": centre, g = pred, 1.0
        else:
            _, g, tb = band[band_cfg]; centre = apply_ls(F, Pt, "both", band_cfg[0], band_cfg[1], tb)
        out = boost(pred, F["X"]) if use_boost else pred
        s = score(out, Yt, centre, g * F["h8"])
        rows.setdefault(name, {})[pn] = s
    b = rows["v8"]; s = rows[name]
    dA, dB = delta(s["A"], b["A"]), delta(s["B"], b["B"])
    fmt = lambda d: " ".join(f"{k} {d[k]:+.3f}" for k in ("rel", "tke", "mvpe", "sps", "T", "comp"))
    print(f"{name:34} A[{fmt(dA)}]  B[{fmt(dB)}]  (A sps {s['A']['sps']:.2f} T {s['A']['T']:.4f} | B sps {s['B']['sps']:.2f} T {s['B']['T']:.4f})", flush=True)

rows = {}
run("v8", None, None, False, rows)
for pn in "AB": print(" v8", pn, {k: round(v, 4) for k, v in rows["v8"][pn].items()})
run("v16 (boost)", None, None, True, rows)
for pc in (("sl", 6, "A"), ("sl", 6, "AB"), ("d2", 6, "AB"), ("sl", 3, "AB")):
    run(f"pred{pc}", pc, None, False, rows)
    run(f"pred{pc}+boost", pc, None, True, rows)
    run(f"pred{pc}+centre=pred+boost", pc, "pred", True, rows)
for bc in band:
    run(f"band{bc}", None, bc, False, rows)
    run(f"band{bc}+boost", None, bc, True, rows)
for pc in (("sl", 6, "A"), ("sl", 6, "AB")):
    for bc in (("sl", 6, "A"), ("sl", 6, "AB"), ("sl", 6, "B")):
        run(f"pred{pc}+band{bc}+boost", pc, bc, True, rows)
        run(f"pred{pc}+band{bc}", pc, bc, False, rows)
pickle.dump(rows, open("rows_final.pkl", "wb"))
