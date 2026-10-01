"""C2: official CNO vs official FNO, both under proxy AT (each model's own residual inflated, Y' = P + KT (Y - P)),
v17pw recipe refit for CNO on SELECT (as v17pw was for the FNO). Scored per sample (chunked) on SELECT and FINAL.
usage: python experiments/cno_check/eval_cno.py"""
import os, sys, gc, pickle, importlib.util as iu
sys.dont_write_bytecode = True
import numpy as np
LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(LAB, "pw"))
import blend19 as B
TG = B.tg
sys.path.insert(0, os.path.join(LAB, "c1"))
import sflib as L
B.tg = TG                                    # restore the proxy targets (sflib sets honest ones)
MM = os.path.join(os.environ.get("TEMP", "."), "rpde_mm")
OUT = os.path.dirname(os.path.abspath(__file__))
PSRC = dict(official=lambda r: os.path.join(MM, f"{r}_P.npy"), cno=lambda r: os.path.join(OUT, f"P_cno_{r}.npy"))


def mkD(role, X, Y, P):
    E, valid = B.estimators(X)
    D = dict(X=X, Y=Y, P=P, Em=E["mean"], El=E["last"], valid=valid, role=role, L=None)
    D["bins"] = B.rank_bins(B.sigma_local(X).astype(np.float32), valid, 6, 1)
    D["p1b"] = B.rank_bins(np.sqrt(((X[:, 1:] - X[:, :-1]) ** 2).mean(1)).astype(np.float32), valid, 10, 1)
    return D


def fit(model):
    X = np.load(os.path.join(MM, "select_X.npy")); Y = np.load(os.path.join(MM, "select_Y.npy")); P = np.load(PSRC[model]("select"))
    D = mkD("select", X, Y, P); D["h8"] = B.v8_half(X, True, 1)
    pt = B.fit_pred(D, ("A",), False, True); bt = B.fit_band(D, ("A", "B"), False); H = B.fit_width(D, bt, ("AT", "BT"))
    bt_raw = bt; H_raw = H          # raw+intervals variant: same centre/width recipe, prediction left raw, no boost
    del D, X, Y, P; gc.collect()
    return pt, bt, H


def evaluate(role, model, pt, bt, H, boost, prx="AT", chunk=40):
    X = np.load(os.path.join(MM, f"{role}_X.npy"), mmap_mode="r"); Y = np.load(os.path.join(MM, f"{role}_Y.npy"), mmap_mode="r")
    P = np.load(PSRC[model](role), mmap_mode="r"); T = np.load(os.path.join(MM, f"{role}_traj.npy"))
    n = len(X); acc = {k: np.zeros(n) for k in ("rel", "tke", "mvpe")}; S = np.zeros(n); N = np.zeros(n); C = np.zeros(n)
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        D = mkD(role, np.asarray(X[sl]), np.asarray(Y[sl]), np.asarray(P[sl]))
        Yt, Pt = B.tg(D, prx)
        pred = B.apply(D, Pt, pt) if pt is not None else Pt
        if boost: pred = B.boost(pred, D["X"])
        c = B.apply(D, Pt, bt); h = B.width_of(D, H)
        acc["rel"][sl] = L.sc.rel_l2_per_sample(pred, Yt, 2); acc["tke"][sl] = L.sc.tke_rel_l2_per_sample(pred, Yt, 2)
        acc["mvpe"][sl] = L.sc.mvpe_rel_l2_per_sample(pred, Yt)
        ins = np.abs(Yt - c) <= h; scd = Yt != 0
        S[sl] = np.where(ins & scd, np.exp(-2 * h / L.SIG), 0).sum((1, 2, 3, 4)); N[sl] = scd.sum((1, 2, 3, 4)); C[sl] = (ins & scd).sum((1, 2, 3, 4))
        del D, Yt, Pt, pred, c, h; gc.collect()
    return {s: L.summarise(acc, (S, N, C), idx) for s, idx in L.subsets(dict(traj=T)).items()}


if __name__ == "__main__":
    s = iu.spec_from_file_location("t17", os.path.join(LAB, "build_v17pw", "_blend_tables.py")); t = iu.module_from_spec(s); s.loader.exec_module(t)
    v17 = (np.array(t.PRED_TAB), np.array(t.BAND_TAB), np.array(t.WIDTH_TAB))
    tabs = fit("cno"); pickle.dump(tabs, open(os.path.join(OUT, "tabs_cno_v17pw.pkl"), "wb")); print("CNO tables fitted on SELECT", flush=True)
    rows = [("FNO + v17pw (shipped)", "official", v17[0], v17[1], v17[2], True),
            ("FNO raw + v17pw bands", "official", None, v17[1], v17[2], False),
            ("CNO + v17pw refit", "cno", tabs[0], tabs[1], tabs[2], True),
            ("CNO raw + refit bands", "cno", None, tabs[1], tabs[2], False)]
    res = {}
    for role in ("select", "final"):
        for prx in ("AT", "R"):
            if prx == "R": B.tg, tgp = (lambda D, pn: (D["Y"], D["P"])), "AT"
            else: B.tg, tgp = TG, "AT"
            for name, model, pt, bt, H, bo in rows:
                r = evaluate(role, model, pt, bt, H, bo, tgp); res[(role, prx, name)] = r
                for sname, z in r.items():
                    print(f"{role:6s} {'proxyAT' if prx == 'AT' else 'raw-Y  '} {name:24s} {sname:6s} {L.fmt(z)}", flush=True)
    B.tg = TG
    pickle.dump(res, open(os.path.join(OUT, "eval_cno.pkl"), "wb"))
