"""C3: does averaging Run 1 (honest) with the official FNO help? Official forecast de-memorised by inflating its error:
Po' = Y + k (Po - Y) (k = 1.447 as the live-calibrated proxy; k = 1 contaminated, shown for reference).
All rows scored against the real Y (so the two models' error correlation is kept). Accuracy only (rel/tke/mvpe)."""
import os, sys
sys.dont_write_bytecode = True
import numpy as np
LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(LAB, "c1"))
import sflib as L
MM = os.path.join(os.environ.get("TEMP", "."), "rpde_mm")
for role in ("select", "final"):
    Y = np.load(os.path.join(MM, f"{role}_Y.npy"), mmap_mode="r"); Po = np.load(os.path.join(MM, f"{role}_P.npy"), mmap_mode="r")
    P1 = np.load(os.path.join(LAB, "c1", f"P_run1_{role}.npy"), mmap_mode="r"); T = np.load(os.path.join(MM, f"{role}_traj.npy"))
    n = len(Y); idx15 = np.array([t.split("_")[1] == "15" for t in T])
    for k in (1.447, 1.0):
        rows = {}
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):     # w = weight on the official FNO
            a = {m: np.zeros(n) for m in ("rel", "tke", "mvpe")}
            for i in range(0, n, 20):
                y = np.asarray(Y[i:i+20]); po = y + k * (np.asarray(Po[i:i+20]) - y)
                p = (1 - w) * np.asarray(P1[i:i+20]) + w * po
                a["rel"][i:i+20] = L.sc.rel_l2_per_sample(p, y, 2); a["tke"][i:i+20] = L.sc.tke_rel_l2_per_sample(p, y, 2)
                a["mvpe"][i:i+20] = L.sc.mvpe_rel_l2_per_sample(p, y)
            for sname, sel in (("all", np.ones(n, bool)), ("AoA15", idx15)):
                if role == "select" and sname == "AoA15": continue
                r = {m: L.sub(a[m][sel].mean()) for m in a}
                acc_c = sum(L.W[m] * (r[m] - L.LIVE[m]) for m in ("rel", "tke", "mvpe"))
                print(f"{role:6s} {sname:5s} k={k:.3f} w_official={w:.2f}  rel {r['rel']:.3f} tke {r['tke']:.3f} mvpe {r['mvpe']:.3f}"
                      f" | accuracy part of composite vs v17 LIVE {acc_c:+.3f}", flush=True)
