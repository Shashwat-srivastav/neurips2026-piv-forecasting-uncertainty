"""L with PER-WINDOW sextile classes (pw lib19). usage: gen_L_pw.py old  -> shipped W_sxb3 weights
                                                      gen_L_pw.py new  -> ridge re-solved from stats_pw_s4.npz (m10 k5 lam1e-4 sxb lam2 1e-3 = sxb3 recipe)"""
import sys, os
sys.dont_write_bytecode = True
from ridge import *
sys.path.insert(0, os.path.join(HERE, "..", "interval_recipes"))
from common import load
mode = sys.argv[1]
if mode == "old":
    d = np.load(os.path.join(HERE, "..", "v19_ridge_partner", "W_sxb3.npz")); idx, W = d["idx"], d["W"]
else:
    st = dict(np.load("stats_pw_s4.npz"))
    idx, W = solve(st, 10, 5, 1e-4, "sxb", 1e-3)
    np.savez("W_pw.npz", idx=idx, W=W, m=10, k=5)
for role in ("select", "final"):
    L = predict_L(load(role)[0], idx, W); np.save(f"L_{mode}_pw_{role}.npy", L); print(role, L.shape, flush=True)
