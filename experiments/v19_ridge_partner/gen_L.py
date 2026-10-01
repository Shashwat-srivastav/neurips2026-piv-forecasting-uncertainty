"""usage: gen_L.py m k lam mode lam2 tag -> L_{tag}_{select,final}.npy + W_{tag}.npz"""
import sys
from standalone import *
m, k, lam, mode = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
lam2 = float(sys.argv[5]); tag = sys.argv[6]
idx, W = solve(st, m, k, lam, mode, lam2)
np.savez(f"W_{tag}.npz", idx=idx, W=W, m=m, k=k)
for role in ("select", "final"):
    L = predict_L(roles[role][0], idx, W); np.save(f"L_{tag}_{role}.npy", L)
    report(tag, L, role)
