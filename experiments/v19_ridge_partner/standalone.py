"""Standalone honest errors of L vs Em, El, P (P memorized: not honest). usage: standalone.py stats.npz"""
import os
import sys, itertools
from ridge import *
sys.path.insert(0, os.path.join(HERE, "..", "interval_recipes"))
from common import load, sc
st = dict(np.load(os.environ.get("STATS", "stats_s4.npz")))
roles = {r: load(r) for r in ("select", "final")}

def ests(X):
    nz = X != 0; cnt = nz.sum(1); em = X.sum(1) / np.maximum(cnt, 1)
    el = np.where(nz[:, -1], X[:, -1], em)
    return em, el, cnt > 0

def report(tag, L, role):
    X, Y, P, _ = roles[role]
    em, el, valid = ests(X)
    bins = rank_bins_t(sigma_local_t(torch.from_numpy(X)), torch.from_numpy(valid)).numpy()
    m = (Y != 0) & valid[:, None]
    calm = m & (bins[:, None] <= 1)
    res = {}
    for nm, Z in (("L", L), ("Em", np.broadcast_to(em[:, None], Y.shape)), ("El", np.broadcast_to(el[:, None], Y.shape)), ("P", P)):
        Z = np.where(m, Z, 0).astype(np.float32)
        ae = np.abs(Z - Y)
        rel = float(sc.rel_l2_per_sample(np.where(m, Z, P).astype(np.float32), Y, 2).mean())
        res[nm] = (rel, ae[m].mean(), ae[calm].mean(), ae[:, :5][calm[:, :5]].mean(), ae[:, 15:][calm[:, 15:]].mean())
    print(f"{tag:28s} {role:6s} " + "  ".join(f"{k}: rel {v[0]:.4f} mae {v[1]:.5f} calm {v[2]:.5f} (l0-4 {v[3]:.5f}, l15-19 {v[4]:.5f})" for k, v in res.items() if k == "L" or tag == "HDR"), flush=True)
    return res

if __name__ == "__main__":
    for role in ("select", "final"):
        report("HDR", roles[role][2], role)
    for m, k in [(2, 3), (5, 3), (10, 3), (2, 5), (5, 5), (10, 5)]:
        for lam in (1e-4, 1e-2):
            idx, W = solve(st, m, k, lam)
            for role in ("select", "final"):
                report(f"sext m{m} k{k} lam{lam:g}", predict_L(roles[role][0], idx, W), role)
