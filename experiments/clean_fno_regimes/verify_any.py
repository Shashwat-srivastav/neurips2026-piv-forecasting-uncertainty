"""Run a build's actual submission.py over all SELECT and FINAL windows (chunked, mmap inputs), score per AoA
subset vs v17 LIVE, and check per-window independence.
usage: python experiments/clean_fno_regimes/verify_any.py <build_dir> [kY]   (kY != 1: proxy target Y' = Pofficial + kY (Y - Pofficial))"""
import os, sys, gc
sys.dont_write_bytecode = True
import numpy as np
bd = os.path.abspath(sys.argv[1]); kY = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
sys.path.insert(0, bd); sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import submission as S
import sflib as L
MM = os.path.join(os.environ.get("TEMP", "."), "rpde_mm")
SIG = L.SIG
for role in ("select", "final"):
    X = np.load(os.path.join(MM, f"{role}_X.npy"), mmap_mode="r"); Y = np.load(os.path.join(MM, f"{role}_Y.npy"), mmap_mode="r")
    Po = np.load(os.path.join(MM, f"{role}_P.npy"), mmap_mode="r"); T = np.load(os.path.join(MM, f"{role}_traj.npy"))
    n = len(X); acc = {k: np.zeros(n) for k in ("rel", "tke", "mvpe")}; Sx = np.zeros(n); N = np.zeros(n); C = np.zeros(n)
    for i in range(0, n, 5):
        xb = np.asarray(X[i:i+5]); yb = np.asarray(Y[i:i+5])
        if kY != 1.0:
            pb = np.asarray(Po[i:i+5]); yb = np.where(yb == 0, 0, pb + kY * (yb - pb)).astype(np.float32)
        o = S.predict(np.concatenate([xb, np.zeros(xb.shape[:-1] + (1,), np.float32)], -1))
        assert all(np.isfinite(o[k]).all() for k in o) and (o["lower"] <= o["upper"]).all()
        pr, lo, up = o["prediction"][..., :2], o["lower"][..., :2], o["upper"][..., :2]
        acc["rel"][i:i+5] = L.sc.rel_l2_per_sample(pr, yb, 2); acc["tke"][i:i+5] = L.sc.tke_rel_l2_per_sample(pr, yb, 2)
        acc["mvpe"][i:i+5] = L.sc.mvpe_rel_l2_per_sample(pr, yb)
        ins = (yb >= lo) & (yb <= up); scd = yb != 0
        Sx[i:i+5] = np.where(ins & scd, np.exp(-(up - lo) / SIG), 0).sum((1, 2, 3, 4)); N[i:i+5] = scd.sum((1, 2, 3, 4)); C[i:i+5] = (ins & scd).sum((1, 2, 3, 4))
    for sname, idx in L.subsets(dict(traj=T)).items():
        print(f"BUILD {os.path.basename(bd)} {role:6s} {sname:6s} ({len(set(T[idx]))} traj) {L.fmt(L.summarise(acc, (Sx, N, C), idx))}"
              + ("" if kY == 1 else f"  [proxy k={kY}]"), flush=True)
    del X, Y; gc.collect()
X = np.load(os.path.join(MM, "final_X.npy"), mmap_mode="r")
x3 = np.concatenate([np.asarray(X[:20]), np.zeros((20,) + X.shape[1:-1] + (1,), np.float32)], -1)
w = 0.0
for rows in (np.arange(0, 5), np.array([7, 3, 11, 19, 2]), np.array([4, 9]), np.array([13, 0, 17])):
    ob = S.predict(x3[rows])
    for j, r in enumerate(rows):
        oa = S.predict(x3[r:r + 1])
        w = max(w, max(float(np.abs(ob[k][j] - oa[k][0]).max()) for k in ob))
print(f"INDEPENDENCE worst |batched - alone| = {w:.2e} -> {'PASS' if w < 1e-6 else 'FAIL'}", flush=True)
