"""Dump, per role & channel, scored-element arrays: |e| (raw, k=1), lead t, and candidate scores."""
import os, numpy as np
from common import load, v8_half
from feats2 import chunk_feats, NAMES
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "dump"); os.makedirs(OUT, exist_ok=True)
EPS = 1e-4
LOGF = ["p1", "p2", "p5", "sl", "sl5", "d1", "dT", "gP", "aP", "ps", "oth", "pg", "gx", "axm", "pdev"]
LINF = ["t", "yy", "xx", "nb5", "slw"]
def rank_in_call(a, valid):
    """within-call (5 windows) fractional rank of per-pixel a (n,H,W) among valid cells."""
    out = np.zeros_like(a, dtype=np.float32)
    for i in range(0, len(a), 5):
        blk = a[i:i+5]; v = valid[i:i+5]; pool = np.sort(blk[v])
        out[i:i+5] = np.searchsorted(pool, blk) / max(len(pool), 1)
    return out
for role in ("select", "final"):
    X, Y, P, traj = load(role)
    h8 = v8_half(X)
    for c in (0, 1):
        cols = {k: [] for k in ["ae", "h8", "rk_p1", "rk_sl", "win"] + LOGF + LINF}
        for i in range(0, len(X), 10):
            x, p, y = X[i:i+10], P[i:i+10], Y[i:i+10, ..., c]
            f = chunk_feats(x, p, c); m = y != 0
            cols["ae"].append(np.abs(y - p[..., c])[m])
            cols["h8"].append(np.broadcast_to(h8[i:i+10, ..., c], m.shape)[m])
            valid = (x[..., c] != 0).any(1)
            for nm, src in (("rk_p1", "p1"), ("rk_sl", "sl")):
                pass
            cols["win"].append(np.broadcast_to(np.arange(i, i + len(x))[:, None, None, None], m.shape)[m].astype(np.int32))
            for k in LOGF + LINF:
                cols[k].append(f[..., NAMES.index(k)][m])
        # within-call ranks need full chunks of 5 aligned with harness calls: recompute on pixel level
        rk = {"rk_p1": [], "rk_sl": []}
        for i in range(0, len(X), 10):
            x, y = X[i:i+10], Y[i:i+10, ..., c]; m = y != 0; valid = (x[..., c] != 0).any(1)
            f0 = chunk_feats(x, P[i:i+10], c)
            for nm, src in (("rk_p1", "p1"), ("rk_sl", "sl")):
                r = rank_in_call(f0[:, 0, ..., NAMES.index(src)], valid)
                rk[nm].append(np.broadcast_to(r[:, None], m.shape)[m])
        cols.update(rk)
        np.savez(os.path.join(OUT, f"{role}_{c}.npz"), **{k: np.concatenate(v) for k, v in cols.items()})
        print(role, c, "done", flush=True)
    del X, Y, P
