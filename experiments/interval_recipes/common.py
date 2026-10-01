import os, sys, importlib.util
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", "..", "realpde_t1_starting_kit_v9", "realpde_t1_starting_kit_v9"))
spec = importlib.util.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
SIG = sc.SIGMA_GLOBAL
LIVE = dict(rel=0.12393, tke=(100/74.025874-1)/0.5, mvpe=(100/92.836233-1)/0.5)

def load(role, stride=20):
    d = np.load(os.path.join(HERE, f"cache_{role}_s{stride}.npz"))
    return d["X"], d["Y"], d["P"], d["traj"]

def proxy_target(Y, P, k):
    Yp = P + k * (Y - P)
    return np.where(Y == 0.0, 0.0, Yp).astype(np.float32)

def rel_err(P, Y):
    return float(sc.rel_l2_per_sample(P, Y, 2).mean())

def fit_k(Y, P, metric="rel", target=None):
    f = {"rel": lambda Yp: sc.rel_l2_per_sample(P, Yp, 2).mean(),
         "tke": lambda Yp: sc.tke_rel_l2_per_sample(P, Yp, 2).mean(),
         "mvpe": lambda Yp: sc.mvpe_rel_l2_per_sample(P, Yp).mean()}[metric]
    target = LIVE[metric] if target is None else target
    lo, hi = 0.5, 6.0
    for _ in range(40):
        m = 0.5 * (lo + hi)
        if f(proxy_target(Y, P, m)) < target: lo = m
        else: hi = m
    return 0.5 * (lo + hi)

def official(P, Yp, h):
    """h: half-width broadcastable to P. Returns sps(0-100), T, cov, Wbar via the official scorer."""
    h = np.broadcast_to(h, P.shape).astype(np.float32)
    lo = (P - h).astype(np.float32); up = (P + h).astype(np.float32)
    w, cov = sc.aggregate_sps(P, Yp, 2, lower=lo, upper=up)
    nil = (up - lo) / SIG
    inside = (Yp >= lo) & (Yp <= up)
    scored = Yp != 0
    T = float(np.sum(np.where(inside, np.exp(-nil), 0.0), where=scored, dtype=np.float64) / scored.sum())
    return 100 * w, T, cov, w / T

# ---- v8 band exactly (per call of 5) ----
A_U, A_V, K = 0.004, 0.003, 1.5
NB = 6
MULT = ((1.50, 1.00, 0.80, 0.60, 0.40, 0.25), (0.90, 0.90, 0.80, 0.60, 0.40, 0.25))

def sigma_local(X):
    xc = X[..., :2]
    fl = xc - xc.mean(axis=1, keepdims=True)
    return np.sqrt((fl * fl).mean(axis=1))  # (n,H,W,2)

def v8_half(X, shape=True, batch=5):
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i+batch]))
        sl = torch.from_numpy(sigma_local(X[i:i+batch]))
        mult = torch.ones_like(sl)
        if shape:
            valid = (xb[..., :2] != 0.0).any(dim=1)
            qs = torch.linspace(1.0/NB, 1.0-1.0/NB, NB-1)
            for c in (0, 1):
                pool = sl[..., c][valid[..., c]]
                edges = torch.quantile(pool.float(), qs)
                idx = torch.bucketize(sl[..., c].contiguous(), edges.contiguous())
                mult[..., c] = torch.tensor(MULT[c])[idx.clamp(0, NB-1)]
        hb = torch.stack([(A_U + K*sl[..., 0])*mult[..., 0], (A_V + K*sl[..., 1])*mult[..., 1]], -1)
        out.append(hb.numpy()[:, None])
    return np.concatenate(out)
