"""Shared helpers for the per-element blend study (experiments/estimator_blend). Proxy-based only."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "interval_recipes"))
from common import load, sc, SIG, sigma_local  # noqa
from common import v8_half as _v8_half_common
def v8_half(X, shape=True, batch=1):
    """pw: v8 half-width with PER-WINDOW sextiles by default (only used as the fitting width for band centres)."""
    return _v8_half_common(X, shape, batch)
KA = {"select": 1.534, "final": 1.665}   # proxy A k (experiments/interval_recipes/validate_proxy.py, rel-matched)
LIVE_REL = 0.12393
W8 = dict(rel=.454, tke=.087, mvpe=.080, sps=.284)

def s_err(e): return 100.0 / (1.0 + 0.5 * e)

def estimators(X):
    """Input-only trivial estimators, (n,H,W,2) each; masked-aware."""
    nz = X != 0
    cnt = nz.sum(1)
    valid = cnt > 0
    mean = X.sum(1) / np.maximum(cnt, 1)
    last = np.where(nz[:, -1], X[:, -1], mean)
    return dict(mean=mean.astype(np.float32), last=last.astype(np.float32)), valid

def d2rms(X):
    nz = X != 0
    d2 = X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]
    ok = nz[:, 2:] & nz[:, 1:-1] & nz[:, :-2]
    return np.sqrt((d2 * d2 * ok).sum(1) / np.maximum(ok.sum(1), 1)).astype(np.float32)

def rank_bins(F, valid, nb, batch=1):
    """PER-WINDOW rank bins (per channel) by default (batch=1): edges from that window's own valid pixels only.
    batch=5 reproduces the shipped within-call bins. Pools smaller than nb -> bin 0 (as torch)."""
    out = np.zeros(F.shape, np.int64)
    if nb == 1:
        return out
    qs = np.linspace(1.0 / nb, 1.0 - 1.0 / nb, nb - 1)
    for i in range(0, len(F), batch):
        for c in (0, 1):
            f = F[i:i+batch, ..., c]; v = valid[i:i+batch, ..., c]
            if v.sum() < nb: continue
            edges = np.quantile(f[v].astype(np.float32), qs).astype(np.float32)
            out[i:i+batch, ..., c] = np.searchsorted(edges, f, side="left")
    return np.clip(out, 0, nb - 1)

def proxies(role, Y, P):
    ka = KA[role]
    YA = np.where(Y == 0.0, 0.0, P + ka * (Y - P)).astype(np.float32)
    kb = LIVE_REL / float(sc.rel_l2_per_sample(P, Y, 2).mean())
    PB = (Y + kb * (P - Y)).astype(np.float32)
    return dict(A=(YA, P), B=(Y, PB)), (ka, kb)

def boost(pred, X):
    """v16 tke fluctuation boost, exact numpy transcription."""
    xc = X[..., :2]; fl = xc - xc.mean(1, keepdims=True)
    e_in = (fl * fl).mean(1).sum((1, 2, 3))
    ym = pred.mean(1, keepdims=True); fp = pred - ym
    e_pr = (fp * fp).mean(1).sum((1, 2, 3))
    a = np.clip((e_in / np.maximum(e_pr, 1e-12)) ** 0.25, 1.0, 1.6).astype(np.float32)
    return (ym + a[:, None, None, None, None] * fp).astype(np.float32)

def score(pred, Yt, centre, h):
    rel = float(sc.rel_l2_per_sample(pred, Yt, 2).mean())
    tke = float(sc.tke_rel_l2_per_sample(pred, Yt, 2).mean())
    mv = float(sc.mvpe_rel_l2_per_sample(pred, Yt).mean())
    h = np.broadcast_to(h, pred.shape)
    lo = (centre - h).astype(np.float32); up = (centre + h).astype(np.float32)
    assert np.all(lo <= up) and np.isfinite(lo).all() and np.isfinite(up).all()
    w, cov = sc.aggregate_sps(pred, Yt, 2, lower=lo, upper=up)
    inside = (Yt >= lo) & (Yt <= up); scored = Yt != 0
    T = float(np.sum(np.where(inside, np.exp(-(up - lo) / SIG), 0.0), where=scored, dtype=np.float64) / scored.sum())
    return dict(rel=rel, tke=tke, mvpe=mv, sps=100 * w, T=T, cov=cov)

def delta(s, b):
    d = dict(rel=s_err(s["rel"]) - s_err(b["rel"]), tke=s_err(s["tke"]) - s_err(b["tke"]),
             mvpe=s_err(s["mvpe"]) - s_err(b["mvpe"]), sps=s["sps"] - b["sps"], T=s["T"] - b["T"])
    d["comp"] = sum(W8[k] * d[k] for k in W8)
    return d
