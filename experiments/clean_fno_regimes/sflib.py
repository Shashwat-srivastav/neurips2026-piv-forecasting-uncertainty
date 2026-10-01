"""C1 helpers: memory-lean, per-window features, band-centre table fit, width tables (absolute and scale-free),
and a per-sample-decomposed scorer (so AoA subsets can be scored from one pass).
Everything a window gets depends on its own input only (per-window ranks, per-window medians)."""
import os, sys, gc
sys.dont_write_bytecode = True
import numpy as np, torch
LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(LAB, "pw"))
import blend19 as B  # noqa: E402
from base import sc, SIG  # noqa: E402

B.tg = lambda D, pn: (D["Y"], D["P"])     # honest: real targets
sub = lambda e: 100.0 / (1.0 + 0.5 * e)
W = dict(rel=.453, tke=.087, mvpe=.081, sps=.284)
LIVE = dict(rel=94.104, tke=75.368, mvpe=93.085, sps=37.287)
G = 400
HGRID = np.logspace(-5, 0, G)          # absolute half-width grid
CGRID = np.logspace(-3, 2, G)          # scale-free multiplier grid
SCALES = tuple(os.environ.get("SF_SCALES", "s_p1,s_sl").split(","))


def wmedian(F, valid):
    """(n,2) per-window, per-channel median of F over that window's valid pixels (linear interp, = np.median)."""
    n = F.shape[0]; out = np.ones((n, 2), np.float32)
    for i in range(n):
        for c in (0, 1):
            v = F[i, ..., c][valid[i, ..., c]]
            if v.size: out[i, c] = max(float(np.quantile(v.astype(np.float32), 0.5)), 1e-12)
    return out


def feats(X):
    E, valid = B.estimators(X)
    sl = B.sigma_local(X).astype(np.float32)
    p1 = np.sqrt(((X[:, 1:] - X[:, :-1]) ** 2).mean(1)).astype(np.float32)
    F = dict(Em=E["mean"], El=E["last"], valid=valid,
             bins=B.rank_bins(sl, valid, 6, 1), p1b=B.rank_bins(p1, valid, 10, 1),
             s_p1=wmedian(p1, valid), s_sl=wmedian(sl, valid))
    F["s_p1h"] = np.sqrt(F["s_p1"]); F["s_slh"] = np.sqrt(F["s_sl"])
    F["h8"] = B.v8_half(X, True, 1)                        # (n,1,H,W,2)
    return F


def load_split(role, Ptag):
    """Ptag: 'official' -> cache P; else path to a .npy of forecasts (mmap)."""
    d = np.load(os.path.join(LAB, "bands", f"cache_{role}_s20.npz"))
    X = d["X"]; F = feats(X); del X; gc.collect()
    F["Y"] = d["Y"]; F["traj"] = d["traj"]
    F["P"] = d["P"] if Ptag == "official" else np.load(Ptag, mmap_mode="r")
    F["role"] = role; F["L"] = None
    return F


def centre_t(F, P_t, bt, t):
    ch = np.arange(2)[None, None, None, :]; b = F["bins"]
    o = F["Em"] + bt[t, ch, b, 0] * (P_t - F["Em"]) + bt[t, ch, b, 1] * (F["El"] - F["Em"])
    return np.where(F["valid"], o, P_t).astype(np.float32)


def fit_band(F):
    D = dict(F); D["P"] = np.asarray(F["P"])
    return B.fit_band(D, ("A",), False)


def width_hists(F, bt, Yfn=None):
    """abs: (4,2,10,G) counts of |err| <= HGRID[g]; sf_p1 / sf_sl: (4,2,10,n,G) counts of |err|/s_w <= CGRID[g]."""
    n = len(F["Y"])
    Ha = np.zeros((4, 2, 10, G)); Hs = {k: np.zeros((4, 2, 10, n, G), np.float32) for k in SCALES}
    widx = np.broadcast_to(np.arange(n)[:, None, None], F["p1b"].shape[:3])
    for t in range(20):
        Y_t = F["Y"][:, t] if Yfn is None else Yfn(t)
        P_t = np.asarray(F["P"][:, t])
        ae = np.abs(Y_t - centre_t(F, P_t, bt, t)); m = (Y_t != 0) & F["valid"]; lg = t // 5
        for c in (0, 1):
            mc = m[..., c]; q = F["p1b"][..., c][mc]; a = ae[..., c][mc]; w = widx[mc]
            gi = np.searchsorted(HGRID, a, side="left")
            Ha[lg, c] += np.bincount(q * (G + 1) + gi, minlength=10 * (G + 1)).reshape(10, G + 1)[:, :G]
            for k in Hs:
                r = a / F[k][w, c]
                gi = np.searchsorted(CGRID, r, side="left")
                cnt = np.bincount((q * n + w) * (G + 1) + gi, minlength=10 * n * (G + 1)).reshape(10, n, G + 1)[..., :G]
                Hs[k][lg, c] += cnt
    return Ha, Hs


def solve_abs(Ha):
    cum = np.cumsum(Ha, -1); f = cum * np.exp(-2 * HGRID / SIG)
    return HGRID[f.argmax(-1)]                       # (4,2,10)


def solve_sf(Hs, s):
    """s: (n,2) window scales of the FIT split."""
    cum = np.cumsum(Hs, -1)                          # (4,2,10,n,G)
    out = np.zeros((4, 2, 10))
    for c in (0, 1):
        ex = np.exp(-2 * CGRID[None, :] * s[:, c][:, None] / SIG)   # (n,G)
        f = (cum[:, c] * ex[None, None]).sum(-2)                      # (4,10,G)
        out[:, c] = CGRID[f.argmax(-1)]
    return out


def acc_per_sample(F, chunk=20):
    n = len(F["Y"]); r = np.zeros(n); k = np.zeros(n); m = np.zeros(n)
    for i in range(0, n, chunk):
        P = np.asarray(F["P"][i:i+chunk]); Y = F["Y"][i:i+chunk]
        r[i:i+chunk] = sc.rel_l2_per_sample(P, Y, 2); k[i:i+chunk] = sc.tke_rel_l2_per_sample(P, Y, 2)
        m[i:i+chunk] = sc.mvpe_rel_l2_per_sample(P, Y)
    return dict(rel=r, tke=k, mvpe=m)


def band_per_sample(F, bt, H, mode):
    """per-sample S_i = sum over scored&inside of exp(-2h/SIG), n_i scored, cov_i inside."""
    n = len(F["Y"]); S = np.zeros(n); N = np.zeros(n); C = np.zeros(n)
    ch = np.arange(2)[None, None, None, :]
    for t in range(20):
        Y_t = F["Y"][:, t]; P_t = np.asarray(F["P"][:, t])
        h = H[t // 5][ch, F["p1b"]]
        if mode != "abs":
            h = h * F[mode][:, None, None, :]
        ins = np.abs(Y_t - centre_t(F, P_t, bt, t)) <= h; sc_ = Y_t != 0
        S += np.where(ins & sc_, np.exp(-2 * h / SIG), 0).sum((1, 2, 3)); N += sc_.sum((1, 2, 3)); C += (ins & sc_).sum((1, 2, 3))
    return S, N, C


def summarise(acc, band, idx):
    """leaderboard subscores + composite delta vs v17 live on sample subset idx."""
    S, N, C = band
    a = {k: acc[k][idx] for k in acc}
    pm = lambda e: e / (0.5 + e)
    wv = .5 * (1 - pm(a["rel"])) + .3 * (1 - pm(a["tke"])) + .2 * (1 - pm(a["mvpe"]))
    sps = 100 * float((wv * S[idx]).sum() / N[idx].sum())
    r = dict(rel=sub(a["rel"].mean()), tke=sub(a["tke"].mean()), mvpe=sub(a["mvpe"].mean()), sps=sps,
             cov=float(C[idx].sum() / N[idx].sum()))
    r["comp"] = sum(W[k] * (r[k] - LIVE[k]) for k in W)
    return r


def fmt(r):
    return (f"rel {r['rel']:.3f} tke {r['tke']:.3f} mvpe {r['mvpe']:.3f} sps {r['sps']:.2f} cov {r['cov']:.3f}"
            f" | vs v17 LIVE {r['comp']:+.3f}")


def subsets(F):
    T = F["traj"]; aoa = np.array([t.split("_")[1] for t in T])
    out = {"all": np.arange(len(T))}
    for a in sorted(set(aoa)): out[f"AoA{a}"] = np.where(aoa == a)[0]
    return out
