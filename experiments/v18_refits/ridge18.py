"""Ridge log-|error| score (experiments/interval_recipes/rlib.py feature set), refit to v18 centre residuals; used via within-call rank bins."""
import numpy as np, torch, torch.nn.functional as Fn
EPS = 1e-4
LOGF = ["p1", "p2", "p5", "sl", "sl5", "d1", "dT", "gP", "aP", "ps", "oth", "pg", "gx", "axm", "pdev"]
LINF = ["t", "yy", "xx", "nb5", "slw"]
NF = len(LOGF) + len(LINF) + 4


def _grad(a):
    gy, gx = np.gradient(a, axis=(-2, -1))
    return np.sqrt(gx * gx + gy * gy)


def feats(X, P, c):
    """X,P (n,20,H,W,2) -> (n,20,H,W,NF) design for channel c (float32)."""
    n, T, H, W, _ = X.shape
    x = X[..., c]; p = P[..., c]
    valid = (X != 0).any(axis=1).any(-1)
    xm = x.mean(1); sl = np.sqrt(((x - xm[:, None]) ** 2).mean(1))
    xo = X[..., 1 - c]; oth = np.sqrt(((xo - xo.mean(1, keepdims=True)) ** 2).mean(1))
    per = {k: np.sqrt(((x[:, k:] - x[:, :-k]) ** 2).mean(1)) for k in (1, 2, 5)}
    pm = p.mean(1)
    vm = torch.from_numpy(valid.astype(np.float32))[:, None]
    nb5 = Fn.avg_pool2d(vm, 5, 1, 2, count_include_pad=False)[:, 0].numpy()
    slw = np.array([sl[i][valid[i]].mean() if valid[i].any() else 0.0 for i in range(n)], np.float32)
    yy, xx = np.meshgrid(np.arange(H) / H, np.arange(W) / W, indexing="ij")
    pix = dict(sl=sl, sl5=x[:, -5:].std(1), p1=per[1], p2=per[2], p5=per[5], d1=np.abs(x[:, -1] - x[:, -2]),
               gx=_grad(x[:, -1]), axm=np.abs(xm), oth=oth, ps=np.sqrt(((p - pm[:, None]) ** 2).mean(1)), pg=_grad(pm))
    elm = dict(aP=np.abs(p), dT=np.abs(p - x[:, -1:]), gP=_grad(p), pdev=np.abs(p - pm[:, None]))
    out = np.empty((n, T, H, W, NF), np.float32); j = 0
    for k in LOGF:
        v = pix[k][:, None] if k in pix else elm[k]
        out[..., j] = np.log(v + EPS); j += 1
    t = np.arange(T, dtype=np.float32)[None, :, None, None]
    lin = dict(t=t, yy=yy[None, None], xx=xx[None, None], nb5=nb5[:, None], slw=slw[:, None, None, None])
    for k in LINF:
        out[..., j] = lin[k]; j += 1
    out[..., j] = t ** 2 / 400; out[..., j + 1] = (yy ** 2)[None, None]; out[..., j + 2] = (xx ** 2)[None, None]; out[..., j + 3] = 1.0
    return out


def fit(X, P, Yt, C, valid, nsub=1_500_000, seed=0, chunk=20):
    """W (2,NF): ridge of log(|Yt-C|+EPS) on feats, scored elements only."""
    rng = np.random.default_rng(seed); Wt = np.zeros((2, NF))
    for c in (0, 1):
        AtA = np.zeros((NF, NF)); Atb = np.zeros(NF); ntot = 0
        for i in range(0, len(X), chunk):
            f = feats(X[i:i+chunk], P[i:i+chunk], c)
            m = (Yt[i:i+chunk, ..., c] != 0) & valid[i:i+chunk, None, ..., c]
            A = f[m].astype(np.float64); b = np.log(np.abs(Yt[i:i+chunk, ..., c] - C[i:i+chunk, ..., c])[m] + EPS)
            keep = rng.random(len(b)) < 0.5
            A, b = A[keep], b[keep]
            AtA += A.T @ A; Atb += A.T @ b; ntot += len(b)
        Wt[c] = np.linalg.solve(AtA + 1e-3 * np.eye(NF), Atb)
    return Wt


def score(X, P, Wt, chunk=20):
    out = np.empty(X.shape, np.float32)
    for i in range(0, len(X), chunk):
        for c in (0, 1):
            out[i:i+chunk, ..., c] = feats(X[i:i+chunk], P[i:i+chunk], c) @ Wt[c].astype(np.float32)
    return out


def rank_bins_lg(S, valid, nb, batch=5):
    """Within-call rank bins of S (n,20,H,W,2) per (lead group of 5, channel), valid pixels only."""
    out = np.zeros(S.shape, np.int64); qs = np.linspace(1.0 / nb, 1.0 - 1.0 / nb, nb - 1)
    for i in range(0, len(S), batch):
        for lg in range(4):
            for c in (0, 1):
                s = S[i:i+batch, lg*5:(lg+1)*5, ..., c]
                v = np.broadcast_to(valid[i:i+batch, None, ..., c], s.shape)
                e = np.quantile(s[v].astype(np.float32), qs).astype(np.float32)
                out[i:i+batch, lg*5:(lg+1)*5, ..., c] = np.searchsorted(e, s, side="left")
    return np.clip(out, 0, nb - 1)
