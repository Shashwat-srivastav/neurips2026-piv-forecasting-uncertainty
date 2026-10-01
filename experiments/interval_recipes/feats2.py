"""Per-element features (input window + prediction only), chunked, float32."""
import numpy as np, torch, torch.nn.functional as F

PIX = ["sl", "sl5", "p1", "p2", "p5", "p10", "d1", "d5", "gx", "axm", "nb5", "yy", "xx", "slw", "oth", "ps", "pg"]
ELM = ["t", "aP", "dT", "gP", "pdev"]
NAMES = PIX + ELM

def _grad(a):  # a (...,H,W)
    gy, gx = np.gradient(a, axis=(-2, -1))
    return np.sqrt(gx * gx + gy * gy)

def chunk_feats(X, P, c):
    """X,P (n,20,H,W,2); channel c. Returns (n,20,H,W,F) float32."""
    n, T, H, W, _ = X.shape
    x = X[..., c]; p = P[..., c]
    valid = (X != 0).any(axis=1).any(-1)          # (n,H,W)
    xm = x.mean(1)
    sl = np.sqrt(((x - xm[:, None]) ** 2).mean(1))
    xo = X[..., 1 - c]
    oth = np.sqrt(((xo - xo.mean(1, keepdims=True)) ** 2).mean(1))
    pers = {k: np.sqrt(((x[:, k:] - x[:, :-k]) ** 2).mean(1)) for k in (1, 2, 5, 10)}
    pm = p.mean(1)
    vm = torch.from_numpy(valid.astype(np.float32))[:, None]
    nb5 = F.avg_pool2d(vm, 5, 1, 2, count_include_pad=False)[:, 0].numpy()
    slw = np.array([sl[i][valid[i]].mean() if valid[i].any() else 0.0 for i in range(n)])
    yy, xx = np.meshgrid(np.arange(H) / H, np.arange(W) / W, indexing="ij")
    pix = dict(sl=sl, sl5=x[:, -5:].std(1), p1=pers[1], p2=pers[2], p5=pers[5], p10=pers[10],
               d1=np.abs(x[:, -1] - x[:, -2]), d5=np.abs(x[:, -1] - x[:, -6]), gx=_grad(x[:, -1]),
               axm=np.abs(xm), nb5=nb5, yy=np.broadcast_to(yy, sl.shape), xx=np.broadcast_to(xx, sl.shape),
               slw=np.broadcast_to(slw[:, None, None], sl.shape), oth=oth,
               ps=np.sqrt(((p - pm[:, None]) ** 2).mean(1)), pg=_grad(pm))
    out = np.empty((n, T, H, W, len(NAMES)), np.float32)
    for j, k in enumerate(PIX):
        out[..., j] = pix[k][:, None]
    out[..., len(PIX) + 0] = np.arange(T, dtype=np.float32)[None, :, None, None]
    out[..., len(PIX) + 1] = np.abs(p)
    out[..., len(PIX) + 2] = np.abs(p - x[:, -1:])
    out[..., len(PIX) + 3] = _grad(p)
    out[..., len(PIX) + 4] = np.abs(p - pm[:, None])
    return out
