"""Per-element features computable from the input window + the prediction only."""
import numpy as np
import torch
import torch.nn.functional as F

def pixel_feats(X, P):
    """X,P: (n,20,H,W,2). Returns dict of (n,H,W,2) arrays (per pixel, per channel)."""
    n, T, H, W, C = X.shape
    valid = (X != 0).any(axis=1)                         # (n,H,W,2)
    xm = X.mean(1)
    sl = np.sqrt(((X - xm[:, None]) ** 2).mean(1))
    fd = np.sqrt((np.diff(X, axis=1) ** 2).mean(1))      # input frame-diff rms
    pm = P.mean(1)
    ps = np.sqrt(((P - pm[:, None]) ** 2).mean(1))       # pred temporal std
    jump = np.abs(P[:, 0] - X[:, -1])                    # continuity jump
    # last-5 input std
    sl5 = X[:, -5:].std(1)
    # spatial grad magnitude of input mean
    gy, gx = np.gradient(xm, axis=(1, 2))
    grad = np.sqrt(gx ** 2 + gy ** 2)
    pgy, pgx = np.gradient(pm, axis=(1, 2))
    pgrad = np.sqrt(pgx ** 2 + pgy ** 2)
    # valid-neighbour fraction (5x5) -- distance-to-mask proxy
    vm = torch.from_numpy(valid.any(-1).astype(np.float32))[:, None]
    nb5 = F.avg_pool2d(vm, 5, 1, 2, count_include_pad=False)[:, 0].numpy()
    nb9 = F.avg_pool2d(vm, 9, 1, 4, count_include_pad=False)[:, 0].numpy()
    # smoothed sl (5x5)
    slt = torch.from_numpy(np.ascontiguousarray(sl.transpose(0, 3, 1, 2)))
    sls = F.avg_pool2d(slt, 5, 1, 2, count_include_pad=False).numpy().transpose(0, 2, 3, 1)
    # window-level activity
    vv = valid
    slw = np.array([[sl[i, ..., c][vv[i, ..., c]].mean() if vv[i, ..., c].any() else 0 for c in range(2)] for i in range(n)])
    slw = np.broadcast_to(slw[:, None, None, :], sl.shape)
    yy, xx = np.meshgrid(np.arange(H) / H, np.arange(W) / W, indexing="ij")
    yy = np.broadcast_to(yy[None, :, :, None], sl.shape); xx = np.broadcast_to(xx[None, :, :, None], sl.shape)
    other = sl[..., ::-1]
    return dict(sl=sl, fd=fd, absxm=np.abs(xm), xm=xm, ps=ps, pm=pm, jump=jump, sl5=sl5, grad=grad,
                pgrad=pgrad, nb5=np.broadcast_to(nb5[..., None], sl.shape), nb9=np.broadcast_to(nb9[..., None], sl.shape),
                sls=sls, slw=slw, y=yy, x=xx, other=other, valid=valid)

def elem_feats(X, P, pf=None):
    """Per-element feature stack (n,20,H,W,2,F) as float32 plus names."""
    if pf is None:
        pf = pixel_feats(X, P)
    n, T, H, W, C = P.shape
    names = [k for k in pf if k != "valid"]
    cols = [np.broadcast_to(pf[k][:, None], P.shape) for k in names]
    t = np.broadcast_to(np.arange(T, dtype=np.float32)[None, :, None, None, None], P.shape)
    dev = np.abs(P - pf["pm"][:, None])
    cols += [t, np.abs(P), dev]
    names += ["t", "absP", "pdev"]
    return names, cols
