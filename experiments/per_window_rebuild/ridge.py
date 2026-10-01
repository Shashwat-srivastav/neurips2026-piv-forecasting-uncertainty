"""Solve ridge from stats and predict L on cache windows."""
from lib19 import *
NBLK = 16

def solve(st, m, k, lam, mode="sext", lam2=None):
    idx = feat_index(m, k)
    G = st["G"][:, :, idx][:, :, :, idx]; B = st["B"][:, :, idx]
    F = len(idx)
    def sol(g, b, prior=None, l2=None):
        s = np.trace(g) / F
        R = lam * s * np.eye(F); R[-1, -1] = 1e-9 * s
        if prior is None:
            return np.linalg.solve(g + R, b)
        R2 = l2 * s * np.eye(F)
        return np.linalg.solve(g + R + R2, b + R2 @ prior)
    Gs = G.reshape(2, NB, NBLK, F, F).sum(2); Bs = B.reshape(2, NB, NBLK, F, 20).sum(2)
    Ws = np.stack([[sol(Gs[c, q], Bs[c, q]) for q in range(NB)] for c in (0, 1)])   # (2,NB,F,20)
    if mode == "sext":
        return idx, Ws
    W = np.zeros((2, NB * NBLK, F, 20))
    for c in (0, 1):
        for q in range(NB):
            for b in range(NBLK):
                kk = q * NBLK + b
                W[c, kk] = sol(G[c, kk], B[c, kk], Ws[c, q], lam2)
    return idx, W

def predict_L(X, idx, W, batch=5, bins=None):
    """X numpy (n,20,H,W,2) -> L (n,20,H,W,2). Classes use PER-WINDOW sextiles (pw lib19); bins may be given."""
    Xt = torch.from_numpy(np.ascontiguousarray(X))
    n, T, H, Wd, _ = X.shape
    Wt = torch.from_numpy(W).float()
    nclass = Wt.shape[1]
    blk = block_id()
    out = np.zeros((n, 20, H, Wd, 2), np.float32)
    for i in range(0, n, 60):
        Xb = Xt[i:i+60]; nb_ = len(Xb)
        Fm, Emf, valid = feats(Xb)
        Fm = Fm[..., torch.from_numpy(idx)]
        bins_ = rank_bins_t(sigma_local_t(Xb), valid) if bins is None else bins[i:i+60]
        for c in (0, 1):
            cls = bins_[..., c].reshape(nb_, -1)
            if nclass > NB: cls = cls * NBLK + blk[None]
            Wg = Wt[c]                                            # (nclass,F,20)
            o = torch.zeros(nb_, H * Wd, 20)
            for q in range(nclass):
                msk = cls == q
                if msk.any(): o[msk] = Fm[msk] @ Wg[q]
            o = o.transpose(1, 2).reshape(nb_, 20, H, Wd) + Emf[..., c].unsqueeze(1)
            out[i:i+nb_, ..., c] = o.numpy()
    return out
