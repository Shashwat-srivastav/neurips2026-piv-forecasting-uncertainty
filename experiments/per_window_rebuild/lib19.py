"""v19: small honest forecaster L (ridge local linear operator) trained on train_pool_v12_3 only."""
import os, sys, json
sys.dont_write_bytecode = True
import numpy as np, torch, h5py
import torch.nn.functional as Fn
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_KIT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
MMAX, KMAX = 10, 5
NB = 6

def train_names():
    names = json.load(open(os.path.join(ROOT_KIT, "data/splits/train_pool_v12_3.json")))["pool"]
    ho = json.load(open(os.path.join(ROOT_KIT, "data/splits/holdout_v12_3.json")))["holdout"]
    names = [n for n in names if n not in ho and n != "7575_0"]
    return names

def load_traj(nm):
    with h5py.File(os.path.join(DATA, nm + ".h5")) as f:
        u = np.asarray(f["u"][:, ::2, ::2], np.float32)
        v = np.asarray(f["v"][:, ::2, ::2], np.float32)
    fr = np.stack([u, v], -1)
    return np.nan_to_num(fr, nan=0.0, posinf=0.0, neginf=0.0)

def windows(fr, stride):
    idx = np.arange(0, len(fr) - 40 + 1, stride)
    X = np.stack([fr[i:i+20] for i in idx]); Y = np.stack([fr[i+20:i+40] for i in idx])
    return X, Y

def em_fill(X):
    """X torch (n,T,H,W,2). Returns Emf (n,H,W,2) filled at invalid pixels, valid (n,H,W,2), Xf-deviation D (n,T,H,W,2)."""
    nz = X != 0
    cnt = nz.sum(1)
    valid = cnt > 0
    Em = X.sum(1) / cnt.clamp_min(1).to(X.dtype)
    e = Em.permute(0, 3, 1, 2); v = valid.permute(0, 3, 1, 2).to(X.dtype)
    ker = torch.ones(2, 1, 3, 3, dtype=X.dtype)
    ef, vf = e * v, v.clone()
    for _ in range(4):
        num = Fn.conv2d(Fn.pad(ef, (1, 1, 1, 1), mode="replicate"), ker, groups=2)
        den = Fn.conv2d(Fn.pad(vf, (1, 1, 1, 1), mode="replicate"), ker, groups=2)
        newv = (den > 0).to(X.dtype)
        ef = torch.where(vf > 0, ef, num / den.clamp_min(1e-6) * newv)
        vf = torch.maximum(vf, newv)
    gm = (ef * vf).sum((2, 3), keepdim=True) / vf.sum((2, 3), keepdim=True).clamp_min(1)
    ef = torch.where(vf > 0, ef, gm.expand_as(ef))
    Emf = ef.permute(0, 2, 3, 1).contiguous()
    Xf = torch.where(nz, X, Emf.unsqueeze(1))
    D = Xf - Emf.unsqueeze(1)
    return Emf, valid, D

def feats(X, m=MMAX, k=KMAX):
    """X torch (n,20,H,W,2) -> (n, H*W, F) features, Emf, valid.
    Feature order: for lag j (0=last frame) in range(m), channel c, offset (dy,dx) row-major: D[-1-j]
                   then Em[nbr]-Em[ctr] for c, offsets except centre; then const 1."""
    n, T, H, W, _ = X.shape
    Emf, valid, D = em_fill(X)
    p = k // 2
    dfr = D[:, T-m:].flip(1).permute(0, 1, 4, 2, 3).reshape(n, m * 2, H, W)   # lag-major
    fd = Fn.unfold(Fn.pad(dfr, (p, p, p, p)), k)                              # (n, m*2*k*k, HW)
    e = Emf.permute(0, 3, 1, 2)
    fe = Fn.unfold(Fn.pad(e, (p, p, p, p), mode="replicate"), k).reshape(n, 2, k * k, H * W)
    fe = fe - e.reshape(n, 2, 1, H * W)
    keep = [i for i in range(k * k) if i != (k * k) // 2]
    fe = fe[:, :, keep].reshape(n, 2 * (k * k - 1), H * W)
    one = torch.ones(n, 1, H * W, dtype=X.dtype)
    Fm = torch.cat([fd, fe, one], 1).transpose(1, 2).contiguous()
    return Fm, Emf, valid

def feat_index(m, k, M=MMAX, K=KMAX):
    """indices into the (MMAX,KMAX) feature vector for a sub-config (m,k)."""
    P, p = K // 2, k // 2
    offs = [(dy, dx) for dy in range(-P, P + 1) for dx in range(-P, P + 1)]
    sel_off = [i for i, (dy, dx) in enumerate(offs) if abs(dy) <= p and abs(dx) <= p]
    idx = []
    for j in range(m):
        for c in range(2):
            idx += [(j * 2 + c) * K * K + i for i in sel_off]
    base = M * 2 * K * K
    keep = [i for i in range(K * K) if i != (K * K) // 2]
    for c in range(2):
        for q, i in enumerate(keep):
            if i in sel_off:
                idx.append(base + c * (K * K - 1) + q)
    idx.append(base + 2 * (K * K - 1))
    return np.array(idx)

def sigma_local_t(X):
    fl = X - X.mean(1, keepdim=True)
    return (fl * fl).mean(1).sqrt()

def rank_bins_t(F, valid, nb=NB, batch=1):   # PER-WINDOW (was batch=5)
    out = torch.zeros(F.shape, dtype=torch.long)
    qs = torch.linspace(1.0 / nb, 1.0 - 1.0 / nb, nb - 1)
    for i in range(0, len(F), batch):
        for c in (0, 1):
            f = F[i:i+batch, ..., c]; v = valid[i:i+batch, ..., c]
            if v.sum() < nb: continue
            edges = torch.quantile(f[v].float(), qs)
            out[i:i+batch, ..., c] = torch.bucketize(f.contiguous(), edges.contiguous()).clamp(0, nb - 1)
    return out

def block_id(H=32, W=64, by=4, bx=4):
    yy = torch.arange(H) * by // H; xx = torch.arange(W) * bx // W
    return (yy[:, None] * bx + xx[None, :]).reshape(-1)
