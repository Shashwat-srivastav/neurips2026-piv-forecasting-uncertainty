"""v19 honest forecaster L: ridge local linear operator (experiments/v19_ridge_partner, trained on train_pool_v12_3 only).
L[t,p,c] = Emf[p,c] + sum_f feat[p,f] * W[block(p), f, c, sextile_c(p), t]
feat = last 10 frames' deviation from the window mean over a 5x5 nbhd of u,v (zero-padded),
       + (Emf[nbr] - Emf[ctr]) over the 5x5 nbhd (centre dropped, replicate-padded), + 1.
Classes: within-call sigma_local sextile (same bins as the v17 blend) x 4x4 spatial block."""
import os
import numpy as np
import torch
import torch.nn.functional as Fn

_d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ridge_w.npz"))
_W_NP = _d["W"].astype(np.float32)          # (16 blocks, F, 2*6*20)
_M, _K = int(_d["m"]), int(_d["k"])
_dev = {}


def _consts(dev):
    if dev not in _dev:
        _dev[dev] = (torch.from_numpy(_W_NP).to(dev),
                     torch.ones(2, 1, 3, 3, dtype=torch.float32, device=dev))
    return _dev[dev]


def _em_fill(X, ker):
    nz = X != 0
    cnt = nz.sum(1)
    valid = cnt > 0
    Em = X.sum(1) / cnt.clamp_min(1).to(X.dtype)
    e = Em.permute(0, 3, 1, 2); v = valid.permute(0, 3, 1, 2).to(X.dtype)
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
    D = torch.where(nz, X, Emf.unsqueeze(1)) - Emf.unsqueeze(1)
    return Emf, D


def ridge_L(xc, idx):
    """xc (b,Tin,H,W,2) input u,v; idx (b,H,W,2) long sextile bins. Returns L (b,20,H,W,2)."""
    W, ker = _consts(xc.device)
    n, T, H, Wd, _ = xc.shape
    Emf, D = _em_fill(xc, ker)
    p = _K // 2
    dfr = D[:, T - _M:].flip(1).permute(0, 1, 4, 2, 3).reshape(n, _M * 2, H, Wd)
    fd = Fn.unfold(Fn.pad(dfr, (p, p, p, p)), _K)
    e = Emf.permute(0, 3, 1, 2)
    fe = Fn.unfold(Fn.pad(e, (p, p, p, p), mode="replicate"), _K).reshape(n, 2, _K * _K, H * Wd)
    fe = fe - e.reshape(n, 2, 1, H * Wd)
    c0 = (_K * _K) // 2
    fe = torch.cat([fe[:, :, :c0], fe[:, :, c0 + 1:]], 2).reshape(n, 2 * (_K * _K - 1), H * Wd)
    one = torch.ones(n, 1, H * Wd, dtype=xc.dtype, device=xc.device)
    F = torch.cat([fd, fe, one], 1)                                   # (n,F,HW)
    nf = F.shape[1]
    F = F.reshape(n, nf, 4, H // 4, 4, Wd // 4).permute(2, 4, 0, 3, 5, 1).reshape(16, -1, nf)
    R = torch.bmm(F, W)                                               # (16, n*hb*wb, 240)
    R = R.reshape(4, 4, n, H // 4, Wd // 4, 2, 6, 20).permute(2, 0, 3, 1, 4, 5, 6, 7)
    R = R.reshape(n, H, Wd, 2, 6, 20)
    L = torch.gather(R, 4, idx.unsqueeze(-1).unsqueeze(-1).expand(n, H, Wd, 2, 1, 20)).squeeze(4)
    return L.permute(0, 4, 1, 2, 3) + Emf.unsqueeze(1)
