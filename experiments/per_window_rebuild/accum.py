"""Accumulate ridge sufficient stats on train_pool (holdout + 7575_0 excluded). usage: accum.py STRIDE"""
import sys, time
from lib19 import *
torch.set_num_threads(12)
S = int(sys.argv[1]) if len(sys.argv) > 1 else 4  # pw: per-window bins via lib19.rank_bins_t batch=1
names = train_names()
NBLK = 16; blk = block_id()
Fdim = len(feat_index(MMAX, KMAX)); NC = NB * NBLK
G = torch.zeros(2, NC, Fdim, Fdim, dtype=torch.float64)
B = torch.zeros(2, NC, Fdim, 20, dtype=torch.float64)
YY = torch.zeros(2, NC, 20, dtype=torch.float64)
N = torch.zeros(2, NC, dtype=torch.float64)
t0 = time.time(); nw = 0
for ti, nm in enumerate(names):
    fr = load_traj(nm)
    X, Y = windows(fr, S)
    for i in range(0, len(X), 60):
        Xb = torch.from_numpy(X[i:i+60]); Yb = torch.from_numpy(Y[i:i+60])
        Fm, Emf, valid = feats(Xb)
        bins = rank_bins_t(sigma_local_t(Xb), valid)
        n = len(Xb); nw += n
        R = (Yb - Emf.unsqueeze(1))                       # (n,20,H,W,2)
        for c in (0, 1):
            m = ((Yb[..., c] != 0).all(1) & valid[..., c]).reshape(n, -1)   # rows with all 20 targets measured
            cls = (bins[..., c].reshape(n, -1) * NBLK + blk[None]).masked_fill(~m, -1)
            fl = Fm.reshape(-1, Fdim); rl = R[..., c].reshape(n, 20, -1).transpose(1, 2).reshape(-1, 20)
            cl = cls.reshape(-1)
            order = torch.argsort(cl); cl_s = cl[order]
            cnts = torch.bincount(cl_s[cl_s >= 0], minlength=NC)
            start = int((cl_s < 0).sum())
            for k in range(NC):
                e = start + int(cnts[k])
                if e > start:
                    ix = order[start:e]; f = fl[ix]; r = rl[ix]
                    G[c, k] += (f.T @ f).double(); B[c, k] += (f.T @ r).double()
                    YY[c, k] += (r * r).sum(0).double(); N[c, k] += e - start
                start = e
    print(f"{ti+1}/{len(names)} {nm} windows {nw} t {time.time()-t0:.0f}s", flush=True)
np.savez(f"stats_pw_s{S}.npz", G=G.numpy(), B=B.numpy(), YY=YY.numpy(), N=N.numpy())
print("done", time.time() - t0)
