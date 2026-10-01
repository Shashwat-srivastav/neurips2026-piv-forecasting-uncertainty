"""Stats per (sextile? x spatial block) for feature set (m,k). usage: accum2.py STRIDE m k BY BX USE_SEXT(0/1) OUT"""
import sys, time
from lib19 import *
torch.set_num_threads(10)
S, m, k, by, bx, us = map(int, sys.argv[1:7]); out = sys.argv[7]
names = train_names()
NBLK = by * bx; blk = block_id(32, 64, by, bx)
NS = NB if us else 1
Fdim = m * 2 * k * k + 2 * (k * k - 1) + 1; NC = NS * NBLK
G = torch.zeros(2, NC, Fdim, Fdim, dtype=torch.float64)
B = torch.zeros(2, NC, Fdim, 20, dtype=torch.float64)
N = torch.zeros(2, NC, dtype=torch.float64)
t0 = time.time()
for ti, nm in enumerate(names):
    X, Y = windows(load_traj(nm), S)
    for i in range(0, len(X), 60):
        Xb = torch.from_numpy(X[i:i+60]); Yb = torch.from_numpy(Y[i:i+60]); n = len(Xb)
        Fm, Emf, valid = feats(Xb, m, k)
        bins = rank_bins_t(sigma_local_t(Xb), valid) if us else torch.zeros(valid.shape, dtype=torch.long)
        R = Yb - Emf.unsqueeze(1)
        for c in (0, 1):
            msk = ((Yb[..., c] != 0).all(1) & valid[..., c]).reshape(n, -1)
            cls = (bins[..., c].reshape(n, -1) * NBLK + blk[None])
            cls = torch.where(msk, cls, torch.full_like(cls, NC))
            Fw = Fm * msk.unsqueeze(-1)
            rl = R[..., c].reshape(n, 20, -1).transpose(1, 2)            # (n,HW,20)
            fl = Fw.reshape(-1, Fdim); r2 = rl.reshape(-1, 20)
            # G[c,q] = sum_rows oh[:,q] f f^T  -> via sorting for efficiency
            cl = cls.reshape(-1); order = torch.argsort(cl); cs = cl[order]
            cnt = torch.bincount(cs, minlength=NC + 1)
            st_ = 0
            for q in range(NC):
                e = st_ + int(cnt[q])
                if e > st_:
                    ix = order[st_:e]; f = fl[ix]
                    G[c, q] += (f.T @ f).double(); B[c, q] += (f.T @ r2[ix]).double(); N[c, q] += e - st_
                st_ = e
    if ti % 10 == 0: print(f"{ti+1}/{len(names)} t {time.time()-t0:.0f}s", flush=True)
np.savez(out, G=G.numpy(), B=B.numpy(), N=N.numpy(), m=m, k=k, by=by, bx=bx, us=us)
print("done", time.time() - t0)
