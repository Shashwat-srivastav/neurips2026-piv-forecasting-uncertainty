"""Load all train_real trajectories at the 2x-subsampled scoring grid into one npz cache (experiments/piv_noise only)."""
import os, h5py, numpy as np
ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_s2.npz")
U, V, names, aoa, re_, lens = [], [], [], [], [], []
xg = yg = None
for fn in sorted(os.listdir(ROOT)):
    if not fn.endswith(".h5") or fn.startswith("7575_0"): continue
    with h5py.File(os.path.join(ROOT, fn), "r") as f:
        U.append(np.asarray(f["u"][:, ::2, ::2], np.float32)); V.append(np.asarray(f["v"][:, ::2, ::2], np.float32))
        aoa.append(int(f["aoa"][()])); re_.append(int(f["re"][()]))
        if xg is None: xg = f["x"][::2, ::2]; yg = f["y"][::2, ::2]
    names.append(fn[:-3]); lens.append(len(U[-1]))
np.savez(OUT, u=np.concatenate(U), v=np.concatenate(V), names=np.array(names), aoa=np.array(aoa), re=np.array(re_), lens=np.array(lens), x=xg, y=yg)
print(len(names), sum(lens), set(aoa))
