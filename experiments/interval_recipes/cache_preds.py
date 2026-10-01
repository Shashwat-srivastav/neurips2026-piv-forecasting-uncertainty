"""Cache (X, P, Y) for the 14 holdout trajectories with the shipped v8 FNO (no corrector in v8)."""
import json, os, sys, time
import h5py, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
V8 = os.path.join(HERE, "v8")
sys.path.insert(0, V8)
from load_baseline import load_baseline
ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], np.float32)
STD_IN = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], np.float32)
STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
split = json.load(open(os.path.join(BASE, "data/splits/holdout_v12_3.json")))
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 20
torch.set_num_threads(os.cpu_count())
model, _ = load_baseline(os.path.join(V8, "sim_real_fno_fp16.pth"), device="cpu")
model = model.eval()
mi, si, mt, st = (torch.from_numpy(z) for z in (MEAN_IN, STD_IN, MEAN_TGT, STD_TGT))
for role in ("select", "final"):
    X, Y, T = [], [], []
    for nm in split[role]:
        with h5py.File(os.path.join(ROOT, nm + ".h5")) as f:
            u = np.asarray(f["u"][:, ::2, ::2], np.float32); v = np.asarray(f["v"][:, ::2, ::2], np.float32)
        fr = np.stack([u, v, np.zeros_like(u)], -1)
        for i in range(0, len(fr) - 40 + 1, STRIDE):
            X.append(fr[i:i+20]); Y.append(fr[i+20:i+40]); T.append(nm)
    X = np.stack(X); Y = np.stack(Y)
    P = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(X), 5):
            xb = torch.nan_to_num(torch.from_numpy(X[i:i+5]))
            yb = model((xb - mi) / si) * st + mt
            yb[..., 2] = 0
            P.append(yb.numpy())
    P = np.concatenate(P)
    print(role, X.shape, f"{time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(os.path.join(HERE, f"cache_{role}_s{STRIDE}.npz"), X=X[..., :2], Y=Y[..., :2], P=P[..., :2], traj=np.array(T))
