"""A2: cache (X,Y,P) stride 20 for 4 train-pool trajectories chosen at random (seed 0), stratified by AoA
(4 of the 5 AoA strata drawn at random, one trajectory per stratum). No flow fingerprint used."""
import json, os, sys, time
import h5py, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
BANDS = os.path.abspath(os.path.join(HERE, "..", "interval_recipes")); BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
V8 = os.path.join(BANDS, "v8"); sys.path.insert(0, V8)
from load_baseline import load_baseline
ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], np.float32); STD_IN = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], np.float32); STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
pool = json.load(open(os.path.join(BASE, "data/splits/train_pool_v12_3.json")))["pool"]
rng = np.random.default_rng(0)
aoas = sorted({int(n.split("_")[1]) for n in pool})
strata = sorted(rng.choice(aoas, 4, replace=False).tolist())
pick = [str(rng.choice(sorted(n for n in pool if int(n.split("_")[1]) == a))) for a in strata]
print("AoA strata", strata, "picked", pick, flush=True)
json.dump(dict(seed=0, strata=strata, rand=pick), open(os.path.join(HERE, "rand_split.json"), "w"))
torch.set_num_threads(6)
model, _ = load_baseline(os.path.join(V8, "sim_real_fno_fp16.pth"), device="cpu"); model = model.eval()
mi, si, mt, st = (torch.from_numpy(z) for z in (MEAN_IN, STD_IN, MEAN_TGT, STD_TGT))
idx = []
for nm in pick:
    with h5py.File(os.path.join(ROOT, nm + ".h5")) as f: n = f["u"].shape[0]
    idx += [(nm, i) for i in range(0, n - 40 + 1, 20)]
X = np.zeros((len(idx), 20, 32, 64, 3), np.float32); Y = np.zeros_like(X); T = [a for a, _ in idx]; j = 0
for nm in pick:
    with h5py.File(os.path.join(ROOT, nm + ".h5")) as f:
        u = np.asarray(f["u"][:, ::2, ::2], np.float32); v = np.asarray(f["v"][:, ::2, ::2], np.float32)
    for i in range(0, len(u) - 40 + 1, 20):
        X[j, ..., 0] = u[i:i+20]; X[j, ..., 1] = v[i:i+20]; Y[j, ..., 0] = u[i+20:i+40]; Y[j, ..., 1] = v[i+20:i+40]; j += 1
    del u, v
print("windows", X.shape, flush=True)
PF = os.path.join(HERE, "rand_P_partial.npy")
P = np.load(PF) if os.path.exists(PF) else np.full(X[..., :2].shape, np.nan, np.float32); t0 = time.time()
with torch.no_grad():
    for i in range(0, len(X), 5):
        if np.isfinite(P[i:i+5]).all(): continue
        xb = torch.nan_to_num(torch.from_numpy(X[i:i+5]))
        yb = model((xb - mi) / si) * st + mt; yb[..., 2] = 0; P[i:i+5] = yb.numpy()[..., :2]; np.save(PF, P)
        if i % 100 == 0: print(i, f"{time.time()-t0:.0f}s", flush=True)
print("done", f"{time.time()-t0:.0f}s", flush=True)
np.savez(os.path.join(HERE, "cache_rand_s20.npz"), X=np.ascontiguousarray(X[..., :2]), Y=np.ascontiguousarray(Y[..., :2]), P=P, traj=np.array(T)); print("saved", flush=True)
