"""Official-FNO forecasts for every usable real trajectory (stride-20 windows), one file per trajectory:
cache_all/<name>.npy holds P (n, 20, 32, 64, 2). Inputs and targets are re-read from train_real by loo_lib.load_traj.
Resumable: existing files are skipped. The duplicate 7575_0 is excluded.
"""
import os, sys, time
import h5py, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "interval_recipes", "v8"))
from load_baseline import load_baseline   # noqa: E402
from loo_lib import ROOT, OUT, names, windows   # noqa: E402

mi, si = (torch.tensor(z) for z in ([0.154960856, -0.000513992854, 0.0], [0.0968056545, 0.015960684, 1.0]))
mt, st = (torch.tensor(z) for z in ([0.154962569, -0.000517793698, 0.0], [0.0968104079, 0.0159636438, 1.0]))
torch.set_num_threads(6)
# official FNO (the exp1 builds hold our Run 1 weights under the same file name)
model, _ = load_baseline(os.path.join(HERE, "..", "..", "submissions", "v17_per_window", "sim_real_fno_fp16.pth"), device="cpu")
model = model.eval()
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
todo = [n for n in names() if not os.path.exists(os.path.join(OUT, n + ".npy"))]
print(len(todo), "trajectories to cache", flush=True)
for k, nm in enumerate(todo):
    X, _ = windows(nm, 20, channels=3)
    P = np.zeros(X[..., :2].shape, np.float32)
    with torch.no_grad():
        for i in range(0, len(X), 5):
            xb = torch.nan_to_num(torch.from_numpy(X[i:i+5]))
            P[i:i+5] = (model((xb - mi) / si) * st + mt).float().numpy()[..., :2]
    np.save(os.path.join(OUT, nm + ".npy"), P)
    print(f"{k + 1}/{len(todo)} {nm} n={len(X)} {time.time() - t0:.0f}s", flush=True)
print("done")
