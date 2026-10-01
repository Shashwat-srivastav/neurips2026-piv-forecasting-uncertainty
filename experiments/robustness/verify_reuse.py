"""Check that a reused forecast (from the SELECT/FINAL caches) equals a fresh official-FNO forecast of the same windows."""
import os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "interval_recipes", "v8"))
from load_baseline import load_baseline   # noqa: E402
from loo_lib import OUT, windows   # noqa: E402

nm = "20325_15"   # a FINAL trajectory
X, _ = windows(nm, 20, channels=3); X = X[:5]
mi, si = (torch.tensor(z) for z in ([0.154960856, -0.000513992854, 0.0], [0.0968056545, 0.015960684, 1.0]))
mt, st = (torch.tensor(z) for z in ([0.154962569, -0.000517793698, 0.0], [0.0968104079, 0.0159636438, 1.0]))
model, _ = load_baseline(os.path.join(HERE, "..", "..", "submissions", "v17_per_window", "sim_real_fno_fp16.pth"), device="cpu")
with torch.no_grad():
    fresh = (model.eval()((torch.nan_to_num(torch.from_numpy(X)) - mi) / si) * st + mt).float().numpy()[..., :2]
reused = np.load(os.path.join(OUT, nm + ".npy"))[:5]
d = float(np.abs(fresh - reused).max())
print(f"verify {nm}: max |fresh - reused| = {d:.2e}", "OK" if d < 1e-3 else "MISMATCH")
sys.exit(0 if d < 1e-3 else 1)
