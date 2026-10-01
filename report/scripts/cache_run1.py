"""Cache the clean sim->real FNO (Run 1) forecasts on SELECT/FINAL for the report figures (float16)."""
import os, sys, gc, numpy as np, torch
sys.path.insert(0, "submissions/build_exp1")
from load_baseline import load_baseline
m, _ = load_baseline("submissions/build_exp1/sim_real_fno_fp16.pth", device="cpu"); m = m.float().eval(); torch.set_num_threads(os.cpu_count())
MI = torch.tensor([0.154960856, -0.000513992854, 0.0]); SI = torch.tensor([0.0968056545, 0.015960684, 1.0])
MT = torch.tensor([0.154962569, -0.000517793698, 0.0]); ST = torch.tensor([0.0968104079, 0.0159636438, 1.0])
out = {}
for role in ("select", "final"):
    X = np.load(f"experiments/interval_recipes/cache_{role}_s20.npz")["X"]
    P = []
    with torch.no_grad():
        for i in range(0, len(X), 5):
            x3 = np.concatenate([X[i:i+5], np.zeros(X[i:i+5].shape[:-1] + (1,), np.float32)], -1)
            P.append((m((torch.from_numpy(x3) - MI) / SI) * ST + MT)[..., :2].numpy().astype(np.float16))
    out[role] = np.concatenate(P); del X, P; gc.collect(); print(role, out[role].shape, flush=True)
np.savez_compressed("report/data/run1_preds.npz", select=out["select"], final=out["final"])
print("saved")
