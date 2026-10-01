"""Predictions of the fp16-packed fine-tuned FNO on the SELECT/FINAL caches (same code path as experiments/interval_recipes/cache_preds.py)."""
import os, sys, time
sys.dont_write_bytecode = True
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "submissions", "v17"))
from load_baseline import load_baseline
MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], np.float32)
STD_IN = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], np.float32)
STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
torch.set_num_threads(3)
model, meta = load_baseline(os.path.join(HERE, "..", "build_v18ft", "fno_finetuned_fp16.pth"), device="cpu"); model.eval(); print(meta)
mi, si, mt, st = (torch.from_numpy(z) for z in (MEAN_IN, STD_IN, MEAN_TGT, STD_TGT))
for role in ("select", "final"):
    X = np.load(os.path.join(HERE, "..", "interval_recipes", f"cache_{role}_s20.npz"))["X"]
    x3 = np.concatenate([X, np.zeros(X.shape[:-1] + (1,), np.float32)], -1)
    P = []; t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(X), 5):
            xb = torch.nan_to_num(torch.from_numpy(x3[i:i+5]))
            P.append((model((xb - mi) / si) * st + mt)[..., :2].numpy())
    np.save(os.path.join(HERE, f"Pft_{role}.npy"), np.concatenate(P).astype(np.float32)); print(role, time.time() - t0, flush=True)
