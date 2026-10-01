"""Cache a model's raw forecasts (u,v) on SELECT/FINAL cache windows -> experiments/clean_fno_regimes/P_<tag>_<role>.npy (float32, memmap write).
usage: python experiments/clean_fno_regimes/cache_model.py <ckpt> <tag> <outdir> [builddir]"""
import os, sys, gc, time, numpy as np, torch
ck, tag, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
bd = sys.argv[4] if len(sys.argv) > 4 else "submissions/build_exp1"
sys.path.insert(0, bd)
from load_baseline import load_baseline
torch.set_num_threads(os.cpu_count())
m, meta = load_baseline(ck, device="cpu"); m = m.float().eval(); gc.collect()
print("loaded", ck, meta if not isinstance(meta, dict) else list(meta)[:5], flush=True)
MI = torch.tensor([0.154960856, -0.000513992854, 0.0]); SI = torch.tensor([0.0968056545, 0.015960684, 1.0])
MT = torch.tensor([0.154962569, -0.000517793698, 0.0]); ST = torch.tensor([0.0968104079, 0.0159636438, 1.0])
for role in ("select", "final"):
    X = np.load(f"experiments/interval_recipes/cache_{role}_s20.npz")["X"]
    out = np.lib.format.open_memmap(os.path.join(outdir, f"P_{tag}_{role}.npy"), "w+", np.float32, X.shape)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(X), 5):
            xb = torch.from_numpy(np.concatenate([X[i:i+5], np.zeros(X[i:i+5].shape[:-1] + (1,), np.float32)], -1))
            yb = m((torch.nan_to_num(xb) - MI) / SI) * ST + MT
            out[i:i+5] = yb[..., :2].numpy()
    out.flush(); del out, X; gc.collect()
    print(role, "done", round(time.time() - t0, 1), "s", flush=True)
