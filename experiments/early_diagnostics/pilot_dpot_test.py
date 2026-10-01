"""Quick sanity + timing + accuracy pilot for DPOT-S, mirroring what was done
for UNet: (1) does it load without error, (2) does a forward pass produce
sane-scale output, (3) how slow is it on CPU, (4) real accuracy on a handful
of holdout windows using the SAME official scorer.

DPOT is imported directly from the cloned RealPDEBench repo (no torchvision-
style broken-dependency issue like UNet had -- only numpy/torch/einops,
verified 29 Aug), so no standalone trimmed copy needed this time.
"""
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

REALPDEBENCH = os.environ.get("REALPDEBENCH_DIR", "")
sys.path.insert(0, REALPDEBENCH)
from realpdebench.model.dpot import DPOT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'src', d) for d in ('training', 'evaluation', 'splits')]
KIT = os.path.join(REPO, "realpde_t1_starting_kit_v9", "realpde_t1_starting_kit_v9")
sys.path.insert(0, KIT)
import importlib.util as _iu
spec = _iu.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = _iu.module_from_spec(spec); spec.loader.exec_module(sc)

CKPT = os.path.join(os.environ.get("RPDEBENCH_CKPTS", ""), "foil", "dpot_s", "finetune.pth")
ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")

# BUG FOUND 29 Aug: dpot_s.yaml has normalize=False, meaning DPOT's own
# wrapper does NOT normalize internally -- exactly like FNO/CNO, the caller
# is expected to Gaussian-normalize before calling forward() and denormalize
# after. The first pilot run skipped this entirely (fed raw physical-scale
# values straight in) and got a catastrophic, wrong-signed result (rel 64.0,
# pred u mean -0.054 vs target +0.197) that looked like a real architecture
# failure but was actually this bug. Same normalizer constants as every other
# model in this campaign (verified against scoring.py to 1e-7).
MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], np.float32)
STD_IN = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], np.float32)
STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
_mi, _si, _mt, _st = (torch.from_numpy(z) for z in (MEAN_IN, STD_IN, MEAN_TGT, STD_TGT))


def dpot_predict(xb):
    """Normalize -> DPOT -> denormalize, matching every other model's convention."""
    with torch.no_grad():
        yb = model((xb - _mi) / _si) * _st + _mt
    return yb


torch.set_num_threads(8)

print("building DPOT-S wrapper (shape_in/out = (20,32,64,3), dpot_s.yaml params)...",
      flush=True)
model = DPOT(
    shape_in=(20, 32, 64, 3), shape_out=(20, 32, 64, 3),
    img_size=128, patch_size=8, in_channels=4, out_channels=4,
    embed_dim=1024, depth=6, n_blocks=8, modes=32, mlp_ratio=1,
    out_layer_dim=32, normalize=False, act="gelu", time_agg="exp_mlp",
    n_cls=12, in_timesteps=20, out_timesteps=20,
    model_type="dpot", checkpoint_path=CKPT,
)
model.eval()
n = sum(p.numel() for p in model.parameters())
print(f"loaded. total params {n:,} ({n*4/1e6:.1f} MB fp32)", flush=True)

# --- sanity: sane-scale forward pass on real data ---
with h5py.File(ROOT + "/20325_15.h5") as f:
    u = np.asarray(f["u"][:, ::2, ::2], np.float32)
    v = np.asarray(f["v"][:, ::2, ::2], np.float32)
fr = np.stack([u, v, np.zeros_like(u)], -1)
X = fr[0:20][None]
Y = fr[20:40][None]
xb = torch.from_numpy(X)

t0 = time.perf_counter()
yb = dpot_predict(xb)
yb[..., 2] = 0.0
dt = time.perf_counter() - t0
print(f"\nforward pass (1 sample): {dt:.2f}s", flush=True)
P = yb.numpy()
print(f"pred u mean/std: {P[...,0].mean():.4f}/{P[...,0].std():.4f}   "
      f"target u mean/std: {Y[...,0].mean():.4f}/{Y[...,0].std():.4f}")
print(f"pred v mean/std: {P[...,1].mean():.4f}/{P[...,1].std():.4f}   "
      f"target v mean/std: {Y[...,1].mean():.4f}/{Y[...,1].std():.4f}")
crude_err = np.linalg.norm(P[..., :2] - Y[..., :2]) / np.linalg.norm(Y[..., :2])
print(f"crude rel err (u,v): {crude_err:.4f}", flush=True)

# --- batch=5 timing ---
xb5 = torch.randn(5, 20, 32, 64, 3)
dpot_predict(xb5)  # warmup
t0 = time.perf_counter()
for _ in range(3):
    dpot_predict(xb5)
dt5 = (time.perf_counter() - t0) / 3
print(f"\nbatch=5 CPU: {dt5*1000:.1f} ms/batch, {dt5/5*1000:.1f} ms/sample", flush=True)

# --- real accuracy on the SELECT pool (168 windows, stride=20) ---
split = json.load(open(os.path.join(REPO, "data/splits/holdout_v12_3.json")))
names = split["select"]


def load_windows(names, stride=20):
    X, Y = [], []
    for nm in names:
        with h5py.File(os.path.join(ROOT, nm + ".h5")) as f:
            u = np.asarray(f["u"][:, ::2, ::2], np.float32)
            v = np.asarray(f["v"][:, ::2, ::2], np.float32)
        fr = np.stack([u, v, np.zeros_like(u)], -1)
        for i in range(0, len(fr) - 40 + 1, stride):
            X.append(fr[i:i + 20]); Y.append(fr[i + 20:i + 40])
    return np.stack(X), np.stack(Y)


Xs, Ys = load_windows(names)
print(f"\nscoring on SELECT pool: {len(Xs)} windows", flush=True)
P = []
t0 = time.perf_counter()
with torch.no_grad():
    for i in range(0, len(Xs), 5):
        chunk = Xs[i:i + 5]
        got = chunk.shape[0]
        if got < 5:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], 5 - got, axis=0)], axis=0)
        xb = torch.from_numpy(np.ascontiguousarray(chunk))
        yb = dpot_predict(xb)
        yb[..., 2] = 0.0
        P.append(yb.numpy()[:got])
        if i % 20 == 0:
            print(f"  {i}/{len(Xs)}  {time.perf_counter()-t0:.0f}s", flush=True)
P = np.concatenate(P)
rel = sc.score_error(float(sc.rel_l2_per_sample(P, Ys, 2).mean()))
tke = sc.score_error(float(sc.tke_rel_l2_per_sample(P, Ys, 2).mean()))
mvpe = sc.score_error(float(sc.mvpe_rel_l2_per_sample(P, Ys).mean()))
print(f"\n=== DPOT-S off-the-shelf on SELECT: rel {rel:.3f}  tke {tke:.3f}  mvpe {mvpe:.3f} ===")
print("compare: stock FNO on SELECT = rel 96.088  tke 79.915  mvpe 97.379")
print("compare: frozen sim_cno on SELECT = rel 77.531  tke 67.077  mvpe 73.807")
