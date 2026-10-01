#!/usr/bin/env python3
"""WiSE-FT: weight-space interpolation between the sim-pretrained and the
sim->real fine-tuned FNO.

    theta(alpha) = (1 - alpha) * theta_sim  +  alpha * theta_sim_real_ft

WHY THIS IS A DIFFERENT AXIS FROM EVERYTHING THE CAMPAIGN HAS CLOSED.
Every previous attempt to improve the model either (a) re-fine-tuned on
train_real -- closed 4/4, and closed for a stated reason: the shipped checkpoint
is ALREADY organizer-fine-tuned to convergence on exactly that data -- or
(b) blended OUTPUTS of two models (FNO+CNO, FNO+Transolver), closed because both
partners lose to FNO on every metric simultaneously. This is neither. It moves
along the pretrain->finetune line in WEIGHT space, a direction never sampled.

MECHANISM. Wortsman et al. 2022 (WiSE-FT) show fine-tuning improves in-
distribution accuracy while DEGRADING robustness under distribution shift, and
that interpolating back toward the zero-shot weights recovers OOD accuracy
without giving up the in-distribution gain -- often beating BOTH endpoints OOD.
That is precisely this campaign's #1 open finding: the fine-tuned checkpoint
scores well on its own training distribution (local held-out rel 96.25) and
below every comparable trajectory on the live set (rel 94.17). Interpolation is
valid here because one checkpoint IS a fine-tune of the other, so they sit in the
same loss basin by construction -- the condition WiSE-FT requires.

COST: one model, so the 256 MB storage cap is untouched. No training. No extra
runtime. ONE scalar alpha, which can be bracketed live exactly the way the band
half-width already was.

alpha = 1.0 reproduces the shipped incumbent EXACTLY and is the correctness
check on this whole rig.
"""
import gc
import importlib.util
import os
import sys

import h5py
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'src', d) for d in ('training', 'evaluation', 'splits')]
KIT = os.path.join(REPO, "realpde_t1_starting_kit_v9", "realpde_t1_starting_kit_v9")
CKPT = os.path.join(os.environ.get("RPDE_T2_KIT_DIR", ""), "NeurIPS 2026 RealPDECompetition", "baseline_checkpoints")
SIM = os.path.join(CKPT, "sim_pretrain", "sim_fno_fp16.pth")
FT = os.path.join(CKPT, "sim_real_ft", "sim_real_fno_fp16.pth")
ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")

sys.path.insert(0, KIT)
sp = importlib.util.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(sc)
from load_baseline import load_baseline, load_checkpoint_state   # noqa: E402

HOLDOUT = ["21600_15", "25425_15", "22875_15"]
BATCH = 5
ALPHAS = [1.0, 0.95, 0.90, 0.80, 0.70, 0.50]

# Official normaliser (val_data/mean_std_real.pt), as the submission uses it.
MEAN = torch.tensor([0.1550, -0.0005, 0.0], dtype=torch.float32)
STD = torch.tensor([0.0968, 0.0160, 1.0], dtype=torch.float32)


def load_windows(names, stride=40):
    X, Y = [], []
    for nm in names:
        p = os.path.join(ROOT, nm + ".h5")
        with h5py.File(p) as f:
            u = np.asarray(f["u"][:, ::2, ::2], np.float32)
            v = np.asarray(f["v"][:, ::2, ::2], np.float32)
        fr = np.stack([u, v, np.zeros_like(u)], -1)
        for i in range(0, len(fr) - 40 + 1, stride):
            X.append(fr[i:i + 20])
            Y.append(fr[i + 20:i + 40])
    return np.stack(X), np.stack(Y)


print("loading the two checkpoints ...", flush=True)
model, meta = load_baseline(FT, device="cpu")
model = model.eval()
print(f"  ft model built: {meta}", flush=True)

sd_ft = load_checkpoint_state(FT)[0]
sd_sim = load_checkpoint_state(SIM)[0]
k_ft, k_sim = set(sd_ft), set(sd_sim)
print(f"  keys: ft {len(k_ft)}  sim {len(k_sim)}  shared {len(k_ft & k_sim)}")
if k_ft != k_sim:
    print(f"  ft-only  {sorted(k_ft - k_sim)[:6]}")
    print(f"  sim-only {sorted(k_sim - k_ft)[:6]}")
bad = [k for k in (k_ft & k_sim) if sd_ft[k].shape != sd_sim[k].shape]
print(f"  shape mismatches: {len(bad)}  {bad[:4]}")

# How far apart are they? If the fine-tune barely moved, interpolation is moot.
num = den = 0.0
for k in (k_ft & k_sim):
    a, b = sd_ft[k], sd_sim[k]
    if not (torch.is_tensor(a) and a.is_floating_point() or torch.is_complex(a)):
        continue
    d = (a - b).abs().float()
    num += float((d ** 2).sum())
    den += float((b.abs().float() ** 2).sum())
print(f"  ||theta_ft - theta_sim|| / ||theta_sim|| = {(num / den) ** 0.5:.5f}")

# Keep both endpoints in fp16 to halve peak memory; the checkpoints were fp16 on
# disk anyway, so this rounds away nothing that was ever there.
end_sim, end_ft = {}, {}
for k in (k_ft & k_sim):
    a, b = sd_ft[k], sd_sim[k]
    if torch.is_complex(a):
        end_ft[k] = torch.view_as_real(a).half()
        end_sim[k] = torch.view_as_real(b).half()
    elif torch.is_tensor(a) and a.is_floating_point():
        end_ft[k] = a.half()
        end_sim[k] = b.half()
    else:
        end_ft[k] = a
        end_sim[k] = b
del sd_ft, sd_sim
gc.collect()


# The view_as_real round-trip needs to know exactly which keys were complex.
complex_keys = {k for k in end_ft
                if torch.is_tensor(end_ft[k]) and end_ft[k].shape[-1:] == (2,)
                and end_ft[k].ndim >= 5}
print(f"  complex (view_as_real) keys: {len(complex_keys)}", flush=True)


def build_sd(alpha):
    sd = {}
    for k in end_ft:
        a, b = end_ft[k], end_sim[k]
        if torch.is_tensor(a) and a.is_floating_point():
            m = alpha * a.float() + (1.0 - alpha) * b.float()
            sd[k] = torch.view_as_complex(m.contiguous()) if k in complex_keys else m
        else:
            sd[k] = a
    return sd


def score(alpha, X, Y):
    model.load_state_dict(build_sd(alpha), strict=True)
    model.eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            chunk = X[i:i + BATCH]
            got = chunk.shape[0]
            if got < BATCH:
                chunk = np.concatenate(
                    [chunk, np.repeat(chunk[-1:], BATCH - got, axis=0)], axis=0)
            xb = torch.from_numpy(np.ascontiguousarray(chunk))
            xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
            yb = model((xb - MEAN) / STD) * STD + MEAN
            yb[..., 2] = 0.0
            P.append(yb.float().numpy()[:got])
    P = np.concatenate(P)
    rel = sc.score_error(float(sc.rel_l2_per_sample(P, Y, 2).mean()))
    tke = sc.score_error(float(sc.tke_rel_l2_per_sample(P, Y, 2).mean()))
    mvpe = sc.score_error(float(sc.mvpe_rel_l2_per_sample(P, Y).mean()))
    del P
    gc.collect()
    return rel, tke, mvpe


SPS_W = dict(rel=0.5, tke=0.3, mvpe=0.2)
W = dict(rel=0.476, tke=0.086, mvpe=0.095, sps=0.243)
T_LIVE = 0.491437          # live band term, from gap_decomposition.py


def wbar(rel, tke, mvpe):
    def omp(s):
        e = 200.0 / s - 2.0
        return 0.5 / (0.5 + e)
    return SPS_W["rel"]*omp(rel) + SPS_W["tke"]*omp(tke) + SPS_W["mvpe"]*omp(mvpe)


print("\nloading holdout windows ...", flush=True)
X, Y = load_windows(HOLDOUT)
print(f"windows: {len(X)}  ({len(HOLDOUT)} held-out trajectories)\n", flush=True)

print(f"{'alpha':>7}{'rel':>10}{'tke':>10}{'mvpe':>10}{'sps*':>10}{'composite*':>12}")
print("-" * 59)
base = None
for a in ALPHAS:
    rel, tke, mvpe = score(a, X, Y)
    sps = 100 * wbar(rel, tke, mvpe) * T_LIVE
    comp = W["rel"]*rel + W["tke"]*tke + W["mvpe"]*mvpe + W["sps"]*sps
    if base is None:
        base = comp
    tag = "  <- incumbent" if a == 1.0 else f"  {comp - base:+.4f}"
    print(f"{a:7.2f}{rel:10.4f}{tke:10.4f}{mvpe:10.4f}{sps:10.4f}{comp:12.4f}{tag}",
          flush=True)
print("\n* sps and composite use the LIVE band term T and exclude time --")
print("  they rank candidates on the accuracy channel only, they are not")
print("  predicted live scores.")
