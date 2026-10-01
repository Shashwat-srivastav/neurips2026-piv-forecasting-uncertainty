#!/usr/bin/env python3
"""Fine-tune the RealPDE FNO baseline on the evaluation regime. For Kaggle GPU.

WHY THIS EXISTS
---------------
The stock `sim_real_ft` FNO scores, live, rel_l2 94.168 / tke 74.026 / mvpe
92.836. The "96.111/79.546/97.556 locally" this line used to claim was NOT a
holdout number -- sim_real_ft is fine-tuned by the organizers on ALL 81 usable
train_real trajectories, so every local figure measured before 22 Aug was a
training score (CAMPAIGN_REPORT.md 12.3). The real holdout, finally run 22 Aug
(14 trajectories nearest the live fingerprint, bracketed on both sides so the
test is interpolation like the live task, never touched by this checkpoint's
training): rel 96.296 / tke 80.715 / mvpe 97.154 (588 windows) -- gap vs live
+2.13 / +6.69 / +4.32. That gap is real and it is a generalization gap, not
architecture: it is what THIS SAME FROZEN MODEL already does on data it has
never seen. Report 14.3 shows it composites to ~81.2-81.3, inside the current
leaderboard's top cluster. GPU inference is 3.19 ms/sample against a 62 ms
break-even, so there is room for a bigger or better-fitted model.

Train from `sim_fno.pth` (sim-pretrained ONLY), not `sim_real_fno.pth` --
the latter has already seen all 81 real trajectories including whatever you
hold out, so it is not a clean starting point for measuring generalization.
Use `--pool-file train_pool_v12_3.json` (the 67 non-holdout trajectories).

WHAT THE EVAL SET IS (measured from the ingestion log, 7 Aug 2026)
------------------------------------------------------------------
A SINGLE flow condition: AoA 15, Reynolds 20,600-25,200. Established from
per-process statistics -- u_mean varies 2.2% and u_std 1.6% across worker
processes while v_std varies 11.9% and tau 57.2%, and Reynolds governs the
streamwise channel at rho 0.98 while AoA governs the transverse one.

So training is weighted toward AoA 15 at Re 19,090-26,761, but NOT restricted to
it: the regime is inferred, and the Decision Phase uses unseen Re/AoA.

THE LOSS IS NOT MSE, AND THAT IS THE POINT
------------------------------------------
Plain MSE regresses to the conditional mean, which systematically UNDER-predicts
temporal variance -- the stock FNO carries only 56% of the target's per-pixel
variance, which is exactly why its tke is weak. Training here optimises the
three scored quantities directly:

    rel_l2  : ||p - t|| / ||t||                     per sample
    tke     : rel_l2 between per-pixel temporal-variance maps
    mvpe    : rel_l2 at the 36 graded probe points

Default weights come from each metric's score sensitivity at the current
operating point -- d(score)/d(err) = -50/(1+0.5e)^2 -- so a unit of rel_l2 error
is worth about 2.3x a unit of tke error. Override with --w-rel etc.

DATA
----
Upload `train_real` (82 .h5 files, ~7 GB) as a Kaggle dataset, or a subset --
--subset-regime trains on the ~25 trajectories that matter and needs ~2 GB.

Each file: top-level `u`, `v` of shape (T, 64, 128) float64, plus scalar `re`
and `aoa`. READ Re FROM THE `re` ATTRIBUTE, NEVER THE FILENAME: 78 of 82 files
disagree, the true step is 1278-1279 rather than 1275, and `6300_0`/`7575_0`
are byte-identical duplicates (7575_0 is excluded below).

USAGE
-----
    python finetune_fno_kaggle.py \
        --data /kaggle/input/train-real/train_real \
        --ckpt /kaggle/input/realpde-ckpt/sim_real_fno.pth \
        --kit  /kaggle/input/realpde-kit \
        --out  /kaggle/working --epochs 30

`--kit` must contain `load_baseline.py`, `rpde_baselines/` and `_vendor/` from
the Track 1 kit. Prefer the fp32 `sim_real_fno.pth` over the fp16 pack for
training; export fp16 at the end with the kit's `pack_ckpt_fp16.py` if you want
the smaller submission.

AUGMENTATION (--noise-std, --amp-jitter): input-only, target untouched. See
augment_input()'s docstring for what's offered and, more importantly, what's
deliberately NOT offered (spatial flips/crops, time-reversal) and why.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------- constants
T_IN = T_OUT = 20
SUB = 2                      # 64x128 -> the scored 32x64
H, W = 32, 64

# GaussianNormalizer statistics, from the organizers' submission_example_fno.py
MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], np.float32)
STD_IN = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], np.float32)
STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)

# The eval regime, and the held-out trajectories. SELECT chooses the epoch;
# FINAL is scored once at the end and never used for any decision.
#
# 22 Aug: replaced the old 4-trajectory SELECT/FINAL with the CAMPAIGN_REPORT
# 12.3 holdout ("hold out >=12 trajectories completely, train on the rest,
# compare against 94.168 / 74.026 / 92.836") -- see build_holdout_v12_3.py.
# The old 4 were the closest trajectories to live but were ALSO the entire
# holdout, so every "local held-out" number all season was measured on 4
# windows-worth of trajectories out of 81 -- not memorized by the model
# (sim_real_ft trains on all 81, so nothing in train_real is unmemorized) but
# a dangerously small sample for an honest generalisation read. The new
# holdout is the 14 trajectories nearest the live fingerprint (the hardest,
# most decision-relevant version of the test), backfilled so the Re ceiling
# (26700) always stays in training and every holdout member is bracketed by a
# training neighbour on both sides at its own AoA -- matching the live task's
# own geometry (eval Re is interpolated, not extrapolated; see data-locations
# memory). SELECT keeps the original quad (already used for SPS band-multiplier
# fitting, which never touches model weights or rel/tke/mvpe, so still valid
# for epoch-picking); FINAL is the 10 new trajectories, untouched by anything
# in this campaign until the confirmation pass at the end of training.
REGIME_AOA = {10, 15, 20}
REGIME_RE = (19000, 27000)
SELECT = ["22875_15", "24150_15", "21600_15", "25425_15"]
FINAL = ["20325_15", "22875_10", "24150_10", "19050_15", "21600_10",
         "25425_10", "20325_20", "21600_20", "19050_20", "17775_15"]
EXCLUDE = ["7575_0"]         # byte-identical duplicate of 6300_0

# MVPE probe points, from the kit's scoring.py
PROBE_Y = list(range(8, 26, 2))
PROBE_X = [13, 21, 29, 37]


# ------------------------------------------------------------------- metrics
def rel_l2(p, t, eps=1e-8):
    """Per-sample relative L2 over the two measured channels."""
    n = p.shape[0]
    d = (p[..., :2] - t[..., :2]).reshape(n, -1)
    r = t[..., :2].reshape(n, -1)
    return d.norm(dim=1) / r.norm(dim=1).clamp_min(eps)


def ke_map(x):
    """0.5 (var_u + var_v) over the time axis -- the scorer's definition."""
    u, v = x[..., 0], x[..., 1]
    return 0.5 * (((u - u.mean(1, keepdim=True)) ** 2).mean(1)
                  + ((v - v.mean(1, keepdim=True)) ** 2).mean(1))


def tke_rel(p, t, eps=1e-8):
    n = p.shape[0]
    a, b = ke_map(p).reshape(n, -1), ke_map(t).reshape(n, -1)
    return (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(eps)


def mvpe_rel(p, t, eps=1e-8):
    """Mean of four per-column relative L2s, each on the TIME-AVERAGED probes.

    Matching scoring.py exactly matters and is easy to get wrong: the metric
    averages over the output horizon FIRST, then takes a separate rel-L2 for
    each of the four probe columns, then averages those four. Concatenating all
    probes into one norm without the time mean gives 0.783 where the scorer
    gives 0.202 on the same arrays.
    """
    n = p.shape[0]
    ys = torch.as_tensor(PROBE_Y, device=p.device)
    errs = []
    for px in PROBE_X:
        pp = p[:, :, ys, px, :2].mean(dim=1).reshape(n, -1)
        tt = t[:, :, ys, px, :2].mean(dim=1).reshape(n, -1)
        errs.append((pp - tt).norm(dim=1) / tt.norm(dim=1).clamp_min(eps))
    return torch.stack(errs, 0).mean(0)


def score_error(e):
    return 100.0 / (1.0 + 0.5 * e)


def sensitivity(e):
    """|d score_error / d err| -- how much a unit of this error is worth,
    in THAT METRIC's own score-space. Does NOT yet include how much that
    metric's score is worth in the composite -- see TRUE_W below."""
    return 50.0 / (1.0 + 0.5 * e) ** 2


# 22 Aug: sps is NOT an independent regressor. scoring.py's aggregate_sps
# builds it from the SAME rel/tke/mvpe errors (weight_dm=0.5, weight_tke=0.3,
# weight_mvpe=0.2 of a (1-pm)*exp(-band) term, pm = err/(0.5+err) -- see
# scoring.py:218-240), so the fitted composite table (rel .476/tke .086/
# mvpe .095/sps .243) is a set of PARTIAL derivatives that understate every
# accuracy metric's true value. TRUE_W is the TOTAL derivative d(composite)/
# d(score_k), i.e. the direct weight PLUS its indirect contribution through
# SPS's W_bar term, evaluated at the live v8 operating point (rel 94.168 /
# tke 74.026 / mvpe 92.836, live band term T=0.4914):
#     rel: 0.476 -> 0.649  (+36%)   tke: 0.086 -> 0.131 (+53%)
#     mvpe: 0.095 -> 0.160  (+68%)
# Previously `wr/wk/wm` below used sensitivity(e) ALONE, i.e. implicitly
# TRUE_W=(1,1,1) -- which at the stock operating point puts rel and mvpe
# within a few percent of each other (0.387 vs 0.376) when rel is actually
# worth ~4x more. That is very likely why prior fine-tunes bought tke/mvpe
# and sold rel (memory: "sensitivity weights" config, FINAL Drel -0.89 /
# Dtke +2.66) -- the loss itself was structurally biased toward the cheap
# metrics, independent of any data problem. Recompute if the operating point
# moves far from v8's (the sensitivity curve is convex, so weights drift
# faster at low error; this is a fixed-point approximation, not exact).
TRUE_W = dict(rel=0.649, tke=0.131, mvpe=0.160)


# ---------------------------------------------------------------------- data
def _stem(name):
    """Trajectory name without any packing artefacts: '10125_0.h5' -> '10125_0'."""
    s = str(name).strip()
    for suf in (".h5", ".hdf5", ".npy"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


class Source:
    """Trajectory source: either a directory of .h5 files or a packed .npz.

    The .npz layout is auto-detected and printed, because a silently
    mis-parsed archive would train on the wrong thing and look fine. Supported
    layouts, in order of preference:

      A) one array per trajectory, keyed by name, shape (T, H, W, 2 or 3)
      B) keys "<name>_u" / "<name>_v", each (T, H, W)
      C) stacked "u","v" of (K, T, H, W) plus "names" (and optionally "re","aoa")

    Re and AoA come from a "re"/"aoa" array when present, else from the name --
    and if they come from the name they are NOT trustworthy, since 78 of 82
    source files disagree with their filename. The script says so if it has to.
    """

    def __init__(self, path):
        self.path = path
        self.is_npz = os.path.isfile(path) and path.endswith(".npz")
        self.name_re = False
        if not self.is_npz:
            self.items = []
            for fn in sorted(os.listdir(path)):
                if fn.endswith(".h5") and fn[:-3] not in EXCLUDE:
                    with h5py.File(os.path.join(path, fn), "r") as f:
                        self.items.append((_stem(fn), float(f["re"][()]),
                                           float(f["aoa"][()])))
            return

        z = np.load(path, allow_pickle=True, mmap_mode=None)
        self.z = z
        keys = list(z.files)
        meta = {k: z[k] for k in ("re", "aoa", "names", "name", "keys", "lengths")
                if k in keys}
        arrs = [k for k in keys if k not in meta]
        # Real per-trajectory frame counts. A stacked archive is padded to the
        # longest trajectory, and three of the 82 are short (607, 492, 282
        # frames). Windows cut past the real length would be trained on padding.
        self.lengths = (np.asarray(meta["lengths"]).astype(int)
                        if "lengths" in meta else None)

        if "u" in keys and "v" in keys and z["u"].ndim == 4:
            self.layout = "C"
            # Names may or may not carry the ".h5" the archive was packed from.
            # Normalise, or every held-out lookup silently misses.
            self._names = [_stem(s) for s in (meta.get("names", meta.get("name")))]
            # Decompress u and v ONCE. NpzFile re-inflates the whole array on
            # every __getitem__, so per-trajectory access would re-read ~291 MB
            # eighty-two times over.
            print("  decompressing u/v once into RAM ...", flush=True)
            self._u, self._v = z["u"], z["v"]
            print(f"  u {self._u.shape} {self._u.dtype}   "
                  f"v {self._v.shape} {self._v.dtype}"
                  + (f"   lengths {self.lengths.min()}-{self.lengths.max()}"
                     if self.lengths is not None else "   (no lengths array)"),
                  flush=True)
        elif any(k.endswith("_u") for k in arrs):
            self.layout = "B"
            self._names = sorted({_stem(k[:-2]) for k in arrs if k.endswith("_u")})
        else:
            self.layout = "A"
            self._names = sorted(_stem(k) for k in arrs)

        self._raw = {_stem(k): k for k in keys}
        print(f"npz layout {self.layout}: {len(self._names)} trajectories")
        self.items = []
        for i, nm in enumerate(self._names):
            if nm in EXCLUDE:
                continue
            if "re" in meta:
                re, aoa = float(np.ravel(meta["re"])[i]), float(np.ravel(meta["aoa"])[i])
            else:
                self.name_re = True
                try:
                    a, b = nm.split("_")[:2]
                    re, aoa = float(a), float(b)
                except Exception:
                    re, aoa = float("nan"), float("nan")
            self.items.append((nm, re, aoa))
        if self.name_re:
            print("  WARNING: no re/aoa arrays in the npz, so Reynolds is being "
                  "parsed from the trajectory NAME. In the source data 78 of 82 "
                  "files disagree with their filename (true step 1278-1279, not "
                  "1275). Regime weighting will be approximate.")

    def frames(self, name):
        """Return (T, 32, 64, 3) float32, already subsampled if needed."""
        if not self.is_npz:
            with h5py.File(os.path.join(self.path, name + ".h5"), "r") as f:
                u = np.asarray(f["u"][:, ::SUB, ::SUB], np.float32)
                v = np.asarray(f["v"][:, ::SUB, ::SUB], np.float32)
        elif self.layout == "B":
            k = self._raw.get(name, name)
            u = np.asarray(self.z[k + "_u"], np.float32)
            v = np.asarray(self.z[k + "_v"], np.float32)
        elif self.layout == "C":
            i = self._names.index(name)
            n = int(self.lengths[i]) if self.lengths is not None else None
            u = np.asarray(self._u[i, :n], np.float32)
            v = np.asarray(self._v[i, :n], np.float32)
        else:
            a = np.asarray(self.z[self._raw.get(name, name)], np.float32)
            u, v = a[..., 0], a[..., 1]
        if u.shape[-2:] == (64, 128):          # not yet subsampled
            u, v = u[:, ::SUB, ::SUB], v[:, ::SUB, ::SUB]
        assert u.shape[-2:] == (H, W), f"{name}: got {u.shape[-2:]}, want ({H},{W})"
        return np.stack([u, v, np.zeros_like(u)], -1)

    def n_frames(self, name):
        return self.frames(name).shape[0] if self.is_npz else \
            h5py.File(os.path.join(self.path, name + ".h5"), "r")["u"].shape[0]


def in_regime(re, aoa):
    return aoa in REGIME_AOA and REGIME_RE[0] <= re <= REGIME_RE[1]


# --------------------------------------------- spectral gain (masterplan v11 sec.3)
# Inlined rather than imported from spectral_gain.py: this file is sometimes
# pasted whole into a Kaggle notebook cell (see _in_notebook below), where a
# sibling-file import would not resolve. Kept numerically identical to
# spectral_gain.py -- see that file for the derivation and the standalone
# verification gate (run it locally against a candidate pool before trusting
# --spectral-aug on Kaggle).
_SPEC_AMP_GAIN = np.sqrt(np.array([1.878, 1.602, 1.419, 1.381]))  # same for u and v


def _radial_bin_index(h, w_rfft):
    kh = np.minimum(np.arange(h), h - np.arange(h))
    kw = np.arange(w_rfft)
    rad = np.sqrt(kh[:, None].astype(float) ** 2 + kw[None, :].astype(float) ** 2)
    rmax = rad.max() + 1e-6
    edges = [0, rmax * 0.25, rmax * 0.5, rmax * 0.75, rmax + 1]
    idx = np.zeros(rad.shape, dtype=np.int64)
    for j in range(4):
        idx[(rad >= edges[j]) & (rad < edges[j + 1])] = j
    return idx


def _inpaint_nearest(frame, valid):
    if valid.all():
        return frame
    from scipy.ndimage import distance_transform_edt
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    return frame[iy, ix]


def spectral_gain_input(x, gain=_SPEC_AMP_GAIN):
    """x: (T, H, W, >=2) float32 INPUT window. Scales the fluctuation field's
    radial spectrum by `gain` (amplitude, sqrt of the measured power ratio),
    preserves phase, restores masked cells exactly. Target is never touched
    by this -- same input-only restriction as augment_input()."""
    x = x.copy()
    u, v = x[..., 0], x[..., 1]
    T, H, W = u.shape
    valid = (u != 0.0) | (v != 0.0)
    ui = np.stack([_inpaint_nearest(u[t], valid[t]) for t in range(T)])
    vi = np.stack([_inpaint_nearest(v[t], valid[t]) for t in range(T)])
    mu, mv = ui.mean(0, keepdims=True), vi.mean(0, keepdims=True)
    Fu = np.fft.rfft2(ui - mu, axes=(1, 2))
    Fv = np.fft.rfft2(vi - mv, axes=(1, 2))
    bidx = _radial_bin_index(*Fu.shape[1:])
    g = gain[bidx][None]
    Fu *= g; Fv *= g
    u_aug = (np.fft.irfft2(Fu, s=(H, W), axes=(1, 2)) + mu).astype(np.float32)
    v_aug = (np.fft.irfft2(Fv, s=(H, W), axes=(1, 2)) + mv).astype(np.float32)
    x[..., 0] = np.where(valid, u_aug, u)
    x[..., 1] = np.where(valid, v_aug, v)
    return x


class Windows(torch.utils.data.Dataset):
    """Stride-N (input, target) windows, sliced LAZILY from per-trajectory
    frames cached once -- NOT pre-sliced into their own copies.

    The previous version pre-sliced every window eagerly, which duplicates
    each frame (T_IN+T_OUT)/stride times: at stride 2 that is 20x, and 77
    training trajectories x ~415 windows each of 40 frames at (32,64,3)
    float16 came to ~15 GB. That is what OOM'd a 13 GB Kaggle T4 instance on
    2026-08-13 -- stride=2 was this file's own default, chosen to fix the
    under-determined-fine-tune bug, without checking its RAM footprint before
    shipping it as the default. Caching only the source frames (~10 MB per
    trajectory, ~800 MB for all 81) costs the same memory at ANY stride,
    including stride 1, because it never duplicates a frame.
    """

    def __init__(self, src, names, stride=2, spectral_aug=False):
        self.frames = {}
        self.index = []                       # (name, start) per window
        self.per_traj = []
        self.spectral_aug = spectral_aug
        for nm in names:
            fr = src.frames(nm).astype(np.float16)
            self.frames[nm] = fr
            starts = list(range(0, len(fr) - (T_IN + T_OUT) + 1, stride))
            self.index.extend((nm, i) for i in starts)
            self.per_traj.append((nm, len(starts)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        nm, s = self.index[i]
        fr = self.frames[nm]
        x = fr[s:s + T_IN].astype(np.float32)
        y = fr[s + T_IN:s + T_IN + T_OUT].astype(np.float32)
        if self.spectral_aug:
            x = spectral_gain_input(x)
        return torch.from_numpy(x), torch.from_numpy(y)


# --------------------------------------------------------------------- model
def build(kit, ckpt, device):
    sys.path.insert(0, kit)
    from load_baseline import load_baseline
    model, meta = load_baseline(ckpt, device=device)
    return model.to(device), meta


def last_frame_term(xb, mt):
    """(N, 1, H, W, 2): masked-aware (x_last - MEAN_TGT) for u, v.

    x_last is the last input frame where the cell is measured (nonzero); a cell
    whose last frame is masked falls back to its masked window mean, and a cell
    never measured in the window contributes 0. This is EXACTLY the `El`
    estimator of lab/build_v17pw/submission.py::_blend, so training and
    inference agree bit-for-bit on the definition."""
    xc = xb[..., :2]
    nz = xc != 0.0
    cnt = nz.sum(dim=1)
    em = xc.sum(dim=1) / cnt.clamp_min(1).to(xc.dtype)
    el = torch.where(nz[:, -1], xc[:, -1], em)
    return torch.where(cnt > 0, el - mt[:2], torch.zeros_like(el)).unsqueeze(1)


def forward_physical(model, xb, mi, si, mt, st, gate=None):
    """Normalize, run, denormalize -- the organizers' evaluation pipeline.

    26 Sep, --residual-last: with a (T_OUT, 2) `gate`, the output becomes
        P = model(xn) * st + mt + gate[lead, ch] * (x_last - mt)
    gate = 0 is the plain model (so a sim_fno start is reproduced EXACTLY at
    step 0); gate = 1 means the network output is the residual y - x_last in
    TARGET-normalised units, o = (y - x_last) / st. The gate is learned per
    (lead, channel), so the net can be residual at short leads and absolute at
    long ones. Rescaling fc2 cannot give 'residual = P_sim - x_last' at init
    (x_last is an input, not a feature), which is why the gate exists."""
    p = model((xb - mi) / si) * st + mt
    if gate is None:
        return p
    g = gate.view(1, -1, 1, 1, 2)
    return torch.cat([p[..., :2] + g * last_frame_term(xb, mt), p[..., 2:]], -1)


def free_gpu(tag=""):
    """Release whatever a PREVIOUS run in the same kernel left on the card.

    A notebook cell does not get a fresh process. Calling run() twice leaves the
    first run's model (201 MB), its gradients (201 MB) and AdamW's two momentum
    buffers (403 MB) resident, plus every activation that was live at the moment
    an interrupt landed -- and an interrupted backward pass keeps the whole graph
    alive. The second run then builds a second model on top and dies at the first
    3-D irfftn. That is exactly the OOM of 2026-08-13: batch 4 was never the
    problem (the prior run did five clean epochs at batch 4); a dirty kernel was.
    Reported rather than only fixed, so a dirty kernel is VISIBLE before it costs
    an hour instead of after.
    """
    import gc
    gc.collect()
    if not torch.cuda.is_available():
        return None
    torch.cuda.empty_cache()
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    free, total = torch.cuda.mem_get_info()
    print(f"  GPU{tag}: {free/2**30:.2f} GiB free of {total/2**30:.2f} GiB"
          f"   (torch allocated {torch.cuda.memory_allocated()/2**30:.2f}, "
          f"reserved {torch.cuda.memory_reserved()/2**30:.2f})", flush=True)
    if free < 0.75 * total:
        print("  ** WARNING: most of the card is already in use before training "
              "starts. Something from an earlier cell is still resident -- "
              "RESTART THE KERNEL. Continuing will very likely OOM. **",
              flush=True)
    return free


# ------------------------------------------------------------- augmentation
# Per the organizers' 2026-08-09 clarification: standard augmentation of the
# released data is allowed in both tracks, provided every training sample
# traces back to it. What's offered here is deliberately narrow -- most of the
# usual image/video augmentations do not apply to this data:
#
#   NO spatial flips. NACA4418 has 4% camber, so the flow is not top/bottom
#   symmetric (mirroring v -> -v, y -> -y) samples an attack angle that was
#   never measured. It is also not front/back symmetric (leading vs trailing
#   edge, wake only forms downstream), so a chordwise flip is equally invalid.
#   NO spatial crop/shift/zoom. MVPE is graded at FIXED pixel indices
#   (PROBE_Y/PROBE_X below) relative to the airfoil; any re-registration
#   desyncs training from the graded geometry.
#   NO time-reversal. This is a forecasting task and Navier-Stokes is not
#   time-symmetric (viscous dissipation has a sign), so a reversed window
#   teaches an arrow of time that does not exist in the eval task.
#
# What's left, and what's applied here, touches ONLY the model's input -- the
# target stays the real measured future exactly as recorded:
#   1. additive per-channel noise, sized as a fraction of that channel's own
#      GaussianNormalizer std (u, v only -- p is always exactly zero and must
#      stay that way, see _MEAN_IN/_STD_IN above).
#   2. optional per-window amplitude jitter: scale the input window's
#      fluctuation about its OWN time-mean by a factor near 1, holding the
#      mean field fixed. This targets the single most robustly measured
#      distribution shift in the whole campaign -- the live eval fingerprint's
#      sigma_local(u) runs 1.94x the local pool's, u_std 1.37x -- i.e. the live
#      flow is measurably more energetic than what SELECT/FINAL see. It does
#      NOT change which physical regime the sample is a measurement of any
#      more than sensor noise would; it is still real, measured train_real
#      data, just with its own fluctuation dialed up or down. Off by default
#      pending a check against the FAQ's exact augmentation boundary text --
#      turn on deliberately, not as a default.
#
# Neither ever touches a masked cell: real masked cells are EXACTLY zero in
# both u and v (see the harness's own zerofrac accounting), and perturbing
# them would erase the one signal the model has for "no measurement here."
def augment_input(xb, noise_std=0.02, amp_jitter=0.0):
    if noise_std <= 0 and amp_jitter <= 0:
        return xb
    xb = xb.clone()
    valid = (xb[..., 0] != 0.0) | (xb[..., 1] != 0.0)      # (N, T, H, W)
    uv = xb[..., :2]
    if amp_jitter > 0:
        n = xb.shape[0]
        # LOG-uniform, not the linear 1 +/- a it used to be. The shift this is
        # meant to cover is MULTIPLICATIVE and large: live sigma_local(u) is
        # 0.0157 against the local pool's 0.00809, a factor of 1.94. Reaching
        # 1.94 with the old linear form needed a=0.94, which also produced
        # gamma as low as 0.06 -- a dead flow, and unphysical. exp(a*U(-1,1))
        # is symmetric in log space, so a=0.70 spans gamma in [0.50, 2.01]:
        # it covers the measured shift in BOTH directions without ever
        # collapsing the field. This is domain randomisation along the one
        # axis whose train/eval gap has actually been measured -- which is the
        # standard tool for exactly the sim2real gap this track is named after.
        gamma = torch.exp(amp_jitter
                          * (2 * torch.rand(n, 1, 1, 1, 1, device=xb.device) - 1))
        mean = uv.mean(dim=1, keepdim=True)
        jittered = mean + (uv - mean) * gamma
        uv = torch.where(valid.unsqueeze(-1), jittered, uv)
    if noise_std > 0:
        std = torch.tensor([STD_IN[0], STD_IN[1]], device=xb.device) * noise_std
        noisy = uv + torch.randn_like(uv) * std
        uv = torch.where(valid.unsqueeze(-1), noisy, uv)
    xb[..., :2] = uv
    return xb


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, dev, mi, si, mt, st, gate=None):
    model.eval()
    R, K, M = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        p = forward_physical(model, xb, mi, si, mt, st, gate)
        p = torch.cat([p[..., :2], torch.zeros_like(p[..., 2:])], -1)
        R.append(rel_l2(p, yb)); K.append(tke_rel(p, yb)); M.append(mvpe_rel(p, yb))
    r, k, m = (torch.cat(z).mean().item() for z in (R, K, M))
    s = dict(rel=score_error(r), tke=score_error(k), mvpe=score_error(m))
    s["sum"] = s["rel"] + s["tke"] + s["mvpe"]
    s["err"] = (r, k, m)
    return s


# ---------------------------------------------------------------------- main
# ---------------------------------------------------------------- notebook
# Paste-into-a-cell configuration. Running this file as a notebook cell means
# argparse sees the kernel's own argv ("colab_kernel_launcher.py -f ...") and
# dies on the required arguments, so the three paths default to these and the
# parser ignores unknown args. Edit these, or call run(data=..., ckpt=...).
#
# 23 Aug: a bare run(ckpt=..., lr=..., ...) call that omitted `kit=` fell
# through to this dict's OLD defaults (a flat "/kaggle/input/realpde-fno",
# missing the "datasets/shashwatsrivastav46/" segment this specific Kaggle
# account's mount actually needs) -- ckpt/data both resolved fine because
# they were passed explicitly in that call, but `kit` silently used the wrong
# fallback and load_baseline.py was not there, crashing build() partway
# through data loading (after the expensive npz decompress). Fixed by making
# these defaults the paths `!find /kaggle/input -name ...` actually verified
# 22 Aug, so any FUTURE bare run(**kw) -- even one that forgets a path -- has
# a working fallback instead of a silent wrong one. Re-verify with
# `!find /kaggle/input -name "load_baseline.py"` etc. if the dataset/model is
# ever re-uploaded (a new version can shift the trailing version number in
# the Kaggle Model path in particular).
CONFIG = dict(
    data="/kaggle/input/datasets/shashwatsrivastav46/realpde-train/kaggle_train_real.npz",
    ckpt="/kaggle/input/datasets/shashwatsrivastav46/realpde-fno/sim_real_fno.pth",
    kit="/kaggle/input/datasets/shashwatsrivastav46/realpde-fno",
    out="/kaggle/working",
    pool_file="/kaggle/working/train_pool_v12_3.json",
)


def _autodiscover():
    """Find the dataset and kit under /kaggle/input wherever Kaggle mounted them."""
    import glob
    hits = glob.glob("/kaggle/input/**/kaggle_train_real.npz", recursive=True)
    if hits:
        CONFIG["data"] = hits[0]
    for pat in ("**/sim_real_fno.pth", "**/sim_real_fno_fp16.pth"):
        hits = glob.glob(f"/kaggle/input/{pat}", recursive=True)
        if hits:
            CONFIG["ckpt"] = hits[0]
            CONFIG["kit"] = os.path.dirname(hits[0])
            break
    hits = glob.glob("/kaggle/input/**/load_baseline.py", recursive=True)
    if hits:
        CONFIG["kit"] = os.path.dirname(hits[0])
    return CONFIG


def run(**kw):
    """Call this from a notebook cell: run(epochs=30) or run(data=..., ckpt=...)."""
    cfg = dict(CONFIG)
    cfg.update(kw)
    argv = []
    for k, v in cfg.items():
        if v is True:
            argv.append(f"--{k.replace('_', '-')}")
        elif v is not None and v is not False:
            # IDENTITY checks, not `v not in (None, False)`. Python compares by
            # EQUALITY there, and `0.0 == False` is True -- so run(w_tke=0.0)
            # silently DROPPED the flag and main() fell back to the
            # sensitivity-derived weight. The zero-tke experiment would have run
            # as an ordinary fine-tune and reported itself as the experiment.
            # Every zero-valued option had this hole: w_*, noise_std,
            # amp_jitter, steps_per_epoch, time_budget.
            argv += [f"--{k.replace('_', '-')}", str(v)]
    return main(argv)


def _in_notebook():
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


def main(argv=None):
    # A notebook cell leaves the kernel's own argv in place ("-f kernel.json"),
    # which argparse rejects. parse_known_args below ignores it; autodiscovery
    # here fixes the paths whenever the configured ones do not exist, so the
    # same file works pasted into a cell or run from a shell.
    if argv is None and (_in_notebook()
                         or not os.path.exists(CONFIG["data"])
                         or not os.path.exists(CONFIG["ckpt"])):
        _autodiscover()
        print("autodiscovered paths under /kaggle/input:", flush=True)
        for k in ("data", "ckpt", "kit"):
            print(f"  {k:5} {CONFIG[k]}"
                  f"   {'OK' if os.path.exists(CONFIG[k]) else 'NOT FOUND'}",
                  flush=True)
        argv = []
        for k, v in CONFIG.items():
            # Match run()'s own dash-replacement: CONFIG now carries
            # `pool_file` (added 23 Aug) and an underscore key built as
            # `--pool_file` here would silently miss --pool-file's argparse
            # flag and get dropped by parse_known_args -- exactly the kind of
            # "no error, wrong default" failure this file's autodiscovery
            # block is trying to avoid in the first place.
            argv += [f"--{k.replace('_', '-')}", str(v)]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=CONFIG["data"])
    ap.add_argument("--ckpt", default=CONFIG["ckpt"])
    ap.add_argument("--kit", default=CONFIG["kit"])
    ap.add_argument("--out", default=CONFIG["out"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=None,
                    help="evaluation-only batch size. Defaults to --batch; may be "
                         "larger because evaluate() uses no_grad and retains no "
                         "training activations.")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    # stride 20 gave 3,362 windows to fine-tune a 50,357,955-parameter model.
    # That is drastically under-determined and produced textbook overfitting:
    # tke +2.66 while rel, mvpe and sps all fell (net -1.86). The result was
    # recorded as "training does not help" and the direction was closed for a
    # week. It was never a capability ceiling, it was too little data.
    # stride 2 gives ~33,000 windows; stride 1 gives 66,748.
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--regime-weight", type=float, default=3.0,
                    help="oversampling factor for in-regime trajectories")
    ap.add_argument("--subset-regime", action="store_true",
                    help="train ONLY on in-regime trajectories (small upload)")
    ap.add_argument("--w-rel", type=float, default=None)
    ap.add_argument("--w-tke", type=float, default=None)
    ap.add_argument("--w-mvpe", type=float, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=1200,
                    help="optimizer-visible iterations per epoch. 0 = a full "
                         "pass over every window (~7800 at stride 2 -- far too "
                         "slow; that is what made the first run look hung).")
    ap.add_argument("--log-every", type=int, default=100,
                    help="print progress every N batches, with a live ETA.")
    ap.add_argument("--time-budget", type=float, default=0.0,
                    help="hours; stop cleanly after this much TRAINING time and "
                         "still run the FINAL confirmation. 0 = no limit. Set it "
                         "below the Kaggle session cap so a kill cannot discard "
                         "the run.")
    ap.add_argument("--noise-std", type=float, default=0.02,
                    help="input-only additive noise, as a fraction of each "
                         "channel's own STD_IN. 0 disables.")
    ap.add_argument("--amp-jitter", type=float, default=0.0,
                    help="input-only per-window amplitude jitter around its "
                         "own time-mean, e.g. 0.15 for a U(0.85,1.15) factor. "
                         "0 disables (default off -- see augment_input's "
                         "docstring before turning this on).")
    ap.add_argument("--amp", action="store_true",
                    help="opt in to fp16 autocast + GradScaler training. Safe for "
                         "convolutional CNO on CUDA; leave off for FNO because its "
                         "non-power-of-two FFT path cannot run under fp16.")
    ap.add_argument("--wd", type=float, default=1e-5, help="AdamW weight decay")
    ap.add_argument("--pool-file", type=str, default=None,
                    help="JSON file with {'pool': [traj_name, ...]} (e.g. "
                         "live_pool.json from build_live_pool.py). When set, "
                         "training uses EXACTLY this trajectory list instead "
                         "of --subset-regime/--regime-weight selection.")
    ap.add_argument("--freeze-bn-fc", action="store_true",
                    help="masterplan v11 sec.5: freeze every parameter except "
                         "fc0, fc1, fc2 and BatchNorm affine (bns.*).")
    ap.add_argument("--unfreeze-extra", type=str, default="",
                    help="25 Sep, direction (b): comma-separated parameter-name "
                         "prefixes to ALSO leave trainable on top of "
                         "--freeze-bn-fc, e.g. 'convs.' (pointwise skip path, "
                         "16.6k params) or 'convs.,spectral_convs.3.' (+ last "
                         "spectral block, 12.6M). Intermediate freeze ratio "
                         "between 15.8 (0 spectral) and Experiments 2/3 (all).")
    ap.add_argument("--snapshot-last-k", type=int, default=0,
                    help="6 Sep, SWA prep: also save EVERY epoch's state_dict to "
                         "out/snap_epNNN.pth, keeping only the most recent K on "
                         "disk (rotating -- older ones deleted as new ones land, "
                         "so this never grows unbounded). 0 = off (default, no "
                         "extra I/O). Independent of the SELECT-best checkpoint "
                         "already saved separately below -- this is raw material "
                         "for weight-averaging (see swa_average.py) AFTER "
                         "training, not a selection mechanism itself. Only "
                         "meaningful when the run shares one basin throughout "
                         "(e.g. --freeze-bn-fc, where spectral weights never "
                         "move) -- averaging epochs from an UNFROZEN run that is "
                         "still moving through weight space is not standard SWA "
                         "and was not what this was built for.")
    ap.add_argument("--spectral-aug", action="store_true",
                    help="masterplan v11 sec.3: apply the corrected sqrt-gain "
                         "radial spectrum augmentation to the INPUT window. "
                         "Requires scipy (nearest-neighbour inpainting).")
    ap.add_argument("--fixed-epochs", action="store_true",
                    help="masterplan v11 sec.5: no SELECT-based epoch "
                         "selection -- always keep the LAST epoch's weights. "
                         "SELECT is still evaluated and logged, just not used "
                         "to pick a checkpoint.")
    ap.add_argument("--abort-if-flat", action="store_true",
                    help="masterplan v11 sec.5: stop after epoch 1 if its "
                         "training loss did not improve on epoch 0's.")
    ap.add_argument("--seed", type=int, default=0,
                    help="23 Aug, Experiment 3: seed 0 is what Experiments 1 "
                         "and 2 both ran with (torch/numpy default global "
                         "state, never explicitly pinned). A single run's "
                         "local delta -- Experiment 2's true-composite -0.10 "
                         "on the official stride-20 rescore -- says nothing "
                         "about whether that's this seed's bad luck or the "
                         "approach itself; change this to get an independent "
                         "draw with the SAME config before concluding either way.")
    ap.add_argument("--restarts", type=int, default=0,
                    help="23 Aug, Experiment 3: if >0, use "
                         "CosineAnnealingWarmRestarts with T_0 = epochs/"
                         "(restarts+1) instead of a single flat "
                         "CosineAnnealingLR decay. Both Experiment 1 (80 "
                         "epochs, plateaued by epoch 10) and Experiment 2 (30 "
                         "epochs, SELECT sum flat after ~epoch 14) stalled "
                         "well before their schedule finished annealing -- a "
                         "restart is specifically meant to kick a stalled run "
                         "out of the basin it settled into, rather than just "
                         "decaying LR toward zero inside it.")
    ap.add_argument("--residual-last", action="store_true",
                    help="26 Sep: add a learned per-(lead, channel) gate times "
                         "(x_last - MEAN_TGT) to the output (see "
                         "forward_physical). Gate saved as 'resid_gate' next to "
                         "model_state_dict; inference must apply it (lab/build_resid).")
    ap.add_argument("--resid-gate-init", type=float, default=0.0,
                    help="initial gate value; 0 reproduces the start checkpoint "
                         "exactly at step 0.")
    ap.add_argument("--resid-gate-lr", type=float, default=1e-2,
                    help="AdamW lr of the 40 gate scalars (no weight decay). "
                         "At the model lr (1e-4) the gate could move only ~0.3 "
                         "in 12 epochs.")
    ap.add_argument("--allow-heldout-in-pool", action="store_true",
                    help="26 Sep: let --pool-file contain SELECT/FINAL "
                         "trajectories (train_pool_all81.json). Without it such "
                         "a pool exits. SELECT/FINAL numbers are then NOT "
                         "honest; use with --fixed-epochs.")
    a, _unknown = ap.parse_known_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}  device {dev}", flush=True)
    if dev == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}", flush=True)
        # Fragmentation was 2.96 GiB of the 14.56 GiB card in the 13 Aug OOM.
        # Must be set before the first allocation to take effect.
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        free_gpu(" before load")

    src = Source(a.data)
    avail = {nm for nm, _, _ in src.items}
    missing = [n for n in SELECT + FINAL if n not in avail]
    if missing:
        sys.exit(f"held-out trajectories missing from the data: {missing}\n"
                 f"available names look like: {sorted(avail)[:6]}")

    held = set(SELECT) | set(FINAL)
    if a.pool_file:
        # 6 Sep bugfix: CONFIG's hardcoded default ("/kaggle/working/...") only
        # holds if someone manually copied the file into the session's SCRATCH
        # dir first. A dataset attached the normal way (Add Data) mounts
        # read-only under /kaggle/input/<slug>/... instead, and this path was
        # never run through _autodiscover() at all (that only fires when
        # data/ckpt are ALSO missing) -- so a fresh session with the json
        # attached as an input dataset crashed here every time with a
        # FileNotFoundError, having already spent minutes decompressing the
        # main .npz first. Fall back to searching /kaggle/input before giving
        # up, exactly like data/ckpt/kit already do in _autodiscover().
        if not os.path.exists(a.pool_file):
            hits = glob.glob("/kaggle/input/**/" + os.path.basename(a.pool_file),
                              recursive=True)
            if hits:
                print(f"--pool-file {a.pool_file} not found; using "
                      f"{hits[0]} instead", flush=True)
                a.pool_file = hits[0]
            else:
                sys.exit(f"--pool-file {a.pool_file} not found, and no "
                          f"{os.path.basename(a.pool_file)} under /kaggle/input "
                          "either -- attach the jsonpool dataset or pass the "
                          "correct --pool-file path explicitly.")
        pool = json.load(open(a.pool_file))["pool"]
        clash = held & set(pool)
        if clash and a.allow_heldout_in_pool:
            print(f"--allow-heldout-in-pool: training ON {len(clash)} SELECT/FINAL "
                  "trajectories. SELECT/FINAL scores below are TRAINING scores, "
                  "not honest." + ("" if a.fixed_epochs else
                  " WARNING: without --fixed-epochs the epoch is picked on "
                  "contaminated SELECT."), flush=True)
        elif clash:
            sys.exit(f"--pool-file {a.pool_file} contains held-out trajectories: "
                      f"{clash} -- fix the pool, do not train on FINAL/SELECT")
        train_names = [nm for nm, _re, _aoa in src.items if nm in pool]
        missing = set(pool) - set(train_names)
        if missing:
            sys.exit(f"--pool-file names not found in --data: {missing}")
        weights = [1.0] * len(train_names)
        print(f"trajectories: {len(train_names)} from --pool-file {a.pool_file} "
              f"(overrides --subset-regime/--regime-weight)", flush=True)
    else:
        train_names, weights = [], []
        for nm, re, aoa in src.items:
            if nm in held:
                continue
            reg = in_regime(re, aoa)
            if a.subset_regime and not reg:
                continue
            train_names.append(nm)
            weights.append(a.regime_weight if reg else 1.0)
        print(f"trajectories: {len(src.items)} total, {len(train_names)} for training, "
              f"{len(held)} held out", flush=True)
        print(f"  in-regime (AoA {sorted(REGIME_AOA)}, Re {REGIME_RE}): "
              f"{sum(1 for w in weights if w > 1)}", flush=True)
    if not train_names:
        sys.exit("no training trajectories selected -- check --subset-regime, "
                 "--pool-file, and whether re/aoa parsed correctly")

    tr = Windows(src, train_names, a.stride, spectral_aug=a.spectral_aug)
    se = Windows(src, SELECT, a.stride)
    fi = Windows(src, FINAL, a.stride)
    frame_gb = sum(f.nbytes for f in tr.frames.values()) / 1e9
    print(f"windows: train {len(tr)}  select {len(se)}  final {len(fi)}"
          f"   (cached frames {frame_gb:.2f} GB -- constant at any stride, "
          f"since windows are sliced lazily, not pre-copied)", flush=True)

    # Oversample in-regime windows. Counts come from the dataset itself rather
    # than being recomputed, so the weights cannot drift out of alignment with
    # the samples they weight.
    wmap = dict(zip(train_names, weights))
    wpw = [wmap[nm] for nm, k in tr.per_traj for _ in range(k)]
    assert len(wpw) == len(tr), f"weight/window mismatch {len(wpw)} vs {len(tr)}"
    # DECOUPLE "epoch" FROM "one full pass over every window". At stride 2 there
    # are ~31,000 train windows, so a full pass is ~7,800 iterations at batch 4
    # -- 9.3x what stride 20 gave, and far too long to finish 30 of them inside
    # a Kaggle session. But the POINT of stride 2 was never to see every window
    # each epoch; it was to widen the pool the sampler draws from, so the model
    # stops being fitted against the same 3,362 windows over and over. Drawing a
    # fixed budget of samples per epoch keeps that benefit and makes epoch time
    # independent of stride: change --stride to change window DIVERSITY, change
    # --steps-per-epoch to change epoch COST. They are different knobs and were
    # wrongly welded together.
    steps = a.steps_per_epoch if a.steps_per_epoch > 0 else len(tr) // a.batch
    n_draw = steps * a.batch
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.tensor(wpw, dtype=torch.double), n_draw, replacement=True)
    print(f"schedule: {steps} steps/epoch x {a.batch} = {n_draw} windows drawn "
          f"per epoch from a pool of {len(tr)}  ({a.epochs} epochs -> "
          f"{steps*a.epochs*a.batch/max(len(tr),1):.1f} passes over the pool)",
          flush=True)

    eval_batch = a.eval_batch if a.eval_batch is not None else a.batch
    if eval_batch < 1:
        sys.exit("--eval-batch must be positive")
    print(f"batches: train {a.batch}  eval {eval_batch}", flush=True)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, sampler=sampler,
                                     num_workers=a.workers, pin_memory=(dev == "cuda"))
    dse = torch.utils.data.DataLoader(se, batch_size=eval_batch)
    dfi = torch.utils.data.DataLoader(fi, batch_size=eval_batch)

    model, meta = build(a.kit, a.ckpt, dev)
    model = model.float()          # fp16-packed weights -> fp32 for training
    print(f"loaded {a.ckpt}  meta {list(meta)[:5] if hasattr(meta,'__iter__') else meta}",
          flush=True)

    if a.freeze_bn_fc:
        # masterplan v11 sec.5: fc0/fc1/fc2 + BatchNorm affine only. `bns` is
        # the model's own name for its BatchNorm3d ModuleList (rpde_baselines
        # fno.py) -- freeze everything, then unfreeze by name so a future
        # rename of any OTHER module can't accidentally leave it trainable.
        n_total = n_trainable = 0
        keep = ("fc0.", "fc1.", "fc2.", "bns.") + tuple(
            s.strip() for s in a.unfreeze_extra.split(",") if s.strip())
        for name, p in model.named_parameters():
            n_total += p.numel()
            p.requires_grad = name.startswith(keep)
            if p.requires_grad:
                n_trainable += p.numel()
        print(f"--freeze-bn-fc: {n_trainable:,} / {n_total:,} parameters "
              f"trainable ({100*n_trainable/n_total:.2f}%)", flush=True)
        assert n_trainable > 0, "freeze-bn-fc found no fc0/fc1/fc2/bns parameters -- model naming changed?"

    mi, si = (torch.from_numpy(z).to(dev) for z in (MEAN_IN, STD_IN))
    mt, st = (torch.from_numpy(z).to(dev) for z in (MEAN_TGT, STD_TGT))

    base = evaluate(model, dse, dev, mi, si, mt, st)
    print(f"\nSTOCK CHECKPOINT on SELECT: rel {base['rel']:.3f}  tke {base['tke']:.3f}"
          f"  mvpe {base['mvpe']:.3f}  sum {base['sum']:.3f}", flush=True)

    # Loss weights = TRUE_W (composite value per unit of THIS metric's score)
    # x sensitivity(e) (how fast that score moves per unit of error) -- the
    # full chain rule d(composite)/d(err_k), not sensitivity alone. See
    # TRUE_W's comment above for why sensitivity alone was wrong.
    er, ek, em = base["err"]
    wr = a.w_rel if a.w_rel is not None else TRUE_W["rel"] * sensitivity(er)
    wk = a.w_tke if a.w_tke is not None else TRUE_W["tke"] * sensitivity(ek)
    wm = a.w_mvpe if a.w_mvpe is not None else TRUE_W["mvpe"] * sensitivity(em)
    tot = wr + wk + wm
    wr, wk, wm = wr / tot, wk / tot, wm / tot
    print(f"loss weights: rel {wr:.3f}  tke {wk:.3f}  mvpe {wm:.3f}"
          f"   (TRUE_W x d(score)/d(err) at the stock errors)", flush=True)
    print(f"augmentation: noise_std {a.noise_std}  amp_jitter {a.amp_jitter}"
          + ("  -- VERIFY amp_jitter against the FAQ's augmentation boundary "
             "before trusting a run that used it" if a.amp_jitter > 0 else ""),
          flush=True)

    gate = None
    if a.residual_last:
        gate = torch.nn.Parameter(torch.full((T_OUT, 2), a.resid_gate_init,
                                             dtype=torch.float32, device=dev))
        print(f"--residual-last: gate (T_OUT=20 x u,v) init {a.resid_gate_init}  "
              f"lr {a.resid_gate_lr}", flush=True)
        opt = torch.optim.AdamW(
            [{"params": [p for p in model.parameters() if p.requires_grad],
              "lr": a.lr, "weight_decay": a.wd},
             {"params": [gate], "lr": a.resid_gate_lr, "weight_decay": 0.0}])
    else:
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                lr=a.lr, weight_decay=a.wd)

    def _ckpt(**extra):
        d = {"model_state_dict": model.state_dict(), **extra}
        if gate is not None:
            d["resid_gate"] = gate.detach().cpu().clone()
        return d
    if a.restarts > 0:
        t0 = max(1, a.epochs // (a.restarts + 1))
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=t0)
        print(f"schedule: CosineAnnealingWarmRestarts T_0={t0} "
              f"({a.restarts} restart(s) across {a.epochs} epochs)", flush=True)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    # No automatic mixed precision by default. FNO is a spectral model: its forward runs
    # torch.fft.rfftn, and cuFFT only supports half precision when every signal
    # dimension is a power of two. Here they come out [26, 38, 70], so autocast
    # raises "cuFFT only supports dimensions whose sizes are powers of two".
    # Evaluation was unaffected because it runs under no_grad without autocast.
    # Turing (T4) has no usable bf16 either. CNO is convolutional, however, and
    # may explicitly opt into fp16 autocast to make a feasible microbatch.
    use_amp = bool(a.amp and dev == "cuda")
    if a.amp and not use_amp:
        print("--amp requested without CUDA; using fp32", flush=True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"training precision: {'fp16 autocast + GradScaler' if use_amp else 'fp32'}",
          flush=True)

    best = (-1e9, -1)
    hist = []
    snaps = []          # rotating buffer of the last K per-epoch snapshot paths
    t_train0 = time.perf_counter()
    stop_early = False
    for ep in range(a.epochs):
        model.train()
        t0, run, nb = time.perf_counter(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (xb, yb) in enumerate(dl):
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            xb = augment_input(xb, a.noise_std, a.amp_jitter)
            with torch.cuda.amp.autocast(enabled=use_amp):
                p = forward_physical(model, xb, mi, si, mt, st, gate)
                loss = (wr * rel_l2(p, yb).mean()
                        + wk * tke_rel(p, yb).mean()
                        + wm * mvpe_rel(p, yb).mean()) / a.accum
            scaler.scale(loss).backward()
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            run += loss.item() * a.accum; nb += 1
            # Progress INSIDE the epoch. Without this the first output of the
            # whole run arrives only after a full epoch plus an 830-window eval,
            # which is indistinguishable from a hang -- and is exactly how the
            # first stride-2 run was read.
            if a.log_every > 0 and (i + 1) % a.log_every == 0:
                el = time.perf_counter() - t0
                rate = el / (i + 1)
                print(f"    ep {ep:3d}  {i+1:5d}/{steps}  loss {run/max(nb,1):.4f}"
                      f"  {rate*1000:6.0f} ms/batch  epoch eta {(steps-i-1)*rate:5.0f}s",
                      flush=True)
        sched.step()
        if a.time_budget > 0 and (time.perf_counter() - t_train0) > a.time_budget * 3600:
            stop_early = True
        s = evaluate(model, dse, dev, mi, si, mt, st, gate)
        ep_loss = run / max(nb, 1)
        hist.append(dict(epoch=ep, loss=ep_loss, **{k: s[k] for k in
                                                     ("rel", "tke", "mvpe", "sum")}))
        flag = ""
        # masterplan v11 sec.5: --fixed-epochs takes the LAST epoch
        # unconditionally, ignoring SELECT ("no local validation"). SELECT is
        # still computed and logged above for visibility, just not used to
        # pick a checkpoint.
        if a.fixed_epochs or s["sum"] > best[0]:
            best = (s["sum"], ep)
            torch.save(_ckpt(), os.path.join(a.out, "fno_finetuned_best.pth"))
            flag = "  <- saved (fixed-epochs)" if a.fixed_epochs else "  <- best, saved"
        if a.snapshot_last_k > 0:
            # Independent of the best-by-SELECT save above: every epoch lands
            # on disk (for SWA averaging afterward), oldest deleted once more
            # than K are on disk. select_sum/epoch are stored alongside so
            # swa_average.py can report what it actually averaged.
            snap_path = os.path.join(a.out, f"snap_ep{ep:03d}.pth")
            torch.save(_ckpt(epoch=ep, select_sum=s["sum"]), snap_path)
            snaps.append(snap_path)
            while len(snaps) > a.snapshot_last_k:
                old = snaps.pop(0)
                try:
                    os.remove(old)
                except OSError:
                    pass
        print(f"epoch {ep:3d}  loss {ep_loss:.4f}  SELECT rel {s['rel']:.3f} "
              f"tke {s['tke']:.3f} mvpe {s['mvpe']:.3f} sum {s['sum']:.3f} "
              f"({s['sum']-base['sum']:+.3f})  {time.perf_counter()-t0:.0f}s{flag}",
              flush=True)
        if gate is not None:
            g = gate.detach().cpu()
            print("    resid gate u/v at leads 1,2,5,10,20: "
                  + "  ".join(f"L{l}:{g[l-1,0]:.2f}/{g[l-1,1]:.2f}" for l in (1, 2, 5, 10, 20)),
                  flush=True)
        if a.abort_if_flat and ep == 1 and ep_loss >= hist[0]["loss"]:
            print(f"\n--abort-if-flat: epoch 1 training loss ({ep_loss:.4f}) did "
                  f"not improve on epoch 0's ({hist[0]['loss']:.4f}). Stopping "
                  f"per masterplan v11 sec.5 -- this run gets recorded as a "
                  f"closed negative, not silently truncated.", flush=True)
            stop_early = True
        if stop_early:
            # Budget spent. Stop HERE, not mid-epoch: the best checkpoint is
            # already on disk and FINAL still gets its single honest look.
            print(f"\n--time-budget of {a.time_budget:.2f} h reached after epoch "
                  f"{ep}; stopping and going straight to FINAL.", flush=True)
            break

    # FINAL is touched exactly once, with the epoch already chosen on SELECT.
    print(f"\nbest epoch {best[1]} (SELECT sum {best[0]:.3f})", flush=True)
    if snaps:
        print(f"snapshots on disk for SWA ({len(snaps)}): "
              + ", ".join(os.path.basename(p) for p in snaps)
              + "\n  -> run: python swa_average.py " + " ".join(snaps)
              + f" --out {os.path.join(a.out, 'fno_swa.pth')}", flush=True)
    _best = torch.load(os.path.join(a.out, "fno_finetuned_best.pth"), map_location=dev)
    model.load_state_dict(_best["model_state_dict"])
    if gate is not None:
        with torch.no_grad():
            gate.copy_(_best["resid_gate"].to(dev))
        json.dump(gate.detach().cpu().tolist(),
                  open(os.path.join(a.out, "resid_gate.json"), "w"))
    del _best
    stock_f = None
    # The stock reference is a SECOND full 201 MB model held alongside the
    # trained one. Free the optimizer first -- its two AdamW momentum buffers
    # are another 403 MB and are finished with by this point.
    opt.zero_grad(set_to_none=True)
    del opt, sched
    free_gpu(" before stock reference")
    m2, _ = build(a.kit, a.ckpt, dev)
    stock_f = evaluate(m2, dfi, dev, mi, si, mt, st)
    del m2
    free_gpu(" after stock reference")
    fin = evaluate(model, dfi, dev, mi, si, mt, st, gate)
    print("\nCONFIRMATION ON FINAL (touched once):")
    print(f"  {'':10}{'rel':>9}{'tke':>9}{'mvpe':>9}{'sum':>10}")
    print(f"  {'stock':10}{stock_f['rel']:9.3f}{stock_f['tke']:9.3f}"
          f"{stock_f['mvpe']:9.3f}{stock_f['sum']:10.3f}")
    print(f"  {'finetuned':10}{fin['rel']:9.3f}{fin['tke']:9.3f}"
          f"{fin['mvpe']:9.3f}{fin['sum']:10.3f}")
    print(f"  {'delta':10}{fin['rel']-stock_f['rel']:+9.3f}"
          f"{fin['tke']-stock_f['tke']:+9.3f}{fin['mvpe']-stock_f['mvpe']:+9.3f}"
          f"{fin['sum']-stock_f['sum']:+10.3f}")

    json.dump({"history": hist, "best_epoch": best[1],
               "select_stock": {k: base[k] for k in ("rel", "tke", "mvpe", "sum")},
               "final_stock": {k: stock_f[k] for k in ("rel", "tke", "mvpe", "sum")},
               "final_finetuned": {k: fin[k] for k in ("rel", "tke", "mvpe", "sum")}},
              open(os.path.join(a.out, "finetune_log.json"), "w"), indent=1)
    print(f"\nsaved {os.path.join(a.out, 'fno_finetuned_best.pth')}")
    print("wrap it with the SAME normalizer constants and the SPS bands from "
          "submission_fno.zip; do not ship it bare -- the default +-5% band cost "
          "17.8 sps points on 7 Aug.")


if __name__ == "__main__":
    if _in_notebook():
        # Pasted directly into a Kaggle cell: do NOT fall through to
        # _autodiscover(). All paths below now also live in CONFIG's own
        # defaults (fixed 23 Aug), so a bare run(**overrides) call is safe
        # even if it omits one -- but they're kept explicit here too so this
        # block stays self-contained and readable on its own.
        #
        # 23 Aug, EXPERIMENT 2: ckpt is sim_real_fno.pth (the organizer's own
        # already-fine-tuned checkpoint), not sim_fno.pth. Experiment 1
        # (sim_fno.pth, lr=1e-4, 80 epochs) plateaued at epoch 10 and NEVER
        # improved for the remaining ~70 epochs -- FINAL(10) came in rel
        # 94.266/tke 78.782/mvpe 95.505 (sum 268.554), a LOSS of -5.928 sum vs
        # this same checkpoint's own frozen holdout score (274.482). Starting
        # from sim_real_fno.pth is NOT a clean holdout test any more (it has
        # already seen these 14 trajectories during the organizers' own
        # training, unlike sim_fno.pth), but it is the one genuinely untested
        # configuration left: does further training with the CORRECTED
        # TRUE_W loss weights and the wider live-similar pool improve on what
        # the organizers' own fine-tune already does. lr dropped 1e-4 -> 2e-5
        # and epochs 80 -> 30: this starting point is already near-optimal
        # (rel~96/tke~81/mvpe~97), so the failure mode to watch for is
        # different -- not a stall, but epoch 0 moving WORSE than the printed
        # "STOCK CHECKPOINT on SELECT" baseline, which would mean even 2e-5
        # is too large a step off a good optimum.
        # 23 Aug, EXPERIMENT 3: identical config to Experiment 2 except seed
        # and schedule. Experiment 2's official stride-20 rescore on the
        # untouched final(10) came back TRUE-composite -0.10 (rel -0.480 /
        # tke +2.099 / mvpe -0.396) -- a wash tipping negative, and the exact
        # "buy tke, sell rel/mvpe" shape that has failed to transfer live
        # every time this campaign has seen it before. One run cannot tell
        # you whether that's this seed's bad luck or the approach itself, so
        # this changes exactly two things: seed=1 (an independent draw from
        # the same pool/loss-weights/lr) and restarts=2 (both prior runs'
        # SELECT sum went flat well before their schedule finished decaying --
        # Experiment 1 by epoch 10 of 80, Experiment 2 by ~epoch 14 of 30 --
        # so a restart gets three separate chances to escape whatever basin
        # the flat decay was settling into, instead of one).
        # 6 Sep: STOPPED auto-running this on paste. Experiment 3 is CLOSED
        # (TRUE-composite -0.10, the "buy tke, sell rel/mvpe" shape -- see the
        # comment block above) -- it has nothing left to teach, and having it
        # fire unconditionally on every fresh Kaggle session burned real GPU
        # minutes decompressing data and loading the checkpoint before even
        # reaching training, every single time this file got pasted into a
        # cell just to make its HELPERS (build/run/main/etc.) importable for
        # a genuinely new experiment. Pasting this file now only defines
        # those helpers -- it trains nothing by itself. To actually run
        # Experiment 3 again on purpose, call _experiment_3() explicitly.
        print("finetune_fno_kaggle.py helpers loaded (build/run/main/etc). "
              "Nothing was trained automatically -- Experiment 3 (sim_real_fno, "
              "already closed, TRUE-composite -0.10) no longer auto-runs on "
              "paste. Call run(...) yourself with the config you actually "
              "want, e.g.:\n"
              "  run(ckpt='<sim_fno.pth path>', pool_file='train_pool_v12_3.json', "
              "freeze_bn_fc=True, spectral_aug=True, snapshot_last_k=5, "
              "epochs=30, time_budget=4)\n"
              "(pool_file no longer needs an exact path -- it now falls back "
              "to searching /kaggle/input automatically if the given one is "
              "missing.) To re-run the old, already-closed Experiment 3 "
              "instead, call _experiment_3().", flush=True)

        def _experiment_3():
            """The 23 Aug run this cell used to fire automatically. Already
            closed (TRUE-composite -0.10) -- kept only for reproducibility,
            not a recommendation to re-run it."""
            return run(
                ckpt="/kaggle/input/datasets/shashwatsrivastav46/realpde-fno/sim_real_fno.pth",
                epochs=30,
                time_budget=4,
                lr=2e-5,
                stride=2,
                batch=4,
                accum=4,
                seed=1,
                restarts=2,
            )
    else:
        main()      # CLI: python finetune_fno_kaggle.py --data ... --ckpt ...
