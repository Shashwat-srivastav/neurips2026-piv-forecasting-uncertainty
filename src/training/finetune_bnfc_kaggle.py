#!/usr/bin/env python3
"""BN/fc-only fine-tune, masterplan v11 section 5. For Kaggle GPU.

WHY THIS IS DIFFERENT FROM THE FOUR PRIOR FAILED FINE-TUNES
-------------------------------------------------------------
Every previous attempt trained the WHOLE model (50,357,955 params, of which
spectral_convs are 50,331,648 -- 99.9%) and every one found the same escape
hatch: buy tke by rescaling output variance, sell rel and mvpe to pay for it.
That happens because the spectral weights ARE the conditional-mean operator
(56% variance carry-through IS the rel-optimal shrinkage, verified 15 Aug) --
giving the optimizer access to them gives it access to that trade.

This freezes `spectral_convs` (and `convs`, the pointwise skip-conv beside
each spectral layer -- not on the plan's trainable list) ENTIRELY. Only
`fc0` (input lift), `fc1`/`fc2` (output projection) and the BatchNorm3d
AFFINE parameters (`bns[i].weight/bias`, NOT the running_mean/running_var
buffers) get gradients -- 448+8,320+387+2*4*64 = 9,667 trainable params out
of 50.36M, 0.019%. The loss has no tke term at all (plain rel_l2), so even
that much smaller capacity has no reward for the old trade.

Verified real attribute names against rpde_baselines/model/fno.py directly
(fc0, spectral_convs, convs, bns, fc1, fc2) before writing this -- not
assumed from the architecture summary in the campaign report.

TWO VARIANTS, decided live only (section 1: local metrics are sanity checks,
never model selectors):
    A: no augmentation
    B: spectral_augment.spectral_boost() on the input, matched to the
       ACTUAL training pool's own spectrum (not the AoA-15 pooled average --
       verified 18 Aug that using the pooled-average gain on this specific
       pool overshoots, since these trajectories are already closer to live
       than the AoA-15 average).

DATA: live_similar_pool.npy (11 AoA-15 trajectories, ranked by live-distance,
window_similarity_aoa15.npy), holdout is ONLY preflight.py's hardcoded FINAL
pair (21600_15, 25425_15) -- kept clean so the pre-submission sanity check
stays meaningful. 4 of the 11 pool trajectories are genuinely far (dist 3-5,
not "live-similar" despite being in the top-11 by rank) -- see
CAMPAIGN_REPORT.md 13.6 for the honest breakdown.

USAGE (Kaggle cell)
--------------------
    import finetune_bnfc_kaggle as F
    F.run(dry_run=True)                       # ALWAYS first
    F.run(variant="A", epochs=3)
    F.run(variant="B", epochs=3)
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import torch

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
for _p in (HERE, "/kaggle/working", "."):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_NEEDED = ("T_IN", "T_OUT", "H", "W", "MEAN_IN", "STD_IN", "MEAN_TGT", "STD_TGT",
           "Source", "Windows", "rel_l2", "score_error", "build",
           "forward_physical", "free_gpu", "CONFIG", "_autodiscover")
_ft = None
try:
    import finetune_fno_kaggle as _ft
except ImportError:
    pass
if _ft is None or any(not hasattr(_ft, n) for n in _NEEDED):
    _main = sys.modules.get("__main__")
    if _main is not None and all(hasattr(_main, n) for n in _NEEDED):
        _ft = _main
if _ft is None or any(not hasattr(_ft, n) for n in _NEEDED):
    _hit = (glob.glob("/kaggle/**/finetune_fno_kaggle.py", recursive=True)
            + glob.glob("./finetune_fno_kaggle.py"))
    if _hit:
        import importlib.util as _iu
        _s = _iu.spec_from_file_location("finetune_fno_kaggle", _hit[0])
        _ft = _iu.module_from_spec(_s); sys.modules["finetune_fno_kaggle"] = _ft
        _s.loader.exec_module(_ft)
if _ft is None or any(not hasattr(_ft, n) for n in _NEEDED):
    raise ImportError(
        "Run the finetune_fno_kaggle.py cell FIRST in this notebook -- this "
        "file reuses its data pipeline, metrics (checked against scoring.py "
        "to 1e-7), normalizer constants and free_gpu.")

(T_IN, T_OUT, H, W, MEAN_IN, STD_IN, MEAN_TGT, STD_TGT, Source, Windows,
 rel_l2, score_error, build, forward_physical, free_gpu, CONFIG,
 _autodiscover) = (getattr(_ft, n) for n in _NEEDED)

try:
    from spectral_augment import spectral_boost, AMP_GAIN_U, AMP_GAIN_V
except ImportError:
    _hit = glob.glob("/kaggle/**/spectral_augment.py", recursive=True)
    if _hit:
        import importlib.util as _iu
        _s = _iu.spec_from_file_location("spectral_augment", _hit[0])
        _sa = _iu.module_from_spec(_s); _s.loader.exec_module(_sa)
        spectral_boost, AMP_GAIN_U, AMP_GAIN_V = (_sa.spectral_boost,
                                                   _sa.AMP_GAIN_U, _sa.AMP_GAIN_V)
    else:
        raise ImportError("spectral_augment.py not found -- needed for variant B")

# The 11-trajectory live-similar pool. Only preflight.py's hardcoded FINAL
# pair is held out (see module docstring for why not the full 5-trajectory
# holdout). Positions 0-6 are genuinely close (dist 1.19-2.45); 7-10 are not
# (dist 3.0-3.9) -- included only to reach 11 per the plan, not because they
# are actually live-similar.
POOL = ["24150_15", "22875_15", "26700_15", "20325_15", "19050_15",
        "17775_15", "16500_15", "13950_15", "12675_15", "11400_15", "10125_15"]
SANITY_HOLDOUT = ["21600_15", "25425_15"]     # preflight.py's own FINAL pair
EXCLUDE = ["7575_0"]                          # not in this AoA-15 pool anyway


def augment_spectral_input(xb):
    """Batched wrapper around spectral_boost for a torch tensor (N,T,H,W,C)
    on the CPU/numpy side -- called before the batch goes to device, since
    spectral_boost is numpy/FFT-based, not a GPU op. This is training-time
    data augmentation, not part of any predict() hot path, so the host-sync
    latency rule (13.1) does not apply here."""
    x = xb.numpy()
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = spectral_boost(x[i], gain_u=AMP_GAIN_U, gain_v=AMP_GAIN_V)
    return torch.from_numpy(out)


def _trainable_params(model):
    """Freeze everything, then unfreeze ONLY fc0/fc1/fc2 and BN affine
    (weight/bias) -- NOT running_mean/running_var, which are buffers, not
    parameters, and are not touched by requires_grad either way; NOT
    spectral_convs or convs, which stay frozen throughout."""
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for name in ("fc0", "fc1", "fc2"):
        mod = getattr(model, name)
        for p in mod.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    for bn in model.bns:
        bn.weight.requires_grad_(True)
        bn.bias.requires_grad_(True)
        trainable.append(bn.weight); trainable.append(bn.bias)
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {n_train:,} / {n_total:,} params "
          f"({100*n_train/n_total:.4f}%)  -- fc0, fc1, fc2, BN affine only",
          flush=True)
    return trainable


@torch.no_grad()
def evaluate(model, loader, dev, mi, si, mt, st):
    model.eval()
    R = []
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        p = forward_physical(model, xb, mi, si, mt, st)
        p = torch.cat([p[..., :2], torch.zeros_like(p[..., 2:])], -1)
        R.append(rel_l2(p, yb))
    r = torch.cat(R).mean().item()
    return dict(rel=score_error(r), err=r)


def run(variant="A", epochs=3, batch=4, lr=1e-5, weight_decay=1e-4,
        steps_per_epoch=800, log_every=100, dry_run=False, **kw):
    assert variant in ("A", "B"), "variant must be 'A' (no aug) or 'B' (spectral aug)"
    cfg = dict(CONFIG); cfg.update(kw)
    if not os.path.exists(cfg["data"]) or not os.path.exists(cfg["ckpt"]):
        _autodiscover(); cfg.update(CONFIG)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}  device {dev}  variant {variant}", flush=True)
    if dev == "cuda":
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        free_gpu(" before load")

    src = Source(cfg["data"])
    avail = {nm for nm, _, _ in src.items}
    train_names = [n for n in POOL if n in avail]
    holdout_names = [n for n in SANITY_HOLDOUT if n in avail]
    missing = [n for n in POOL + SANITY_HOLDOUT if n not in avail]
    if missing:
        print(f"  NOT IN DATASET (skipped): {missing}", flush=True)
    print(f"train pool: {len(train_names)} trajectories  "
          f"sanity holdout: {holdout_names}", flush=True)

    tr = Windows(src, train_names, stride=2)
    ho = Windows(src, holdout_names, stride=2) if holdout_names else None
    gb = sum(f.nbytes for f in tr.frames.values()) / 1e9
    print(f"windows: train {len(tr)}"
          + (f"  sanity-holdout {len(ho)}" if ho else "")
          + f"   (cached frames {gb:.2f} GB)", flush=True)

    steps = steps_per_epoch if steps_per_epoch > 0 else len(tr) // batch
    dl = torch.utils.data.DataLoader(
        tr, batch_size=batch, num_workers=0,
        sampler=torch.utils.data.RandomSampler(
            tr, replacement=True, num_samples=steps * batch))
    dho = torch.utils.data.DataLoader(ho, batch_size=batch) if ho else None

    model, _ = build(cfg["kit"], cfg["ckpt"], dev)
    model = model.float()
    trainable = _trainable_params(model)
    mi, si = (torch.from_numpy(z).to(dev) for z in (MEAN_IN, STD_IN))
    mt, st = (torch.from_numpy(z).to(dev) for z in (MEAN_TGT, STD_TGT))

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(epochs, 1))

    if not dry_run and dho is not None:
        # Skipped entirely under dry_run -- it ran UNCONDITIONALLY in the
        # first version, defeating the purpose of a quick check (measured:
        # still executing after 10+ min on CPU at batch 2, 415 forward
        # passes over the 830-window holdout, before the actual 5-batch
        # timing test even started). The real epoch loop below still
        # evaluates it every epoch regardless.
        base = evaluate(model, dho, dev, mi, si, mt, st)
        print(f"\nSTOCK on sanity-holdout (informational only, NOT a selector): "
              f"rel {base['rel']:.3f}", flush=True)

    if dry_run:
        print("\n--- DRY RUN: one fwd+bwd, measured ---", flush=True)
        it = iter(dl); xb, yb = next(it)
        if variant == "B":
            xb = augment_spectral_input(xb)
        xb, yb = xb.to(dev), yb.to(dev)
        t0 = time.perf_counter()
        for _ in range(5):
            p = forward_physical(model, xb, mi, si, mt, st)
            p = torch.cat([p[..., :2], torch.zeros_like(p[..., 2:])], -1)
            loss = rel_l2(p, yb).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 5 * 1000
        print(f"  {ms:.0f} ms/batch -> epoch {ms*steps/1000:.0f}s -> "
              f"{epochs} epochs {ms*steps*epochs/3600000:.2f} h", flush=True)
        if dev == "cuda":
            free_gpu(" after dry run")
        return None

    t_all = time.perf_counter()
    loss_hist = []
    for ep in range(epochs):
        model.train()
        # BN layers need train() to compute batch stats, but spectral_convs
        # and convs never receive gradients regardless -- requires_grad=False
        # on their parameters is what actually prevents any weight update.
        t0, run_l, nb = time.perf_counter(), 0.0, 0
        for i, (xb, yb) in enumerate(dl):
            if variant == "B":
                xb = augment_spectral_input(xb)
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            p = forward_physical(model, xb, mi, si, mt, st)
            p = torch.cat([p[..., :2], torch.zeros_like(p[..., 2:])], -1)
            loss = rel_l2(p, yb).mean()       # NO tke term -- section 5 spec
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run_l += loss.item(); nb += 1
            if log_every > 0 and (i + 1) % log_every == 0:
                el = time.perf_counter() - t0
                print(f"    ep {ep:3d} {i+1:5d}/{steps} loss {run_l/nb:.4f} "
                      f"{el/(i+1)*1000:5.0f} ms/b  eta {(steps-i-1)*el/(i+1):4.0f}s",
                      flush=True)
        sched.step()
        epoch_loss = run_l / max(nb, 1)
        loss_hist.append(epoch_loss)
        msg = f"epoch {ep:3d} loss {epoch_loss:.4f}  {time.perf_counter()-t0:.0f}s"
        if dho is not None:
            s = evaluate(model, dho, dev, mi, si, mt, st)
            msg += f"  sanity-holdout rel {s['rel']:.3f} (info only)"
        print(msg, flush=True)

        # ABORT CONDITION (section 5): if training loss does not decrease
        # after epoch 1, stop -- do not burn the remaining epochs.
        if ep == 1 and loss_hist[1] >= loss_hist[0]:
            print(f"\nABORT: loss did not decrease after epoch 1 "
                  f"({loss_hist[0]:.4f} -> {loss_hist[1]:.4f}). Stopping.",
                  flush=True)
            break

    out = cfg.get("out", "/kaggle/working")
    fname = f"bnfc_finetune_{variant}.pth"
    torch.save({"model_state_dict": model.state_dict(), "variant": variant,
               "trainable_only": True, "loss_history": loss_hist},
              os.path.join(out, fname))
    print(f"\nsaved {os.path.join(out, fname)}", flush=True)
    print(f"total wall {(time.perf_counter()-t_all)/60:.1f} min", flush=True)
    json.dump({"variant": variant, "loss_history": loss_hist, "pool": train_names},
              open(os.path.join(out, f"bnfc_log_{variant}.json"), "w"), indent=1)
    print("\nNEXT: wrap this checkpoint with the SAME normalizer constants and "
          "SPS bands as submission_fno_eta0_v8.zip -- do not ship it bare. "
          "Decide A vs B LIVE only (section 1/6): submit whichever ships "
          "first, read rel/mvpe/tke deltas directly, board keeps the max.")
    return loss_hist


if __name__ == "__main__":
    run(variant="A", dry_run=True)
