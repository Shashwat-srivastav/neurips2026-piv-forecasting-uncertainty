#!/usr/bin/env python3
"""Operator boosting on the frozen RealPDE FNO. For Kaggle GPU.

WHY THIS INSTEAD OF FINE-TUNING
-------------------------------
Two full fine-tunes both made the model WORSE on held-out FINAL, in the same way:

    run 1 (w_rel 0.37)   rel -0.896  tke +3.010  mvpe -0.466   FINAL sum -0.858
    run 2 (w_rel 0.70)   rel -0.269  tke +2.245  mvpe -0.242   still negative

Both sold rel_l2 and mvpe to buy tke -- and on the live board rel_l2 transfers at
+45% while tke transferred at -31% (sign flip). Re-weighting the loss reduced the
damage but did not change the shape, because rel_l2 at 96 has no cheap gradient
left and tke at 79.5 does. No weighting fixes that; the STRUCTURE has to.

Operator boosting (arXiv 2606.17460) freezes the base and fits a tiny operator to
its residual, combining additively with a shrinkage coefficient chosen on
validation. Two properties matter here:

  1. eta = 0 is IN the candidate set, so a stage that does not help is rejected.
     Selection is on the SUBSCORE SUM, not squared loss, so a stage that trades
     rel_l2 for tke is rejected even if it lowers the training objective.
  2. the corrector's output layer is ZERO-INITIALISED, so at step 0 the ensemble
     is bit-identical to the stock FNO and training starts from it exactly.

Together those mean the worst case is "no change", not "worse". That is the
whole point: every previous attempt could degrade the base, and did.

THE CORRECTOR
-------------
A small 2-D spectral operator written here rather than pulled from the kit, so it
has no dependency beyond torch and we control every line. Time is folded into
channels (20 x 3 = 60), so it is a 2-D rFFT over the 32x64 grid -- no 3-D FFT and
no fp16 FFT, which is what broke the first fine-tune on a T4.

USAGE (Kaggle cell)
-------------------
    import boost_operator_kaggle as B
    B.run(stages=3, epochs=12)

Saves `boost_stages.pth` (one entry per ACCEPTED stage; a run that accepts
zero stages saves nothing -- ship the unmodified FNO). `submission.py` reads
either that file or a legacy single-stage `corrector.pth`, trying the latter
name first, so no renaming is required to ship this output directly.

Augmentation (`noise_std`, `amp_jitter`) is input-only, applied to the
corrector's own input only -- see `augment_input()` in
`finetune_fno_kaggle.py` for what's offered, what's deliberately not, and why.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# __file__ does not exist when this is pasted into a notebook cell.
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
for _p in (HERE, "/kaggle/working", "."):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the already-verified pieces: Source (npz layouts, lengths, name stems),
# the three metrics (checked against scoring.py to 1e-7), the normalizer
# constants and the split definitions. Duplicating them would be a bug source.
_NEEDED = ("T_IN", "T_OUT", "H", "W", "MEAN_IN", "STD_IN", "MEAN_TGT", "STD_TGT",
           "SELECT", "FINAL", "Source", "Windows", "rel_l2", "tke_rel",
           "mvpe_rel", "score_error", "sensitivity", "TRUE_W", "build",
           "forward_physical", "CONFIG", "_autodiscover", "augment_input")
_ft = None

# 1. a real module, if the file is on disk
try:
    import finetune_fno_kaggle as _ft            # noqa: F401
except ImportError:
    pass

# 2. the notebook kernel itself. If the fine-tune file was pasted and run in an
#    earlier cell, every name we need is already defined in __main__ -- no file,
#    no re-upload, and it is the SAME code that produced the runs above.
if _ft is None or any(not hasattr(_ft, n) for n in _NEEDED):
    _main = sys.modules.get("__main__")
    if _main is not None and all(hasattr(_main, n) for n in _NEEDED):
        _ft = _main
        print("reusing the helpers already defined in this notebook", flush=True)

# 3. the file somewhere under /kaggle
if _ft is None or any(not hasattr(_ft, n) for n in _NEEDED):
    _hit = (glob.glob("/kaggle/**/finetune_fno_kaggle.py", recursive=True)
            + glob.glob("./finetune_fno_kaggle.py"))
    if _hit:
        import importlib.util as _iu
        _s = _iu.spec_from_file_location("finetune_fno_kaggle", _hit[0])
        _ft = _iu.module_from_spec(_s)
        sys.modules["finetune_fno_kaggle"] = _ft
        _s.loader.exec_module(_ft)
        print(f"loaded helpers from {_hit[0]}", flush=True)

if _ft is None or any(not hasattr(_ft, n) for n in _NEEDED):
    _have = [] if _ft is None else [n for n in _NEEDED if hasattr(_ft, n)]
    raise ImportError(
        "Could not find the helper definitions (data loader, the three metrics "
        "verified against scoring.py, normalizer constants).\n"
        f"  found so far: {_have}\n"
        "Fix: run the finetune_fno_kaggle.py cell FIRST in this same notebook -- "
        "just executing it defines everything this needs. Alternatively write it "
        "to disk with:\n"
        "    %%writefile finetune_fno_kaggle.py\n"
        "    <paste that file>")
(T_IN, T_OUT, H, W, MEAN_IN, STD_IN, MEAN_TGT, STD_TGT, SELECT, FINAL,
 Source, Windows, rel_l2, tke_rel, mvpe_rel, score_error, sensitivity,
 TRUE_W, build, forward_physical, CONFIG, _autodiscover, augment_input) = (
    getattr(_ft, n) for n in _NEEDED)

ETA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 1.0, 1.2,
            1.4, 1.6, 1.8, 2.0, 2.5, 3.0]


# ------------------------------------------------------------------ corrector
class SpectralConv2d(nn.Module):
    """Truncated-mode spectral convolution over the two spatial axes."""

    def __init__(self, cin, cout, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        s = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(s * torch.rand(cin, cout, m1, m2, 2))
        self.w2 = nn.Parameter(s * torch.rand(cin, cout, m1, m2, 2))

    @staticmethod
    def _mul(a, b):
        return torch.einsum("bixy,ioxy->boxy", a, torch.view_as_complex(b))

    def forward(self, x):                       # (B, C, H, W)
        b, c, h, w = x.shape
        ft = torch.fft.rfft2(x)
        out = torch.zeros(b, self.w1.shape[1], h, w // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        m1, m2 = min(self.m1, h // 2), min(self.m2, w // 2 + 1)
        out[:, :, :m1, :m2] = self._mul(ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out[:, :, -m1:, :m2] = self._mul(ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(h, w))


class TinyOperator(nn.Module):
    """Small FNO-style corrector. Time is folded into channels: 20 x 3 = 60.

    The final projection is ZERO-INITIALISED, so forward() returns exactly zero
    before any training step and the boosted ensemble starts bit-identical to
    the frozen base.
    """

    def __init__(self, cio=T_IN * 3, width=24, modes=(12, 12), layers=2):
        super().__init__()
        self.lift = nn.Conv2d(cio, width, 1)
        self.spec = nn.ModuleList(
            [SpectralConv2d(width, width, *modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj1 = nn.Conv2d(width, width, 1)
        self.proj2 = nn.Conv2d(width, cio, 1)
        nn.init.zeros_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x):                       # (N, T, H, W, C) -> same
        n, t, h, w, c = x.shape
        y = x.permute(0, 1, 4, 2, 3).reshape(n, t * c, h, w)
        y = self.lift(y)
        for s, p in zip(self.spec, self.pw):
            y = torch.nn.functional.gelu(s(y) + p(y))
        y = self.proj2(torch.nn.functional.gelu(self.proj1(y)))
        return y.reshape(n, t, c, h, w).permute(0, 1, 3, 4, 2)


# ------------------------------------------------------------------ utilities
@torch.no_grad()
def base_predict(model, X, dev, mi, si, mt, st, batch=8):
    """Frozen base predictions, cached once as fp16 so epochs do not recompute."""
    out = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch]).to(dev).float()
        p = forward_physical(model, xb, mi, si, mt, st)
        p = torch.cat([p[..., :2], torch.zeros_like(p[..., 2:])], -1)
        out.append(p.half().cpu())
    return torch.cat(out)


def subscores(p, y):
    return dict(rel=score_error(rel_l2(p, y).mean().item()),
                tke=score_error(tke_rel(p, y).mean().item()),
                mvpe=score_error(mvpe_rel(p, y).mean().item()))


@torch.no_grad()
def eval_ensemble(corr, eta, Xc, Bc, Yc, dev, batch=8):
    """Score G0 + eta*H on a cached pool. eta = 0 must reproduce the base."""
    R, K, M = [], [], []
    for i in range(0, len(Xc), batch):
        xb = torch.as_tensor(Xc[i:i + batch]).to(dev).float()
        yb = torch.as_tensor(Yc[i:i + batch]).to(dev).float()
        p = torch.as_tensor(Bc[i:i + batch]).to(dev).float()
        if eta != 0.0 and corr is not None:
            d = corr(xb)
            p = p + eta * torch.cat([d[..., :2], torch.zeros_like(d[..., 2:])], -1)
        R.append(rel_l2(p, yb)); K.append(tke_rel(p, yb)); M.append(mvpe_rel(p, yb))
    r, k, m = (torch.cat(z).mean().item() for z in (R, K, M))
    s = dict(rel=score_error(r), tke=score_error(k), mvpe=score_error(m), err=(r, k, m))
    s["sum"] = s["rel"] + s["tke"] + s["mvpe"]
    return s


def true_composite(s):
    """d(composite)/d(score_k) weighted sum -- see TRUE_W in finetune_fno_kaggle.py.
    Comparable ONLY between two `s` dicts scored the same way (same stride,
    same split); never compare against a naive-sum number from elsewhere."""
    return sum(TRUE_W[k] * s[k] for k in TRUE_W)


# stride=20 was the same under-determination bug that cost the full fine-tune
# a week (see finetune_fno_kaggle.py): fewer windows to fit a model against.
# The corrector is far smaller (668k vs the FNO's 50.36M params) so it is less
# exposed, but there is no reason to pay any of that risk when stride=2 is
# just as cheap to cache and the eta-grid rejection (eta=0 in ETA_GRID) already
# protects against a bad stage either way.
def run(stages=3, epochs=12, batch=8, lr=3e-3, width=24, modes=12,
        stride=2, w_rel=None, w_tke=None, w_mvpe=None, workers=0,
        noise_std=0.02, amp_jitter=0.0, pool_file=None, **kw):
    cfg = dict(CONFIG)
    cfg.update(kw)
    if not os.path.exists(cfg["data"]) or not os.path.exists(cfg["ckpt"]):
        _autodiscover()
        cfg.update(CONFIG)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}  device {dev}", flush=True)
    print(f"augmentation: noise_std {noise_std}  amp_jitter {amp_jitter}"
          + ("  -- VERIFY amp_jitter against the FAQ's augmentation boundary "
             "before trusting a run that used it" if amp_jitter > 0 else ""),
          flush=True)

    src = Source(cfg["data"])
    held = set(SELECT) | set(FINAL)
    # 27 Aug: prefer the explicit, duplicate-excluded pool
    # (train_pool_v12_3.json, the SAME 67-trajectory list finetune_fno_kaggle.py's
    # Experiment 2 trained on) over deriving "everything not held out" from
    # src.items -- that derivation does not know 7575_0 is a byte-identical
    # duplicate of 6300_0 (CAMPAIGN_REPORT.md sec.5) and would silently
    # include it.
    if pool_file is None:
        for cand in ("/kaggle/working/train_pool_v12_3.json",
                     os.path.join(HERE, "train_pool_v12_3.json"),
                     "train_pool_v12_3.json"):
            if os.path.exists(cand):
                pool_file = cand
                break
    if pool_file is not None and os.path.exists(pool_file):
        names = json.load(open(pool_file))["pool"]
        names = [n for n in names if n not in held]
        print(f"training pool: {len(names)} trajectories from {pool_file}", flush=True)
    else:
        names = [n for n, re, aoa in src.items if n not in held]
        print(f"training pool: {len(names)} trajectories, derived as "
              f"all-minus-held (WARNING: no pool_file found -- this may "
              f"include 7575_0, a known duplicate of 6300_0)", flush=True)
    tr = Windows(src, names, stride)
    se = Windows(src, SELECT, stride)
    fi = Windows(src, FINAL, stride)

    # This script caches (X, base-prediction, Y) for every window ONCE so that
    # `epochs` of corrector training do not re-pay the frozen FNO's forward
    # pass -- that is the whole performance idea. But it means the cache is
    # necessarily one resident tensor per split, unlike finetune_fno_kaggle.py
    # which can afford to slice lazily. Estimate the footprint BEFORE
    # allocating and refuse past a safety cap, rather than finding out from a
    # dead kernel -- which is exactly how stride=2 (this file's own default,
    # copied from finetune_fno_kaggle.py without checking ITS footprint) cost
    # a Kaggle T4 instance its RAM on 2026-08-13.
    n_windows = len(tr) + len(se) + len(fi)
    bytes_per_window = (T_IN + 2 * T_OUT) * H * W * 3 * 2   # x + y + B, all fp16
    est_gb = n_windows * bytes_per_window / 1e9
    frame_gb = sum(f.nbytes for f in tr.frames.values()) / 1e9
    print(f"windows: train {len(tr)}  select {len(se)}  final {len(fi)}"
          f"   (cached frames {frame_gb:.2f} GB, projected pool cache "
          f"{est_gb:.2f} GB)", flush=True)
    SAFETY_GB = 8.0
    if est_gb > SAFETY_GB:
        raise MemoryError(
            f"projected pool cache is {est_gb:.2f} GB, over the {SAFETY_GB:.0f} GB "
            f"safety cap -- this is the same OOM shape that killed a Kaggle T4 "
            f"kernel at stride=2 in finetune_fno_kaggle.py on 2026-08-13. Raise "
            f"--stride (the corrector is 668k params, far less exposed to "
            f"under-determination than the 50.36M-param full fine-tune this "
            f"default was copied from), or raise SAFETY_GB here if you have "
            f"verified headroom on your instance.")

    base, _ = build(cfg["kit"], cfg["ckpt"], dev)
    base = base.float().eval()
    for p in base.parameters():
        p.requires_grad_(False)
    mi, si, mt, st = (torch.from_numpy(z).to(dev)
                      for z in (MEAN_IN, STD_IN, MEAN_TGT, STD_TGT))

    print("caching frozen base predictions ...", flush=True)
    pools = {}
    for nm, ds in (("train", tr), ("select", se), ("final", fi)):
        # Streamed through the frozen base in DataLoader-sized batches rather
        # than materialising a dense fp32 X first (torch.from_numpy(ds.x...)
        # no longer exists -- ds is the lazy Windows above, and building a
        # dense array here would be the second copy of the exact bug just
        # fixed in finetune_fno_kaggle.py's Windows).
        dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=0)
        Xs, Ys, Bs = [], [], []
        for xb, yb in dl:
            Xs.append(xb.half())
            Ys.append(yb.half())
            Bs.append(base_predict(base, xb, dev, mi, si, mt, st))
        cat = lambda zs: torch.cat(zs) if zs else torch.zeros(0)
        X, Y, B = cat(Xs), cat(Ys), cat(Bs)
        pools[nm] = (X, B, Y)
        print(f"  {nm:7} {tuple(B.shape)}", flush=True)
    del base
    if dev == "cuda":
        torch.cuda.empty_cache()

    s0 = eval_ensemble(None, 0.0, *pools["select"], dev)
    f0 = eval_ensemble(None, 0.0, *pools["final"], dev)
    print(f"\nFROZEN BASE  SELECT rel {s0['rel']:.3f} tke {s0['tke']:.3f} "
          f"mvpe {s0['mvpe']:.3f} sum {s0['sum']:.3f}", flush=True)
    print(f"             FINAL  rel {f0['rel']:.3f} tke {f0['tke']:.3f} "
          f"mvpe {f0['mvpe']:.3f} sum {f0['sum']:.3f}", flush=True)
    print("  (both computed at stride=2 for pool-caching reasons -- NOT "
          "comparable to holdout_v12_3_stock_scores.json's stride=20 figures; "
          "rescore with score_boosted_ckpt.py before deciding to submit)\n",
          flush=True)

    # Loss weights = TRUE_W (composite value per unit of THIS metric's score)
    # x sensitivity(e) (how fast that score moves per unit of error), evaluated
    # at the frozen base's own SELECT errors -- the exact same chain rule
    # finetune_fno_kaggle.py uses (see its TRUE_W comment). The corrector's old
    # hardcoded defaults (w_rel=0.70/w_tke=0.05/w_mvpe=0.25) predate the 22 Aug
    # fix and are not this computation; keep them only as an explicit override.
    er, ek, em = s0["err"]
    wr = w_rel if w_rel is not None else TRUE_W["rel"] * sensitivity(er)
    wk = w_tke if w_tke is not None else TRUE_W["tke"] * sensitivity(ek)
    wm = w_mvpe if w_mvpe is not None else TRUE_W["mvpe"] * sensitivity(em)
    _tot = wr + wk + wm
    w_rel, w_tke, w_mvpe = wr / _tot, wk / _tot, wm / _tot
    print(f"loss weights: rel {w_rel:.3f}  tke {w_tke:.3f}  mvpe {w_mvpe:.3f}"
          f"   (TRUE_W x d(score)/d(err) at the frozen base's SELECT errors)",
          flush=True)

    Xtr, Btr, Ytr = pools["train"]
    accepted, log = [], []
    cur_sel, cur_fin = s0, f0

    for stage in range(stages):
        corr = TinyOperator(width=width, modes=(modes, modes)).to(dev)
        npar = sum(p.numel() for p in corr.parameters())
        print(f"--- stage {stage}: corrector {npar:,} params ---", flush=True)

        # zero-init check: the ensemble must start EXACTLY at the base
        chk = eval_ensemble(corr, 1.0, *pools["select"], dev)
        # cur_sel is recomputed from the folded pool below, so both sides are
        # measured the same way; the tolerance covers fp16 storage of the
        # folded base (observed 2.3e-5 on a sum of ~275).
        assert abs(chk["sum"] - cur_sel["sum"]) < 1e-3, (
            f"zero-init broken: {chk['sum']} vs {cur_sel['sum']}")
        print(f"  zero-init verified: eta=1 reproduces the base exactly", flush=True)

        opt = torch.optim.AdamW(corr.parameters(), lr=lr, weight_decay=1e-6)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        idx = np.arange(len(Xtr))
        for ep in range(epochs):
            corr.train()
            np.random.shuffle(idx)
            tot, nb, t0 = 0.0, 0, time.perf_counter()
            for i in range(0, len(idx), batch):
                j = idx[i:i + batch]
                xb = Xtr[j].to(dev).float()
                yb = Ytr[j].to(dev).float()
                pb = Btr[j].to(dev).float()
                # pb is the frozen base's cached output on the CLEAN input --
                # augmenting only what the corrector itself sees still targets
                # the true residual (Y - base(clean_x)); re-running the frozen
                # base per-batch to match a noisy input would defeat the whole
                # point of caching it once. See augment_input()'s docstring in
                # finetune_fno_kaggle.py for what's offered and why.
                d = corr(augment_input(xb, noise_std, amp_jitter))
                p = pb + torch.cat([d[..., :2], torch.zeros_like(d[..., 2:])], -1)
                loss = (w_rel * rel_l2(p, yb).mean()
                        + w_tke * tke_rel(p, yb).mean()
                        + w_mvpe * mvpe_rel(p, yb).mean())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(corr.parameters(), 1.0)
                opt.step()
                tot += loss.item(); nb += 1
            sched.step()
            s = eval_ensemble(corr, 1.0, *pools["select"], dev)
            print(f"  ep {ep:2d} loss {tot/max(nb,1):.4f}  SELECT@eta1 "
                  f"rel {s['rel']:.3f} tke {s['tke']:.3f} mvpe {s['mvpe']:.3f} "
                  f"sum {s['sum']:.3f} ({s['sum']-cur_sel['sum']:+.3f})  "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

        # ---- shrinkage chosen on SELECT, with eta = 0 in the grid ----------
        # Selection criterion is TRUE-composite (rel .649/tke .131/mvpe .160),
        # not naive equal-weight sum -- maximising naive sum is exactly the
        # "buy tke, sell rel" trap that made every full fine-tune attempt this
        # campaign look like a win locally and lose live (CAMPAIGN_REPORT.md
        # sec.14.1). eta=0 (REJECT) stays in the grid as the TRUE-composite of
        # the unmodified base, so a stage that only helps naive sum is refused.
        print("  eta grid on SELECT, ranked by TRUE-composite "
              "(0.0 means REJECT the stage):", flush=True)
        best = (true_composite(cur_sel), 0.0, cur_sel)
        for eta in ETA_GRID:
            s = eval_ensemble(corr, eta, *pools["select"], dev)
            tc = true_composite(s)
            mark = ""
            if tc > best[0]:
                best = (tc, eta, s); mark = "  <-"
            print(f"    eta {eta:4.2f}  rel {s['rel']:.3f} tke {s['tke']:.3f} "
                  f"mvpe {s['mvpe']:.3f} sum {s['sum']:.3f} "
                  f"TRUE-comp {tc:.3f}{mark}", flush=True)
        _, eta, s_sel = best
        if eta == 0.0:
            print(f"  stage {stage} REJECTED (no eta beat the current ensemble)\n",
                  flush=True)
            log.append(dict(stage=stage, eta=0.0, rejected=True))
            break
        accepted.append((corr, eta))
        cur_sel = s_sel
        cur_fin = eval_ensemble(corr, eta, *pools["final"], dev)
        print(f"  stage {stage} ACCEPTED at eta {eta:.2f}\n", flush=True)
        log.append(dict(stage=stage, eta=eta, rejected=False,
                        select={k: s_sel[k] for k in ("rel", "tke", "mvpe", "sum")}))
        # residual for the next stage: fold this stage into the cached base
        with torch.no_grad():
            for nm in ("train", "select", "final"):
                Xc, Bc, Yc = pools[nm]
                new = []
                for i in range(0, len(Xc), batch):
                    xb = Xc[i:i + batch].to(dev).float()
                    d = corr(xb)
                    d = torch.cat([d[..., :2], torch.zeros_like(d[..., 2:])], -1)
                    new.append((Bc[i:i + batch].to(dev).float() + eta * d).half().cpu())
                pools[nm] = (Xc, torch.cat(new), Yc)
        # Re-measure from the folded pools: the fp16 round-trip means the
        # folded ensemble is not bit-identical to the fp32 evaluation above,
        # and the next stage's zero-init check must compare against what the
        # next stage will actually start from.
        cur_sel = eval_ensemble(None, 0.0, *pools["select"], dev)
        cur_fin = eval_ensemble(None, 0.0, *pools["final"], dev)
        print(f"  folded: SELECT sum {cur_sel['sum']:.3f}   "
              f"FINAL sum {cur_fin['sum']:.3f}\n", flush=True)
        Xtr, Btr, Ytr = pools["train"]

    print("=" * 62)
    print("BOOSTED ENSEMBLE vs FROZEN BASE  (stride=2, INTERNAL -- see note below)")
    print(f"  {'':8}{'rel':>9}{'tke':>9}{'mvpe':>9}{'sum':>10}{'TRUE-comp':>11}")
    for nm, a, b in (("SELECT", s0, cur_sel), ("FINAL", f0, cur_fin)):
        print(f"  {nm:8}{'base':>9}")
        print(f"  {'':8}{a['rel']:9.3f}{a['tke']:9.3f}{a['mvpe']:9.3f}{a['sum']:10.3f}"
              f"{true_composite(a):11.3f}")
        print(f"  {'':8}{b['rel']:9.3f}{b['tke']:9.3f}{b['mvpe']:9.3f}{b['sum']:10.3f}"
              f"{true_composite(b):11.3f}")
        print(f"  {'delta':8}{b['rel']-a['rel']:+9.3f}{b['tke']-a['tke']:+9.3f}"
              f"{b['mvpe']-a['mvpe']:+9.3f}{b['sum']-a['sum']:+10.3f}"
              f"{true_composite(b)-true_composite(a):+11.3f}")
    out = cfg["out"]
    if accepted:
        torch.save({"stages": [{"eta": e, "state": c.state_dict()} for c, e in accepted],
                    "width": width, "modes": modes},
                   os.path.join(out, "boost_stages.pth"))
        print(f"\nsaved {os.path.join(out,'boost_stages.pth')} "
              f"({len(accepted)} accepted stage(s))")
        print("\nNEXT STEP -- do not trust the numbers printed above for a submit")
        print("decision. They are stride=2 (dense, overlapping windows), and on")
        print("27 Aug the plain fine-tune's stride=2 internal delta (+2.374 sum)")
        print("and its stride=20 official rescore (+1.223 sum, TRUE-composite")
        print("-0.10) disagreed on the SIGN of whether to submit. Download this")
        print("file, then on the local machine run:")
        print(f"    python score_boosted_ckpt.py {os.path.join(out,'boost_stages.pth')}")
        print("which rescores at stride=20 against holdout_v12_3.json and prints")
        print("the TRUE-composite delta against the same stock baseline. Only a")
        print("positive TRUE-composite there justifies a submission slot.")
    else:
        print("\nNO STAGE ACCEPTED -- the frozen base is the best model. "
              "Ship submission_fno.zip unchanged.")
    json.dump(log, open(os.path.join(out, "boost_log.json"), "w"), indent=1)
    return log


if __name__ == "__main__":
    run()
