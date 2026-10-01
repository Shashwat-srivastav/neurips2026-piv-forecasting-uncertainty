#!/usr/bin/env python3
"""Fine-tune CNO instead of FNO, reusing finetune_fno_kaggle.py's harness
unchanged (`build()` and `forward_physical()` are already architecture-
agnostic -- `load_baseline(ckpt)` auto-detects model type from the checkpoint
itself, and the TRUE_W loss weighting operates on generic rel/tke/mvpe error
values, not FNO internals). Only the checkpoint path differs.

WHY CNO, and why this is a genuinely new test (not a repeat of the campaign's
closed FNO-fine-tune directions): the campaign only ever compared the
organizers' STOCK sim_real_cno.pth against stock FNO (CNO lost on all three
metrics -- CAMPAIGN_REPORT.md sec 6.3) -- nobody has fine-tuned CNO the
disciplined way (train_pool_v12_3.json, TRUE_W loss, honest holdout). A
different architecture may not hit the same "buy tke, sell rel" gradient-
dominance wall FNO's four fine-tune attempts all hit -- that failure mode was
diagnosed as a property of the LOSS LANDSCAPE FNO's stock checkpoint sits in,
not proven to be architecture-independent.

CNO is also much cheaper to iterate on: 7,960,467 params / 31.8 MB fp32
(vs FNO's 50,357,955 / 403 MB) -- no fp16 packing needed for the 256 MB
submission cap, and epochs should run faster.

DEPENDENCY NOTE (submission time only, not training time): CNO's own code
unconditionally imports `requests` via `CNO_libs/dnnlib/util.py`, which is
NOT in the evaluation container's package set (confirmed missing 27 Aug via a
real import-blocking test; scipy/h5py/einops/matplotlib/pandas were already
known-absent from the starting kit's own README). requests IS present on
Kaggle itself, so this does not block TRAINING -- only building/submitting a
CNO zip needs the stub in `_vendor_requests_stub.py` (see build script) doing
`sys.modules['requests'] = <stub>` BEFORE any `rpde_baselines` CNO import.
`open_url()`, the only function that touches `requests` for real, is never
reached at inference time with a local checkpoint -- it is a StyleGAN-style
remote-weight downloader nothing here calls.

USAGE (Kaggle cell, AFTER pasting+running finetune_fno_kaggle.py in an
earlier cell so its helpers are importable):
    import finetune_cno_kaggle as C
    C.run_cno1()          # Experiment CNO-1: clean holdout from sim_cno.pth
"""
import glob
import os
import sys

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
for _p in (HERE, "/kaggle/working", "."):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ft = None
try:
    import finetune_fno_kaggle as _ft            # noqa: F401
except ImportError:
    pass
if _ft is None or not hasattr(_ft, "run"):
    _main = sys.modules.get("__main__")
    if _main is not None and hasattr(_main, "run"):
        _ft = _main
        print("reusing finetune_fno_kaggle's helpers already defined in this notebook",
              flush=True)
if _ft is None or not hasattr(_ft, "run"):
    _hit = (glob.glob("/kaggle/**/finetune_fno_kaggle.py", recursive=True)
            + glob.glob("./finetune_fno_kaggle.py"))
    if _hit:
        import importlib.util as _iu
        _s = _iu.spec_from_file_location("finetune_fno_kaggle", _hit[0])
        _ft = _iu.module_from_spec(_s)
        sys.modules["finetune_fno_kaggle"] = _ft
        _s.loader.exec_module(_ft)
        print(f"loaded helpers from {_hit[0]}", flush=True)
if _ft is None or not hasattr(_ft, "run"):
    raise ImportError(
        "Could not find finetune_fno_kaggle's run()/CONFIG -- paste and run "
        "finetune_fno_kaggle.py in an earlier cell first (just executing it "
        "defines everything; its own bottom if __name__ block runs "
        "Experiment 3 when done that way, which is fine -- this script never "
        "calls that, only imports the helpers above it).")


def _find_cno_ckpt():
    """Same style as finetune_fno_kaggle.py's own _autodiscover(), for CNO."""
    for pat in ("**/sim_cno.pth",):
        hits = glob.glob(f"/kaggle/input/{pat}", recursive=True)
        if hits:
            return hits[0], os.path.dirname(hits[0])
    return None, None


def run_cno1(**kw):
    """Experiment CNO-1: the ONLY valid holdout test per the campaign's evaluation protocol
    sec.0 -- start from sim_cno.pth (sim-pretrained ONLY, never seen ANY real
    trajectory, unlike sim_real_cno.pth which the organizers already
    fine-tuned on all of train_real). If this beats stock FNO's true
    generalization numbers (rel 94.168 / tke 74.026 / mvpe 92.836) on the
    untouched FINAL pool, that is unambiguous evidence CNO generalizes better
    here -- not a memorization artifact.

    Mirrors finetune_fno_kaggle.py's own Experiment 1 (same starting-point
    logic, same pool, same TRUE_W loss) so the comparison is apples-to-apples
    against that already-run, already-failed FNO experiment.
    """
    ckpt, kit = _find_cno_ckpt()
    cfg = dict(
        epochs=40,          # CNO is ~6.3x fewer params than FNO -- afford more epochs
        lr=1e-4,
        stride=2,
        # FP32 batch=64 OOMs before backward; AMP batch=32 still reaches
        # 13.84 GiB and OOMs on a 320 MiB conv3d buffer on a 14.56 GiB T4
        # (29 Aug). Batch 16 is the first configuration with real headroom.
        # Accumulation preserves a 128-sample optimizer step without retaining
        # batch-128 activations. CNO's BatchNorm still sees microbatch-16 stats.
        batch=16,
        accum=8,
        # CNO batch=64 was already proven to fit during no_grad SELECT scoring.
        # Keep that throughput for evaluation; only the training microbatch must
        # be small enough to retain backward activations.
        eval_batch=64,
        time_budget=4,
        steps_per_epoch=400,
        amp=True,
        pool_file="/kaggle/working/train_pool_v12_3.json",
    )
    if ckpt:
        cfg["ckpt"] = ckpt
        cfg["kit"] = kit
    cfg.update(kw)
    print(f"Experiment CNO-1: ckpt={cfg.get('ckpt', '<using CONFIG default -- FIX THIS, '
          'that default is sim_real_fno.pth, not CNO>')}", flush=True)
    return _ft.run(**cfg)


if __name__ == "__main__":
    run_cno1()
