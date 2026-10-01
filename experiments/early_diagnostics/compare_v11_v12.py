#!/usr/bin/env python3
"""Run v11 (flat band) and v12 (hetero, modifier off) through their ACTUAL
predict() -- the real submission.py, not a re-implementation -- on a broad
real pool (the AoA-15 cluster, 17 trajectories, ~731 windows) instead of
preflight's 2-trajectory FINAL pair (42 windows). No training happened in
either build (both are pure band-formula variants of the frozen stock FNO),
so there is no leakage concern using a broad pool here.

rel/tke/mvpe are expected BIT-IDENTICAL between the two (neither touches the
model) -- this run exists to get a statistically sturdier LOCAL sps/coverage
read than the tiny FINAL pair gives. Per section 1 of the masterplan: this is
a sanity check, not a selector -- local sps has a documented history of not
predicting live (2.2% transfer, "no combination... reproduces the board's
measured T"). Sequential, not concurrent, given tonight's memory state.
"""
import gc
import importlib.util
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'src', d) for d in ('training', 'evaluation', 'splits')]
SC = os.path.join(REPO, "realpde_t1_starting_kit_v9", "realpde_t1_starting_kit_v9", "scoring.py")
ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
EXCLUDE = {"7575_0"}

sp = importlib.util.spec_from_file_location("sc9", SC)
sc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(sc)


def windows(stride=40):
    files = sorted(f for f in os.listdir(ROOT) if f.endswith("_15.h5")
                   and f[:-3] not in EXCLUDE)
    X, Y = [], []
    for fn in files:
        with h5py.File(os.path.join(ROOT, fn)) as f:
            u = np.asarray(f["u"][:, ::2, ::2], np.float32)
            v = np.asarray(f["v"][:, ::2, ::2], np.float32)
        fr = np.stack([u, v, np.zeros_like(u)], -1)
        for i in range(0, len(fr) - 40 + 1, stride):
            X.append(fr[i:i + 20]); Y.append(fr[i + 20:i + 40])
    return np.stack(X), np.stack(Y)


def run_build(stage_dir, tag, X, Y):
    print(f"\n=== {tag} ({stage_dir}) ===", flush=True)
    # fresh import each time -- clears the previous build's module state
    for m in list(sys.modules):
        if m == "submission":
            del sys.modules[m]
    sys.path.insert(0, stage_dir)
    import submission as S

    P, L, U = [], [], []
    for i in range(0, len(X), 5):
        chunk = X[i:i + 5]
        o = S.predict(chunk)
        P.append(np.asarray(o["prediction"], np.float32))
        L.append(np.asarray(o["lower"], np.float32))
        U.append(np.asarray(o["upper"], np.float32))
    P = np.concatenate(P); L = np.concatenate(L); U = np.concatenate(U)

    rel = sc.score_error(float(sc.rel_l2_per_sample(P, Y, 2).mean()))
    tke = sc.score_error(float(sc.tke_rel_l2_per_sample(P, Y, 2).mean()))
    mvpe = sc.score_error(float(sc.mvpe_rel_l2_per_sample(P, Y).mean()))
    w, cov = sc.aggregate_sps(P, Y, 2, L, U)
    sps = sc.score_sps(w)
    print(f"  windows: {len(X)}")
    print(f"  rel {rel:.3f}  tke {tke:.3f}  mvpe {mvpe:.3f}  "
          f"sps {sps:.3f}  coverage {cov:.4f}", flush=True)

    sys.path.remove(stage_dir)
    del S, sys.modules["submission"]
    gc.collect()
    return dict(rel=rel, tke=tke, mvpe=mvpe, sps=sps, coverage=cov)


print("building the AoA-15 pool (17 trajectories) ...", flush=True)
X, Y = windows(stride=60)
print(f"total windows: {len(X)}", flush=True)

r11 = run_build(os.path.join(HERE, ".dist_flatband"), "v11 flat-band", X, Y)
r12 = run_build(os.path.join(HERE, ".dist_hetero_nomod"), "v12 hetero-no-modifier", X, Y)

print("\n" + "=" * 60)
print(f"{'':10}{'rel':>9}{'tke':>9}{'mvpe':>9}{'sps':>9}{'coverage':>10}")
print(f"{'v11':10}{r11['rel']:9.3f}{r11['tke']:9.3f}{r11['mvpe']:9.3f}"
      f"{r11['sps']:9.3f}{r11['coverage']:10.4f}")
print(f"{'v12':10}{r12['rel']:9.3f}{r12['tke']:9.3f}{r12['mvpe']:9.3f}"
      f"{r12['sps']:9.3f}{r12['coverage']:10.4f}")
print(f"{'delta':10}{r11['rel']-r12['rel']:+9.3f}{r11['tke']-r12['tke']:+9.3f}"
      f"{r11['mvpe']-r12['mvpe']:+9.3f}{r11['sps']-r12['sps']:+9.3f}"
      f"{r11['coverage']-r12['coverage']:+10.4f}")
print("\nrel/tke/mvpe should be ~0 delta (neither build touches the model) --")
print("any nonzero delta there would indicate a bug in one of the builds.")
