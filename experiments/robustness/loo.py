"""Leave-one-AoA-out and leave-one-Re-band-out transfer of v17's fitted tables (official FNO, proxy AT, per-window bins).

For each fold (5 AoA folds, 3 Reynolds bands): the held-out trajectories are the evaluation set (windows at stride 60).
  transfer : tables fitted on 8 trajectories drawn from the other folds (2 per AoA from 4 AoA strata), 3 seeds
  in-fold  : tables fitted on the held-out trajectories themselves (upper bound; in-sample)
Each is scored against v8 on the held-out set under proxy AT, with the proxy scale fitted on that set (as K9 does for
FINAL). Coverage is the fraction of scored elements inside the interval (optimum 0.70-0.80).
Caveat: the official FNO very likely trained on every trajectory, as in K9; the proxy inflates its errors to live level.
Usage: python loo.py [fold ...]   (default: all)   -> loo.json, updated after every fold
"""
import gc, json, os, sys, time
import numpy as np
from lib2 import prep, fit_recipe, outputs, score, delta   # noqa: E402
from loo_lib import names, aoa, re_, RE_BANDS, load_set   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTF = os.path.join(HERE, "loo.json")
ALL = names()
FOLDS = {f"AoA {a}": [n for n in ALL if aoa(n) == a] for a in sorted({aoa(n) for n in ALL})}
FOLDS.update({b: [n for n in ALL if lo <= re_(n) < hi] for b, (lo, hi) in RE_BANDS.items()})
SEEDS, EVAL_STRIDE = (0, 1, 2), 60


def fit_draw(pool, seed):
    rng = np.random.default_rng(seed)
    strata = sorted({aoa(n) for n in pool})
    use = sorted(rng.choice(strata, min(4, len(strata)), replace=False).tolist())
    pick = []
    for a in use:
        c = sorted(n for n in pool if aoa(n) == a)
        pick += rng.choice(c, min(2, len(c)), replace=False).tolist()
    return [str(p) for p in pick]


def summary(F, Yt, v8, tabs):
    _, out, c, h = outputs(F, tabs)
    s = score(out, Yt, c, h); d = delta(s, v8)
    return dict(comp=round(float(d["comp"]), 4), sps=round(float(d["sps"]), 3), tke=round(float(d["tke"]), 3),
                rel=round(float(d["rel"]), 3), cov=round(float(s["cov"]), 4))


res = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
for fold in (sys.argv[1:] or list(FOLDS)):
    t0 = time.time()
    held = FOLDS[fold]; pool = [n for n in ALL if n not in held]
    X, Y, P, _ = load_set(held, EVAL_STRIDE)
    F = prep(X, Y, P, only=("AT",)); Yt, Pt = F["pr"]["AT"]
    v8 = score(Pt, Yt, Pt, F["h8"])
    row = dict(held_out=held, n_traj=len(held), n_windows=int(len(X)), k_eval=[round(float(k), 3) for k in F["k"]],
               v8_cov=round(float(v8["cov"]), 4))
    S = prep(X, Y, P); row["in_fold"] = summary(F, Yt, v8, fit_recipe(S)); del S; gc.collect()
    row["transfer"] = []
    for seed in SEEDS:
        pick = fit_draw(pool, seed + 1000 * list(FOLDS).index(fold))
        Xf, Yf, Pf, _ = load_set(pick, 20)
        S = prep(Xf, Yf, Pf); tabs = fit_recipe(S); del S, Xf, Yf, Pf; gc.collect()
        row["transfer"].append(dict(seed=seed, fit=pick, **summary(F, Yt, v8, tabs)))
    tc = [t["comp"] for t in row["transfer"]]
    row["transfer_mean"] = round(float(np.mean(tc)), 4)
    row["kept_share"] = round(float(np.mean(tc) / row["in_fold"]["comp"]), 3) if row["in_fold"]["comp"] > 0 else None
    res[fold] = row; json.dump(res, open(OUTF, "w"), indent=1)
    print(f"{fold:10} n={len(held):2} in-fold {row['in_fold']['comp']:+.3f} (cov {row['in_fold']['cov']:.3f})  "
          f"transfer {min(tc):+.3f}..{max(tc):+.3f} mean {np.mean(tc):+.3f} "
          f"(cov {', '.join(format(t['cov'], '.3f') for t in row['transfer'])})  v8 cov {row['v8_cov']:.3f}  "
          f"({time.time() - t0:.0f}s)", flush=True)
    del F, X, Y, P, Yt, Pt; gc.collect()
