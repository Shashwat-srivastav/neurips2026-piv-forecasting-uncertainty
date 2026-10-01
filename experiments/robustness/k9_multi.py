"""K9 over several random AoA-stratified splits (the reviewers' 'one random draw' objection).

For each seed: draw 4 AoA strata and one training-pool trajectory per stratum (same rule as cache_rand.py, seed 0 is
the original draw), cache official-FNO forecasts at stride 20 (CPU), fit the v17 recipe with per-window bins twice
(proxy k refitted on the split; SELECT's k), and score on FINAL under proxy AT against v8. The fingerprint-selected
SELECT fit gives +1.356; the share of that gain lost with a random fit is the regime-match share.
Usage: python k9_multi.py 0 1 2 3 4 5 6 7 8   -> k9_multi.json (appended per seed)
"""
import gc, json, os, sys, time
import numpy as np
from lib2 import *   # noqa: F401,F403  (prep, fit_recipe, outputs, score, delta, load, KA, KT)

HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "k9_multi.json")
SELECT_GAIN = 1.356
pool = json.load(open(os.path.join(BASE, "data/splits/train_pool_v12_3.json")))["pool"]


def draw(seed):
    rng = np.random.default_rng(seed)
    aoas = sorted({int(n.split("_")[1]) for n in pool})
    strata = sorted(rng.choice(aoas, 4, replace=False).tolist())
    return strata, [str(rng.choice(sorted(n for n in pool if int(n.split("_")[1]) == a))) for a in strata]


def cache(seed, pick):
    path = os.path.join(HERE, "cache_rand_s20.npz" if seed == 0 else f"cache_rand{seed}_s20.npz")
    if os.path.exists(path):
        d = np.load(path); return d["X"], d["Y"], d["P"]
    import h5py, torch
    sys.path.insert(0, os.path.join(HERE, "..", "interval_recipes", "v8"))
    from load_baseline import load_baseline
    root = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
    mi, si = (torch.tensor(z) for z in ([0.154960856, -0.000513992854, 0.0], [0.0968056545, 0.015960684, 1.0]))
    mt, st = (torch.tensor(z) for z in ([0.154962569, -0.000517793698, 0.0], [0.0968104079, 0.0159636438, 1.0]))
    torch.set_num_threads(6)
    # official FNO (the exp1 builds hold our Run 1 weights under the same file name)
    model, _ = load_baseline(os.path.join(HERE, "..", "..", "submissions", "v17_per_window", "sim_real_fno_fp16.pth"), device="cpu")
    model = model.eval()
    Xs, Ys = [], []
    for nm in pick:
        with h5py.File(os.path.join(root, nm + ".h5")) as f:
            u = np.asarray(f["u"][:, ::2, ::2], np.float32); v = np.asarray(f["v"][:, ::2, ::2], np.float32)
        for i in range(0, len(u) - 40 + 1, 20):
            x = np.zeros((20, 32, 64, 3), np.float32); y = np.zeros_like(x)
            x[..., 0], x[..., 1], y[..., 0], y[..., 1] = u[i:i+20], v[i:i+20], u[i+20:i+40], v[i+20:i+40]
            Xs.append(x); Ys.append(y)
    X = np.stack(Xs); Y = np.stack(Ys); P = np.zeros(X[..., :2].shape, np.float32)
    with torch.no_grad():
        for i in range(0, len(X), 5):
            xb = torch.nan_to_num(torch.from_numpy(X[i:i+5]))
            yb = model((xb - mi) / si) * st + mt
            P[i:i+5] = yb.float().numpy()[..., :2]
    X, Y = np.ascontiguousarray(X[..., :2]), np.ascontiguousarray(Y[..., :2])
    np.savez(path, X=X, Y=Y, P=P)
    del model; gc.collect()
    return X, Y, P


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
Xf, Yf, Pf, traj = load("final")
F = prep(Xf, Yf, Pf, KA["final"], KT["final"], only=("AT",)); Yt, Pt = F["pr"]["AT"]
v8 = score(Pt, Yt, Pt, F["h8"])
for seed in map(int, sys.argv[1:]):
    t0 = time.time()
    strata, pick = draw(seed)
    if any(r["pick"] == pick for key, r in res.items() if key != str(seed)):
        print(seed, "duplicate split, skipped", pick, flush=True); continue
    X, Y, P = cache(seed, pick)
    row = dict(strata=strata, pick=pick, n=int(len(X)))
    for tag, ka, kt in (("refit_k", None, None), ("select_k", KA["select"], KT["select"])):
        S = prep(X, Y, P, ka, kt); row[f"k_{tag}"] = [round(float(z), 3) for z in S["k"]]
        tabs = fit_recipe(S); del S; gc.collect()
        _, out, c, h = outputs(F, tabs)
        d = delta(score(out, Yt, c, h), v8)
        row[tag] = {k: round(float(d[k]), 4) for k in ("rel", "tke", "mvpe", "sps", "T", "comp")}
        del out, c, h; gc.collect()
    row["share_refit_k"] = round(1 - row["refit_k"]["comp"] / SELECT_GAIN, 3)
    row["share_select_k"] = round(1 - row["select_k"]["comp"] / SELECT_GAIN, 3)
    res[str(seed)] = row; json.dump(res, open(OUT, "w"), indent=1)
    print(seed, pick, "comp refit %+.3f selk %+.3f  share %.2f / %.2f  (%.0fs)" % (
        row["refit_k"]["comp"], row["select_k"]["comp"], row["share_refit_k"], row["share_select_k"], time.time() - t0), flush=True)
    del X, Y, P; gc.collect()
