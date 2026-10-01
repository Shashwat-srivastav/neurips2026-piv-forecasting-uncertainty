"""GBM predictor of log|e| (trajectory-grouped OOF on SELECT), then h = g(score) lookup or Laplace formula.
Evaluated once on FINAL under the proxy. Scale-free w.r.t. k for the score; g is fitted under proxy k."""
import sys, time, json
import xgboost as xgb
from common import *
from feats2 import chunk_feats, NAMES
rng = np.random.default_rng(0)
KMODE = sys.argv[1] if len(sys.argv) > 1 else "rel"
FEATS = sys.argv[2].split(",") if len(sys.argv) > 2 else NAMES
fi = [NAMES.index(f) for f in FEATS]

def build(role, c, sub=None):
    X, Y, P, traj = load(role)
    Fs, E, G, M = [], [], [], []
    for i in range(0, len(X), 20):
        f = chunk_feats(X[i:i+20], P[i:i+20], c)[..., fi]
        y = Y[i:i+20, ..., c]; e = np.abs(y - P[i:i+20, ..., c]); m = y != 0
        g = np.broadcast_to(np.array([hash(t) % 1000 for t in traj[i:i+20]])[:, None, None, None], m.shape)
        if sub:
            sel = m & (rng.random(m.shape) < sub)
        else:
            sel = m
        Fs.append(f[sel]); E.append(e[sel]); G.append(g[sel]); M.append(m)
    return np.concatenate(Fs), np.concatenate(E), np.concatenate(G)

def lookup_fit(s, ae, nb=40):
    edges = np.unique(np.quantile(s, np.linspace(0, 1, nb + 1)[1:-1]))
    idx = np.searchsorted(edges, s); hs = np.zeros(len(edges) + 1)
    for b in range(len(hs)):
        e = np.sort(ae[idx == b]); m = len(e)
        if m: hs[b] = e[np.argmax(np.exp(-2 * e / SIG) * np.arange(1, m + 1) / m)]
    # enforce monotone non-decreasing? keep raw but smooth by isotonic-ish cummax is wrong for saturating; keep raw
    return edges, hs

def Tval(h, ae):
    return float(np.mean(np.exp(-2 * h / SIG) * (ae <= h)))

params = dict(max_depth=7, eta=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=50, tree_method="hist", nthread=12)
res = {}
Xs_all = {}
X, Y, P, _ = load("select"); ksel = fit_k(Y, P, KMODE); del X, Y, P
X, Y, P, _ = load("final"); kfin = fit_k(Y, P, KMODE); del X, Y, P
print("k select", ksel, "k final", kfin, flush=True)
for c in (0, 1):
    t0 = time.time()
    Fs, Es, Gs = build("select", c, sub=0.25)
    Es_p = Es * ksel
    lab = np.log(Es + 1e-4)
    # OOF over the 4 SELECT trajectories
    oof = np.zeros(len(Es), np.float32)
    for g in np.unique(Gs):
        tr = Gs != g
        bst = xgb.train(params, xgb.DMatrix(Fs[tr], lab[tr]), 200)
        oof[~tr] = bst.predict(xgb.DMatrix(Fs[~tr]))
    bst = xgb.train(params, xgb.DMatrix(Fs, lab), 200)
    edges, hs = lookup_fit(oof, Es_p)
    # Laplace: b = c*exp(score)
    bb = np.exp(oof)
    best = max(((Tval(cc * bb * np.log(1 + SIG / (2 * cc * bb)), Es_p), cc) for cc in np.geomspace(0.3, 6, 40)))
    print(f"ch {'uv'[c]} OOF SELECT: lookup T {Tval(hs[np.searchsorted(edges, oof)], Es_p):.4f}  laplace T {best[0]:.4f} (c={best[1]:.2f})  [{time.time()-t0:.0f}s]", flush=True)
    del Fs, Es, Gs, oof, lab
    X, Y, P, _ = load("final"); SF, EF = [], []
    for i in range(0, len(X), 10):
        f = chunk_feats(X[i:i+10], P[i:i+10], c)[..., fi]; m = Y[i:i+10, ..., c] != 0
        SF.append(bst.predict(xgb.DMatrix(f[m]))); EF.append(np.abs(Y[i:i+10, ..., c] - P[i:i+10, ..., c])[m]); del f
    del X, Y, P
    sf = np.concatenate(SF); Ef = np.concatenate(EF); del SF, EF
    Ef_p = Ef * kfin
    hl = hs[np.searchsorted(edges, sf)]
    bf = best[1] * np.exp(sf); hL = bf * np.log(1 + SIG / (2 * bf))
    res["uv"[c]] = dict(lookup=Tval(hl, Ef_p), laplace=Tval(hL, Ef_p), n=len(Ef))
    print(f"ch {'uv'[c]} FINAL: lookup T {res['uv'[c]]['lookup']:.4f} laplace T {res['uv'[c]]['laplace']:.4f}", flush=True)
    imp = bst.get_score(importance_type="gain"); tot = sum(imp.values())
    print("  gain:", ", ".join(f"{FEATS[int(k[1:])]}={v/tot:.3f}" for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:10]))
    del Ef, sf
print(json.dumps(res))
