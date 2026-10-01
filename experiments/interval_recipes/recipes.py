"""Band recipe search on dumped scored elements. Fit on SELECT (proxy k_fit), evaluate once on FINAL (proxy k_eval)."""
import os, json, numpy as np
from common import SIG
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "dump")
LOGF = ["p1", "p2", "p5", "sl", "sl5", "d1", "dT", "gP", "aP", "ps", "oth", "pg", "gx", "axm", "pdev"]
LINF = ["t", "yy", "xx", "nb5", "slw"]
EPS = 1e-4
KEEP = ["ae", "h8", "rk_p1", "rk_sl", "t", "p1", "sl"]
NPZ = {(r, c): np.load(os.path.join(D, f"{r}_{c}.npz")) for r in ("select", "final") for c in (0, 1)}
dat = {k: {n: z[n] for n in KEEP} for k, z in NPZ.items()}
def Tsum(h, ae): return float(np.sum(np.exp(-2 * h / SIG) * (ae <= h)))
def Tall(hs, aes): return sum(Tsum(h, a) for h, a in zip(hs, aes)) / sum(len(a) for a in aes)
def k_for_T(role, target=0.4914):
    lo, hi = 1.0, 3.0
    for _ in range(40):
        k = (lo + hi) / 2
        t = Tall([dat[(role, c)]["h8"] for c in (0, 1)], [dat[(role, c)]["ae"] * k for c in (0, 1)])
        lo, hi = (k, hi) if t > target else (lo, k)
    return (lo + hi) / 2
def opt_h(ae):
    e = np.sort(ae); m = len(e)
    if m == 0: return 0.0
    return float(e[np.argmax(np.exp(-2 * e / SIG) * np.arange(1, m + 1) / m)])
def lookup(s, ae, nb):
    edges = np.unique(np.quantile(s, np.linspace(0, 1, nb + 1)[1:-1]))
    idx = np.searchsorted(edges, s); hs = np.array([opt_h(ae[idx == b]) for b in range(len(edges) + 1)])
    return edges, hs
def lookup2(s, g, ae, nb, ng):
    """per group g (int 0..ng-1), separate lookup on s."""
    return [lookup(s[g == j], ae[g == j], nb) for j in range(ng)]
def apply2(tab, s, g):
    h = np.zeros(len(s), np.float32)
    for j, (e, hs) in enumerate(tab):
        m = g == j; h[m] = hs[np.searchsorted(e, s[m])]
    return h
def _cols(z, idx=None):
    g = (lambda k: z[k] if idx is None else z[k][idx])
    for k in LOGF: yield lambda k=k: np.log(g(k) + EPS)
    for k in LINF: yield lambda k=k: g(k)
    yield lambda: g("t") ** 2 / 400
    yield lambda: g("yy") ** 2
    yield lambda: g("xx") ** 2
    yield lambda: np.ones_like(g("t"))
def design(z, idx):
    return np.stack([f() for f in _cols(z, idx)], 1).astype(np.float32)
def design_dot(z, w):
    acc = None
    for f, wi in zip(_cols(z), w):
        v = f() * np.float32(wi); acc = v if acc is None else acc + v
    return acc.astype(np.float32)
KR = {"select": 1.534306585789409, "final": 1.6645845770829055}
KT = {r: k_for_T(r) for r in ("select", "final")}
print("k_rel", KR, "k_T(v8 live T=0.4914)", KT, flush=True)
rng = np.random.default_rng(0)
# k-invariant pieces fitted once on SELECT raw residuals: ridge coefs
W = {}
for c in (0, 1):
    d = dat[("select", c)]; idx = np.sort(rng.choice(len(d["ae"]), 1_500_000, replace=False))
    A = design(NPZ[("select", c)], idx); b = np.log(d["ae"][idx] + EPS)
    W[c] = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ b)
    pred = design_dot(NPZ[("final", c)], W[c])
    from scipy.stats import spearmanr
    j = rng.choice(len(pred), 300000, replace=False)
    print(f"ch {c} ridge FINAL spearman {spearmanr(pred[j], dat[('final', c)]['ae'][j]).correlation:.3f}"
          f"  p1 {spearmanr(dat[('final', c)]['p1'][j], dat[('final', c)]['ae'][j]).correlation:.3f}"
          f"  sl {spearmanr(dat[('final', c)]['sl'][j], dat[('final', c)]['ae'][j]).correlation:.3f}", flush=True)
    dat[("select", c)]["ridge"] = design_dot(NPZ[("select", c)], W[c])
    dat[("final", c)]["ridge"] = pred
def lg(d, n=4): return np.minimum(d["t"].astype(np.int32) * n // 20, n - 1)
def rbin(r, n): return np.minimum((r * n).astype(np.int32), n - 1)
def run(kf, ke):
    out = {}
    def fit_eval(name, fitfn):
        hs, aes = [], []
        for c in (0, 1):
            ds, df = dat[("select", c)], dat[("final", c)]
            h = fitfn(ds, ds["ae"] * kf, df, c)
            hs.append(h); aes.append(df["ae"] * ke)
        out[name] = (Tall(hs, aes), [Tsum(h, a) / len(a) for h, a in zip(hs, aes)])
    fit_eval("v8", lambda ds, a, df, c: df["h8"])
    def scale_v8(ds, a, df, c):
        best = max((Tsum(s * ds["h8"], a), s) for s in np.geomspace(0.3, 3, 60))[1]
        return best * df["h8"]
    fit_eval("v8 x s_c (global rescale/ch)", scale_v8)
    fit_eval("flat h_c", lambda ds, a, df, c: np.full(len(df["ae"]), opt_h(a), np.float32))
    def lk(key, nb):
        def f(ds, a, df, c):
            e, hs = lookup(ds[key], a, nb); return hs[np.searchsorted(e, df[key])]
        return f
    fit_eval("1D lookup sl (abs,40)", lk("sl", 40))
    fit_eval("1D lookup p1 (abs,40)", lk("p1", 40))
    fit_eval("ridge-logfeat lookup (abs,40)", lk("ridge", 40))
    def lk2(key, nb, ng):
        def f(ds, a, df, c):
            tab = lookup2(ds[key], lg(ds, ng), a, nb, ng); return apply2(tab, df[key], lg(df, ng))
        return f
    fit_eval("p1 lookup x lead4", lk2("p1", 20, 4))
    fit_eval("ridge lookup x lead4", lk2("ridge", 20, 4))
    def rk(key, nb, ng):
        def f(ds, a, df, c):
            gs = rbin(ds[key], nb) * ng + lg(ds, ng); gf = rbin(df[key], nb) * ng + lg(df, ng)
            hs = np.array([opt_h(a[gs == j]) for j in range(nb * ng)]); return hs[gf]
        return f
    fit_eval("rank-in-call p1 10bin x lead4 (abs h)", rk("rk_p1", 10, 4))
    fit_eval("rank-in-call sl 6bin x lead4 (abs h)", rk("rk_sl", 6, 4))
    def mult_tab(ds, a, df, c, nb=6, ng=20):   # lit #2: multiplier on v8 per (lead x sl-rank) cell
        gs = rbin(ds["rk_sl"], nb) * ng + ds["t"].astype(int); gf = rbin(df["rk_sl"], nb) * ng + df["t"].astype(int)
        grid = np.linspace(0.2, 2.0, 37); ms = np.ones(nb * ng)
        for j in range(nb * ng):
            m = gs == j
            if m.any(): ms[j] = grid[np.argmax([Tsum(g * ds["h8"][m], a[m]) for g in grid])]
        return ms[gf] * df["h8"]
    fit_eval("v8 x mult[lead20 x slrank6] (lit#2)", mult_tab)
    def laplace(ds, a, df, c):
        bs = np.exp(ds["ridge"]); best = max((Tsum(cc * bs * np.log(1 + SIG / (2 * cc * bs)), a), cc) for cc in np.geomspace(0.3, 8, 50))[1]
        bf = best * np.exp(df["ridge"]); return bf * np.log(1 + SIG / (2 * bf))
    fit_eval("ridge b + Laplace h=b ln(1+s/2b)", laplace)
    return out
allres = {}
for tag, kf, ke in (("rel/rel", KR["select"], KR["final"]), ("T/T", KT["select"], KT["final"]),
                    ("fit rel, eval T", KR["select"], KT["final"]), ("fit T, eval rel", KT["select"], KR["final"]),
                    ("fit T, eval 2.0", KT["select"], 2.0)):
    r = run(kf, ke); allres[tag] = r
    print(f"\n### proxy {tag}: k_fit {kf:.3f} k_eval {ke:.3f}")
    base = r["v8"][0]
    for n, (T, pc) in r.items():
        print(f"  {n:40} T {T:.4f} (u {pc[0]:.4f} v {pc[1]:.4f})  dT {T-base:+.4f}  ~dSPS@W0.678 {100*0.678*(T-base):+.2f}", flush=True)
json.dump({t: {n: v[0] for n, v in r.items()} for t, r in allres.items()}, open(os.path.join(HERE, "recipes_out.json"), "w"), indent=1)
