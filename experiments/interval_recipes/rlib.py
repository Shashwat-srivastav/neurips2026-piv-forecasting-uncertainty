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
#print("k_rel", KR, "k_T(v8 live T=0.4914)", KT, flush=True)
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
    if 0: print(f"ch {c} ridge FINAL spearman {spearmanr(pred[j], dat[('final', c)]['ae'][j]).correlation:.3f}"
          f"  p1 {spearmanr(dat[('final', c)]['p1'][j], dat[('final', c)]['ae'][j]).correlation:.3f}"
          f"  sl {spearmanr(dat[('final', c)]['sl'][j], dat[('final', c)]['ae'][j]).correlation:.3f}", flush=True)
    dat[("select", c)]["ridge"] = design_dot(NPZ[("select", c)], W[c])
    dat[("final", c)]["ridge"] = pred
