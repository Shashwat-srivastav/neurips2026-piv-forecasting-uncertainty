from common import *
from feats import pixel_feats, elem_feats
from scipy.stats import spearmanr
rng = np.random.default_rng(0)
def binned_lookup_fit(score, ae, nb=20):
    edges = np.unique(np.quantile(score, np.linspace(0, 1, nb + 1)[1:-1]))
    idx = np.searchsorted(edges, score)
    hs = np.zeros(len(edges) + 1)
    for b in range(len(hs)):
        e = np.sort(ae[idx == b]); m = len(e)
        if m == 0: continue
        obj = np.exp(-2 * e / SIG) * np.arange(1, m + 1) / m
        hs[b] = e[np.argmax(obj)]
    return edges, hs
def best_T_pool(ae):
    e = np.sort(ae); m = len(e); obj = np.exp(-2*e/SIG)*np.arange(1,m+1)/m; return obj.max(), e[np.argmax(obj)]
S = {}
for role in ("select", "final"):
    X, Y, P, _ = load(role)
    k = fit_k(Y, P, "rel"); Yp = proxy_target(Y, P, k)
    names, cols = elem_feats(X, P)
    sc_ = (Yp != 0)
    if role == "final": fin_X = X
    S[role] = dict(names=names, cols=cols, ae=np.abs(Yp - P), m=sc_, k=k)
for c in (0, 1):
    print(f"=== channel {'uv'[c]}")
    sel, fin = S["select"], S["final"]
    ms = sel["m"][..., c]; mf = fin["m"][..., c]
    aes = sel["ae"][..., c][ms]; aef = fin["ae"][..., c][mf]
    idx = rng.choice(len(aes), 400000, replace=False)
    from common import v8_half
    h8 = np.broadcast_to(v8_half(fin_X), fin["ae"].shape)[..., c][mf]
    print(f"v8 FINAL T_c {np.mean(np.exp(-2*h8/SIG)*(aef<=h8)):.4f}")
    print(f"flat best FINAL-pool T (in-sample) {best_T_pool(aef)}  oracle h=|e|: {np.exp(-2*aef/SIG).mean():.4f}")
    for j, nm in enumerate(sel["names"]):
        fs = sel["cols"][j][..., c][ms]; ff = fin["cols"][j][..., c][mf]
        rho = spearmanr(fs[idx], aes[idx]).correlation
        edges, hs = binned_lookup_fit(fs, aes)
        hf = hs[np.searchsorted(edges, ff)]
        T = np.mean(np.exp(-2 * hf / SIG) * (aef <= hf))
        print(f"  {nm:6} spearman {rho:+.3f}   1-D lookup FINAL T_c {T:.4f}")
