#!/usr/bin/env python3
"""FULL search: proportions + DMD variables + PERMUTATIONS (compositions).

Closes the three gaps I listed and then failed to close:

  1. SYMBOLIC / FUNCTIONAL FORMS -- not just weighted sums. A linear blend
     a*FNO + b*DMD applies ONE weight to the mean field and the fluctuation
     alike. That is the wrong instrument for this problem: the FNO is
     rel-optimal precisely because it is the conditional mean (it carries 56%
     of the target variance), while the DMD exists to supply fluctuation for
     tke. A COMPOSITION can take the mean from one model and the fluctuation
     from another at an independent gain, which no convex mix can express.

  2. PERMUTATIONS / ORDERINGS -- which model supplies which PART:
        MEANFLUC   mean_t(A) + g * (B - mean_t(B))
        SPECSPLICE low spatial modes from A, high modes from B
        LEADSPLICE lead times 0..t from A, t..20 from B
        CHANSPLICE u from A, v from B
     Motivated by measurement, not decoration: 86.9% of the FNO's error lives
     in modes it CAN represent and 52-60% of it in the lowest four, so a
     low/high mode split is exactly where the models differ most.

  3. THE DMD'S OWN VARIABLES -- RANK / RHO / TKE_GAIN swept, then fed into
     every form above. Those constants were tuned when tke was implicitly
     priced near 0.2; it is actually 0.086, so TKE_GAIN (whose only job is to
     buy tke) was optimised at ~2.3x the true price.

Objective is the full composite INCLUDING sps (weight 0.243, omitted from the
15 Aug search, which is why that search returned a false PASS on the corrector).
w_time = 0.0952, measured directly from v8 vs v9. Any form that runs the DMD
pays -0.219 composite before scoring.

FIT chooses. HOLDOUT is touched once, at the very end, for the single winner.
"""
import sys, os, io, contextlib
import numpy as np, h5py, torch, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'src', d) for d in ('training', 'evaluation', 'splits')]
KIT = os.path.join(HERE, ".dist_eta0")
V7 = os.path.join(HERE, ".dist_v7")
T2 = os.environ.get("RPDE_T2_KIT_DIR", "")
SC = os.path.join(REPO, "realpde_t1_starting_kit_v9", "realpde_t1_starting_kit_v9", "scoring.py")
ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
CACHE = os.path.join(HERE, "combo_cache.npz")

sp = importlib.util.spec_from_file_location("sc9", SC)
sc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(sc)

MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32)
SI = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32)
ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)

W = dict(rel=0.476, tke=0.086, mvpe=0.095, sps=0.243)
DMD_TIME = -0.219
A_U, A_V, K = 0.004, 0.003, 1.5
NB = 6
BM = np.array([[1.50, 1.00, 0.80, 0.60, 0.40, 0.25],
               [0.90, 0.90, 0.80, 0.60, 0.40, 0.25]], np.float32)

FIT = ['24150_10', '20325_20', '22875_10', '19050_20', '25425_15', '25425_10',
       '21600_10', '17775_20', '26700_10', '21600_20', '16500_20']
HOLD = ['21600_15', '22875_15', '20325_15', '24150_15', '19050_15']


def windows(names, stride=60):
    X, Y = [], []
    for t in names:
        p = os.path.join(ROOT, t + ".h5")
        if not os.path.exists(p):
            continue
        with h5py.File(p) as f:
            u = np.asarray(f["u"][:, ::2, ::2], np.float32)
            v = np.asarray(f["v"][:, ::2, ::2], np.float32)
        fr = np.stack([u, v, np.zeros_like(u)], -1)
        for i in range(0, len(fr) - 40 + 1, stride):
            X.append(fr[i:i + 20]); Y.append(fr[i + 20:i + 40])
    return np.stack(X), np.stack(Y)


def band(X, P):
    xc = X[..., :2]
    fl = xc - xc.mean(axis=1, keepdims=True)
    s = np.sqrt((fl * fl).mean(axis=1))
    valid = (xc != 0.0).any(axis=1)
    mult = np.ones_like(s)
    for c in (0, 1):
        col, v = s[..., c], valid[..., c]
        for i in range(0, len(col), 5):
            sl = slice(i, min(i + 5, len(col)))
            pool = col[sl][v[sl]]
            if pool.size < NB:
                continue
            e = np.quantile(pool, np.linspace(0, 1, NB + 1)[1:-1])
            mult[sl, ..., c] = BM[c][np.clip(np.searchsorted(e, col[sl]), 0, NB - 1)]
    h = np.zeros((len(X), 1) + P.shape[2:5], np.float32)
    h[:, 0, :, :, 0] = (A_U + K * s[..., 0]) * mult[..., 0]
    h[:, 0, :, :, 1] = (A_V + K * s[..., 1]) * mult[..., 1]
    return h


def score(P, X, Y):
    s = dict(rel=sc.score_error(float(sc.rel_l2_per_sample(P, Y, 2).mean())),
             tke=sc.score_error(float(sc.tke_rel_l2_per_sample(P, Y, 2).mean())),
             mvpe=sc.score_error(float(sc.mvpe_rel_l2_per_sample(P, Y).mean())))
    hb = band(X, P)
    wv, _ = sc.aggregate_sps(P, Y, 2, P - hb, P + hb)
    s["sps"] = sc.score_sps(wv)
    s["w"] = sum(W[k] * s[k] for k in W)
    return s


def build():
    sys.path.insert(0, KIT)
    from load_baseline import load_baseline
    fno, _ = load_baseline(os.path.join(KIT, "sim_real_fno_fp16.pth"), device="cpu")
    fno = fno.float().eval()
    sys.path.insert(0, T2)
    s2 = importlib.util.spec_from_file_location("lb2", os.path.join(T2, "load_baseline.py"))
    lb2 = importlib.util.module_from_spec(s2); s2.loader.exec_module(lb2)
    CK = os.path.join(T2, "NeurIPS 2026 RealPDECompetition", "baseline_checkpoints", "sim_real_ft")
    cno, _ = lb2.load_baseline(os.path.join(CK, "sim_real_cno.pth"), device="cpu")
    cno = cno.float().eval()
    sys.path.insert(0, V7)
    import submission as S
    corr = S._corr_stages[0][0] if S._corr_stages else None

    def fwd(m, Xa, b=4):
        o = []
        with torch.no_grad():
            for i in range(0, len(Xa), b):
                q = (m((torch.from_numpy(Xa[i:i + b]) - MI) / SI) * ST + MT).numpy().copy()
                q[..., 2] = 0.0; o.append(q)
        return np.concatenate(o)

    out = {}
    for tag, names in (("f", FIT), ("h", HOLD)):
        X, Y = windows(names)
        print(f"  {tag}: {len(X)} windows", flush=True)
        out[f"{tag}_X"] = X; out[f"{tag}_Y"] = Y
        out[f"{tag}_fno"] = fwd(fno, X); print(f"  {tag}: fno", flush=True)
        out[f"{tag}_cno"] = fwd(cno, X); print(f"  {tag}: cno", flush=True)
        if corr is not None:
            c = []
            with torch.no_grad():
                for i in range(0, len(X), 4):
                    c.append(corr(torch.from_numpy(X[i:i + 4])).numpy())
            out[f"{tag}_corr"] = np.concatenate(c)
        np.savez_compressed(CACHE, **out); print(f"  {tag}: cached", flush=True)
    return out


if os.path.exists(CACHE):
    print("loading cache ...", flush=True)
    z = np.load(CACHE); C = {k: z[k] for k in z.files}
    need = ["f_X", "f_Y", "f_fno", "f_cno", "h_X", "h_Y", "h_fno", "h_cno"]
    if any(k not in C for k in need):
        print("  cache incomplete -> rebuilding", flush=True); C = build()
else:
    print("building cache ...", flush=True); C = build()

sys.path.insert(0, os.path.join(HERE, "dmd_src"))
with contextlib.redirect_stderr(io.StringIO()):
    ds = importlib.util.spec_from_file_location("dmdsub", os.path.join(HERE, "dmd_src", "submission.py"))
    dmd = importlib.util.module_from_spec(ds); ds.loader.exec_module(dmd)


def run_dmd(X, rank, rho, gain):
    dmd.RANK, dmd.RHO, dmd.TKE_GAIN = rank, rho, gain
    o = []
    with contextlib.redirect_stderr(io.StringIO()):
        for i in range(0, len(X), 5):
            o.append(np.asarray(dmd.predict(X[i:i + 5])["prediction"], np.float32))
    return np.concatenate(o)


# ------------------------------------------------------------------ FORMS
def f_linear(A, B, D, corr, p):
    P = p["wa"] * A + p["wb"] * B + p["wd"] * D
    P = P.copy()
    if p["eta"] and corr is not None:
        P[..., :2] += p["eta"] * corr[..., :2]
    P[..., 2] = 0.0
    return P


def f_meanfluc(A, B, D, corr, p):
    """mean field from one model, fluctuation from another at its own gain."""
    src = {"fno": A, "cno": B, "dmd": D}
    M, F = src[p["m"]], src[p["f"]]
    P = M.mean(axis=1, keepdims=True) + p["g"] * (F - F.mean(axis=1, keepdims=True))
    P = P.copy()
    if p["eta"] and corr is not None:
        P[..., :2] += p["eta"] * corr[..., :2]
    P[..., 2] = 0.0
    return P


def f_specsplice(A, B, D, corr, p):
    """low spatial modes from one model, high modes from another."""
    src = {"fno": A, "cno": B, "dmd": D}
    L, Hh = src[p["lo"]], src[p["hi"]]
    FL = np.fft.rfft2(L[..., :2], axes=(2, 3))
    FH = np.fft.rfft2(Hh[..., :2], axes=(2, 3))
    nh, nw = FL.shape[2], FL.shape[3]
    kh = np.minimum(np.arange(nh), nh - np.arange(nh))[:, None]
    kw = np.arange(nw)[None, :]
    low = ((kh < p["k"]) & (kw < p["k"]))[None, None, :, :, None]
    P = L.copy()
    P[..., :2] = np.fft.irfft2(np.where(low, FL, FH), s=L.shape[2:4], axes=(2, 3))
    if p["eta"] and corr is not None:
        P[..., :2] += p["eta"] * corr[..., :2]
    P[..., 2] = 0.0
    return P


def f_leadsplice(A, B, D, corr, p):
    """early lead times from one model, later ones from another."""
    src = {"fno": A, "cno": B, "dmd": D}
    P = src[p["e"]].copy()
    P[:, p["t"]:] = src[p["l"]][:, p["t"]:]
    if p["eta"] and corr is not None:
        P[..., :2] += p["eta"] * corr[..., :2]
    P[..., 2] = 0.0
    return P


def f_chansplice(A, B, D, corr, p):
    """u from one model, v from another."""
    src = {"fno": A, "cno": B, "dmd": D}
    P = src[p["u"]].copy()
    P[..., 1] = src[p["v"]][..., 1]
    if p["eta"] and corr is not None:
        P[..., :2] += p["eta"] * corr[..., :2]
    P[..., 2] = 0.0
    return P


Xf, Yf, Xh, Yh = C["f_X"], C["f_Y"], C["h_X"], C["h_Y"]
base_f = score(C["f_fno"], Xf, Yf)
base_h = score(C["h_fno"], Xh, Yh)
print(f"\nFROZEN FNO   FIT w {base_f['w']:.4f}   HOLDOUT w {base_h['w']:.4f}")
print(f"  HOLDOUT rel {base_h['rel']:.3f} tke {base_h['tke']:.3f} "
      f"mvpe {base_h['mvpe']:.3f} sps {base_h['sps']:.3f}")

# ---- STAGE 1: DMD variables ------------------------------------------------
print("\n=== STAGE 1: DMD VARIABLE SEARCH (standalone, FIT) ===", flush=True)
print(f"{'rank':>5}{'rho':>7}{'gain':>6}{'rel':>9}{'tke':>9}{'mvpe':>9}{'sps':>9}{'w':>10}")
res = []
for rank in (2, 4, 6, 8):
    for rho in (0.90, 0.95, 1.00):
        for gain in (0.0, 0.4, 0.8, 1.2):
            s = score(run_dmd(Xf, rank, rho, gain), Xf, Yf)
            res.append((s["w"], rank, rho, gain))
            print(f"{rank:5d}{rho:7.2f}{gain:6.1f}{s['rel']:9.3f}{s['tke']:9.3f}"
                  f"{s['mvpe']:9.3f}{s['sps']:9.3f}{s['w']:10.4f}", flush=True)
res.sort(reverse=True, key=lambda r: r[0])
ship = [r[0] for r in res if r[1] == 6 and r[2] == 0.95 and r[3] == 0.8][0]
print(f"\nshipped (6, 0.95, 0.8) w = {ship:.4f}")
print(f"best    ({res[0][1]}, {res[0][2]}, {res[0][3]}) w = {res[0][0]:.4f}"
      f"   -> DMD variables worth {res[0][0]-ship:+.4f} standalone")

BEST = res[0][1:4]
Df = run_dmd(Xf, *BEST)
Dh = run_dmd(Xh, *BEST)
Af, Bf, cf = C["f_fno"], C["f_cno"], C.get("f_corr")
Ah, Bh, ch = C["h_fno"], C["h_cno"], C.get("h_corr")

# ---- STAGE 2+3: every form ------------------------------------------------
print("\n=== STAGE 2: ALL FORMS (proportions + permutations), FIT ===", flush=True)
ETAS = (0.0, 0.2, 0.4, 0.8, 1.2)
MODELS = ("fno", "cno", "dmd")
cands = []
gg = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
for wb in gg:
    for wd in gg:
        wa = 1.0 - wb - wd
        if wa < 0.5:
            continue
        for eta in ETAS:
            cands.append(("linear", f_linear, dict(wa=wa, wb=wb, wd=wd, eta=eta)))
for m in MODELS:
    for f in MODELS:
        for g in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5):
            for eta in (0.0, 0.4):
                cands.append(("meanfluc", f_meanfluc, dict(m=m, f=f, g=g, eta=eta)))
for lo in MODELS:
    for hi in MODELS:
        if lo == hi:
            continue
        for k in (2, 4, 6, 8, 12):
            for eta in (0.0, 0.4):
                cands.append(("specsplice", f_specsplice, dict(lo=lo, hi=hi, k=k, eta=eta)))
for e in MODELS:
    for l in MODELS:
        if e == l:
            continue
        for t in (2, 5, 8, 12, 16):
            for eta in (0.0, 0.4):
                cands.append(("leadsplice", f_leadsplice, dict(e=e, l=l, t=t, eta=eta)))
for u in MODELS:
    for v in MODELS:
        if u == v:
            continue
        for eta in (0.0, 0.4):
            cands.append(("chansplice", f_chansplice, dict(u=u, v=v, eta=eta)))
print(f"{len(cands):,} candidates across 5 functional forms", flush=True)


def uses_dmd(name, p):
    if name == "linear":
        return p["wd"] > 1e-9
    return "dmd" in [str(x) for x in p.values()]


bestof = {}
for name, fn, p in cands:
    s = score(fn(Af, Bf, Df, cf, p), Xf, Yf)
    net = s["w"] + (DMD_TIME if uses_dmd(name, p) else 0.0)
    if name not in bestof or net > bestof[name][0]:
        bestof[name] = (net, p, s)

print(f"\n{'form':<12}{'net(FIT)':>10}{'rel':>9}{'tke':>9}{'mvpe':>9}{'sps':>9}   params")
print(f"{'frozen FNO':<12}{base_f['w']:10.4f}{base_f['rel']:9.3f}{base_f['tke']:9.3f}"
      f"{base_f['mvpe']:9.3f}{base_f['sps']:9.3f}")
for name in ("linear", "meanfluc", "specsplice", "leadsplice", "chansplice"):
    if name not in bestof:
        continue
    net, p, s = bestof[name]
    print(f"{name:<12}{net:10.4f}{s['rel']:9.3f}{s['tke']:9.3f}{s['mvpe']:9.3f}"
          f"{s['sps']:9.3f}   {p}")

wname = max(bestof, key=lambda k: bestof[k][0])
wnet, wp, wsf = bestof[wname]
fnmap = dict(linear=f_linear, meanfluc=f_meanfluc, specsplice=f_specsplice,
             leadsplice=f_leadsplice, chansplice=f_chansplice)
h = score(fnmap[wname](Ah, Bh, Dh, ch, wp), Xh, Yh)
net_h = h["w"] - base_h["w"] + (DMD_TIME if uses_dmd(wname, wp) else 0.0)
print("\n" + "=" * 68)
print(f"HOLDOUT -- touched once, winner only: form={wname} params={wp}")
print(f"  {'':9}{'rel':>9}{'tke':>9}{'mvpe':>9}{'sps':>9}{'w':>10}")
print(f"  {'frozen':9}{base_h['rel']:9.3f}{base_h['tke']:9.3f}{base_h['mvpe']:9.3f}"
      f"{base_h['sps']:9.3f}{base_h['w']:10.4f}")
print(f"  {'winner':9}{h['rel']:9.3f}{h['tke']:9.3f}{h['mvpe']:9.3f}"
      f"{h['sps']:9.3f}{h['w']:10.4f}")
print(f"  delta {h['w']-base_h['w']:+.4f}"
      + (f"   time {DMD_TIME:+.3f}" if uses_dmd(wname, wp) else "   (no DMD)")
      + f"   NET {net_h:+.4f}")
print(f"  FLOOR +0.10 -> {'PASS' if net_h >= 0.10 else 'FAIL'}")
