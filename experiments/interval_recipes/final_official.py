"""Official-scorer SPS on FINAL (proxy targets) for v8 and the top recipes. All fitting on SELECT only."""
import gc, numpy as np
from clib import *
from common import load, proxy_target, sc
def scored_to_full(vals, Y, c):
    full = np.zeros(Y.shape[:-1], np.float32); off = 0
    for i in range(0, len(Y), 10):
        m = Y[i:i+10, ..., c] != 0; n = int(m.sum())
        blk = full[i:i+10]; blk[m] = vals[off:off+n]; off += n
    assert off == len(vals); return full
import sys
PH = sys.argv[1]
if PH == "fit":
  for tag, kf, ke in (("rel/rel", KR["select"], KR["final"]), ("T/T", KT["select"], KT["final"])):
      recs = {}
      for c in (0, 1):
          ds, df = dat[("select", c)], dat[("final", c)]; a = ds["ae"] * kf
          recs.setdefault("v8", []).append((df["h8"], 0))
          e_, h_ = lookup(ds["p1"], a, 40); recs.setdefault("1D lookup p1 (frame-diff rms)", []).append((h_[np.searchsorted(e_, df["p1"])], 0))
          e_, h_ = lookup(ds["ridge"], a, 40); hr = h_[np.searchsorted(e_, df["ridge"])]
          recs.setdefault("ridge log-feature lookup", []).append((hr, 0))
          bs = np.exp(ds["ridge"]); cc = max((Tsum(q * bs * np.log(1 + SIG / (2 * q * bs)), a), q) for q in np.geomspace(0.3, 8, 50))[1]
          bf = cc * np.exp(df["ridge"]); recs.setdefault("ridge b + Laplace formula", []).append((bf * np.log(1 + SIG / (2 * bf)), 0))
          cs, cf = C[("select", c)], C[("final", c)]; gs, gf = grp(ds["t"]), grp(df["t"])
          ab = fit_ab(cs["e"], cs["cm"], cs["cl"], gs, "l1")
          rs = np.abs(cs["e"] * kf - (ab[gs, 0] * cs["cm"] + ab[gs, 1] * cs["cl"]))
          e_, h_ = lookup(ds["ridge"], rs, 40)
          recs.setdefault("ridge lookup + L1 centre shift", []).append((h_[np.searchsorted(e_, df["ridge"])], ab[gf, 0] * cf["cm"] + ab[gf, 1] * cf["cl"]))
      np.savez(os.path.join(D, f"rec_{tag.replace('/','')}.npz"), **{f"{n}|{c}|{w}": (v if w == 'h' else np.asarray(v, np.float32) if not np.isscalar(v) else np.zeros(0, np.float32))
             for n, per in recs.items() for c, (h, s) in enumerate(per) for w, v in (("h", h), ("s", s))})
else:
  X, Y, P, _ = load("final"); del X
  for tag, ke in (("rel/rel", KR["final"]), ("T/T", KT["final"])):
    z = np.load(os.path.join(D, f"rec_{tag.replace('/','')}.npz")); names = sorted({k.split("|")[0] for k in z.files}, key=lambda n: (n != "v8", n))
    print(f"### proxy {tag} (k_eval {ke:.3f})", flush=True)
    for name in names:
        H = np.stack([scored_to_full(z[f"{name}|{c}|h"], Y, c) for c in (0, 1)], -1)
        S = np.stack([scored_to_full(z[f"{name}|{c}|s"], Y, c) if len(z[f"{name}|{c}|s"]) else np.zeros(Y.shape[:-1], np.float32) for c in (0, 1)], -1)
        wsum = csum = tsum = nsum = 0.0
        for i in range(0, len(P), 60):
            p, yp = P[i:i+60], proxy_target(Y[i:i+60], P[i:i+60], ke)
            lo = p + S[i:i+60] - H[i:i+60]; up = p + S[i:i+60] + H[i:i+60]
            w, cov = sc.aggregate_sps(p, yp, 2, lower=lo, upper=up); n = int(np.count_nonzero(yp != 0))
            inside = (yp >= lo) & (yp <= up)
            tsum += float(np.sum(np.where(inside, np.exp(-(up - lo) / SIG), 0.0), where=yp != 0, dtype=np.float64))
            wsum += w * n; csum += cov * n; nsum += n
        print(f"  {name:34} sps {100*wsum/nsum:6.2f}  T {tsum/nsum:.4f}  cov {csum/nsum:.3f}", flush=True)
        del H, S; gc.collect()
