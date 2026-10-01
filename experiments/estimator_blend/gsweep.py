"""Proxy calibration: v8 band width sweep under proxies A/B on FINAL vs live board (noshape: g .74/1/1.21 -> 30.83/31.88/30.72)."""
from fit import *
X, Y, P, _ = load("final")
pr, ks = proxies("final", Y, P)
h8 = v8_half(X); h0 = v8_half(X, shape=False)
for pn, (Yt, Pt) in pr.items():
    for nm, h in (("noshape", h0), ("v8", h8)):
        out = []
        for g in (0.74, 0.9, 1.0, 1.1, 1.21, 1.35):
            s = score(Pt, Yt, Pt, g * h); out.append(f"g{g}:{s['sps']:.2f}/T{s['T']:.3f}")
        print(pn, nm, " ".join(out), flush=True)
