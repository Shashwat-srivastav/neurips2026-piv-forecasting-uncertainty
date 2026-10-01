"""Which v17 component carries the fingerprint dependence? Start from SELECT-fitted per-window tables and
swap ONE component (pred / band centre / width) for its random-split counterpart. FINAL, proxy AT (k=1.447)."""
from lib2 import *; import pickle, gc
X, Y, P, traj = load("final")
F = prep(X, Y, P, KA["final"], KT["final"], only=("AT",)); Yt, Pt = F["pr"]["AT"]
v8 = score(Pt, Yt, Pt, F["h8"])
S = pickle.load(open("tabs_select_pw.pkl", "rb"))
for rt in ("rand_pw", "randk_pw"):
    R = pickle.load(open(f"tabs_{rt}.pkl", "rb"))
    for name, parts in (("SELECT (all)", ()), ("swap pred", ("pred",)), ("swap centre", ("band",)),
                        ("swap width", ("width",)), ("swap centre+width", ("band", "width")),
                        ("random (all)", ("pred", "band", "width"))):
        tabs = {k: (R[k] if k in parts else S[k]) for k in S}
        _, out, c, h = outputs(F, tabs)
        d = delta(score(out, Yt, c, h), v8)
        print(f"{rt:9} {name:18} " + " ".join(f"{k} {d[k]:+.3f}" for k in ("rel", "tke", "mvpe", "sps", "comp")), flush=True)
        del out, c, h; gc.collect()
