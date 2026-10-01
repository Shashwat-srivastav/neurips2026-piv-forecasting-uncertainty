"""A1 + A2 evaluation on FINAL under AT (k=1.447), per-window bins. Per-trajectory deltas vs v8."""
from lib2 import *; import pickle, sys
X, Y, P, traj = load("final")
F = prep(X, Y, P, KA["final"], KT["final"], only=("AT",)); Yt, Pt = F["pr"]["AT"]
names = list(dict.fromkeys(traj.tolist())); idx = {n: np.where(traj == n)[0] for n in names}
v8 = dict(all=score(Pt, Yt, Pt, F["h8"]), **{n: score_sub(Pt, Yt, Pt, F["h8"], i) for n, i in idx.items()})
res = {}
for tag in sys.argv[1:]:
    tabs = pickle.load(open(f"tabs_{tag}.pkl", "rb"))
    Yt_, out, c, h = outputs(F, tabs)
    r = dict(all=score(out, Yt, c, h), **{n: score_sub(out, Yt, c, h, i) for n, i in idx.items()})
    res[tag] = r; d = delta(r["all"], v8["all"])
    print(f"== {tag}: pooled FINAL AT vs v8: " + " ".join(f"{k} {d[k]:+.3f}" for k in ("rel","tke","mvpe","sps","T","comp")), flush=True)
    per = [(n, delta(r[n], v8[n])) for n in names]
    for n, dd in per: print(f"   {n:9} n={len(idx[n]):3}  comp {dd['comp']:+.3f}  sps {dd['sps']:+.3f}  rel {dd['rel']:+.3f} tke {dd['tke']:+.3f} mvpe {dd['mvpe']:+.3f}")
    cs = np.array([dd["comp"] for _, dd in per]); ss = np.array([dd["sps"] for _, dd in per])
    print(f"   per-traj comp: mean {cs.mean():+.3f} SD {cs.std(ddof=1):.3f} min {cs.min():+.3f} max {cs.max():+.3f} positive {int((cs>0).sum())}/10")
    print(f"   per-traj sps : mean {ss.mean():+.3f} SD {ss.std(ddof=1):.3f} min {ss.min():+.3f} max {ss.max():+.3f} positive {int((ss>0).sum())}/10", flush=True)
if len(res) == 2:
    a, b = sys.argv[1:3]; d = delta(res[b]["all"], res[a]["all"])
    print(f"== {b} minus {a} (pooled FINAL AT): " + " ".join(f"{k} {d[k]:+.3f}" for k in ("rel","tke","mvpe","sps","T","comp")))
    cs = np.array([delta(res[b][n], res[a][n])["comp"] for n in names])
    print(f"   per-traj: mean {cs.mean():+.3f} SD {cs.std(ddof=1):.3f} min {cs.min():+.3f} max {cs.max():+.3f} positive {int((cs>0).sum())}/10")
pickle.dump(dict(v8=v8, res=res), open("eval_final_rows.pkl", "wb"))
