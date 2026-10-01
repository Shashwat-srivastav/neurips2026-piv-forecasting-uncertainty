"""Exp.1 post-processing variants on FINAL (honest): which parts help this model? Compared with v17 LIVE."""
import os, sys, numpy as np, torch
sys.path.insert(0, "submissions/build_exp1"); sys.path.insert(0, "experiments/per_window_rebuild")
from load_baseline import load_baseline
import blend19 as B, importlib.util as iu
m, _ = load_baseline("submissions/build_exp1/sim_real_fno_fp16.pth", device="cpu"); m = m.float().eval(); torch.set_num_threads(os.cpu_count())
MI = torch.tensor([0.154960856, -0.000513992854, 0.0]); SI = torch.tensor([0.0968056545, 0.015960684, 1.0])
MT = torch.tensor([0.154962569, -0.000517793698, 0.0]); ST = torch.tensor([0.0968104079, 0.0159636438, 1.0])
d = np.load("experiments/interval_recipes/cache_final_s20.npz"); X, Y = d["X"], d["Y"]
x3 = np.concatenate([X, np.zeros(X.shape[:-1] + (1,), np.float32)], -1)
with torch.no_grad():
    P = np.concatenate([(m((torch.from_numpy(x3[i:i+5]) - MI) / SI) * ST + MT)[..., :2].numpy() for i in range(0, len(x3), 5)])
E, valid = B.estimators(X)
D = dict(X=X, Y=Y, P=P, Em=E["mean"], El=E["last"], valid=valid, L=None, role="final")
D["bins"] = B.rank_bins(B.sigma_local(X).astype(np.float32), valid, 6, 1)
D["p1b"] = B.rank_bins(np.sqrt(((X[:, 1:] - X[:, :-1]) ** 2).mean(1)).astype(np.float32), valid, 10, 1)
s = iu.spec_from_file_location("t", "submissions/build_exp1/_blend_tables.py"); t = iu.module_from_spec(s); s.loader.exec_module(t)
pt, bt, H = np.array(t.PRED_TAB), np.array(t.BAND_TAB), np.array(t.WIDTH_TAB)
sub = lambda e: 100 / (1 + 0.5 * e)
W = dict(rel=.453, tke=.087, mvpe=.081, sps=.284)
live = dict(rel=94.104, tke=75.368, mvpe=93.085, sps=37.287)
h = B.width_of(D, H); c = B.apply(D, P, bt); pb = B.apply(D, P, pt)
for name, pred, cen in (("raw FNO, tables' centre+width", P, c), ("blend pred, no boost", pb, c),
                        ("blend + boost (built zip)", B.boost(pb, X), c), ("raw FNO, centre=pred", P, P)):
    z = B.score(pred, Y, cen, h); r = dict(rel=sub(z["rel"]), tke=sub(z["tke"]), mvpe=sub(z["mvpe"]), sps=z["sps"])
    dl = sum(W[k] * (r[k] - live[k]) for k in W)
    print(f"{name:32s} rel {r['rel']:.3f} tke {r['tke']:.3f} mvpe {r['mvpe']:.3f} sps {r['sps']:.2f} | vs v17 LIVE {dl:+.3f}", flush=True)
