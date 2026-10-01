"""One-command build of the Exp.1 candidate (clean-start sim->real fine-tuned FNO + v17pw post-processing).

    python experiments/clean_fno_build/build_exp1.py <checkpoint.pth> [tag] [tables_from.py]
    (tag default "exp1" -> submissions/build_<tag>, submissions/submission_<tag>.zip; with tables_from, copy those tables
     instead of fitting -- used for the all-81 model, which has seen SELECT/FINAL)

<checkpoint.pth> is the fine-tune output of finetune_fno_kaggle.py started from sim_fno.pth on
train_pool_v12_3.json (fp32 {'model_state_dict': ...} or an already fp16-packed file).
Exp.1 never saw the 14 hold-out trajectories, so the blend/centre/width tables are fitted on SELECT
HONESTLY (k = 1, real targets, no residual-inflation proxy) and scored once on FINAL.
Steps: pack fp16 -> cache Exp.1 forecasts on SELECT/FINAL -> fit tables (per-window bins) -> report
FINAL scores -> build submissions/build_exp1 (copy of build_v17pw with the checkpoint swapped) -> zip.
Then run:  python submissions/build_exp1/test_independence.py ; python preflight.py submissions/submission_exp1.zip
"""
import os, shutil, subprocess, sys, zipfile
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
ROOT = os.path.dirname(LAB)
KIT = os.path.join(ROOT, "realpde_t1_starting_kit_v9", "realpde_t1_starting_kit_v9")
TAG = sys.argv[2] if len(sys.argv) > 2 else "exp1"
TABLES_FROM = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else None
SRC_BUILD, BUILD = os.path.join(LAB, "build_v17pw"), os.path.join(LAB, f"build_{TAG}")
ZIP = os.path.join(LAB, f"submission_{TAG}.zip")
ck = os.path.abspath(sys.argv[1])

# 1. build dir = v17pw with the checkpoint swapped (same file name, so submission.py is unchanged)
if os.path.exists(BUILD):
    shutil.rmtree(BUILD)
shutil.copytree(SRC_BUILD, BUILD, ignore=shutil.ignore_patterns("__pycache__"))
dst = os.path.join(BUILD, "sim_real_fno_fp16.pth")
obj = torch.load(ck, map_location="cpu", weights_only=False)
is_fp16 = isinstance(obj, dict) and obj.get("format") is not None
if is_fp16:
    shutil.copy(ck, dst)
else:
    subprocess.run([sys.executable, os.path.join(KIT, "pack_ckpt_fp16.py"), ck, dst], check=True)
print("checkpoint ->", dst, os.path.getsize(dst) / 1e6, "MB", flush=True)

# 2. Exp.1 forecasts on the cached SELECT/FINAL windows (same X/Y as every other study)
sys.path.insert(0, BUILD)
from load_baseline import load_baseline  # noqa: E402
model, _ = load_baseline(dst, device="cpu"); model = model.float().eval()
torch.set_num_threads(os.cpu_count())
MI = torch.tensor([0.154960856, -0.000513992854, 0.0]); SI = torch.tensor([0.0968056545, 0.015960684, 1.0])
MT = torch.tensor([0.154962569, -0.000517793698, 0.0]); ST = torch.tensor([0.0968104079, 0.0159636438, 1.0])
caches = {}
for role in ("select", "final"):
    d = np.load(os.path.join(LAB, "bands", f"cache_{role}_s20.npz"))
    X, Y = d["X"], d["Y"]
    x3 = np.concatenate([X, np.zeros(X.shape[:-1] + (1,), np.float32)], -1)
    P = []
    with torch.no_grad():
        for i in range(0, len(x3), 5):
            yb = model((torch.nan_to_num(torch.from_numpy(x3[i:i + 5])) - MI) / SI) * ST + MT
            P.append(yb[..., :2].numpy())
    caches[role] = dict(X=X, Y=Y, P=np.concatenate(P).astype(np.float32), Pofficial=d["P"], traj=d["traj"])
    print(role, "cached", X.shape, flush=True)

# 3. honest per-window fit (k = 1): reuse the per-window v17 pipeline with the proxies disabled
sys.path.insert(0, os.path.join(LAB, "pw"))
import blend19 as B  # noqa: E402
for dct in (B.KREL, B.KT):
    for r in dct:
        dct[r] = 1.0
B.tg = lambda D, pn: (D["Y"], D["P"])          # every "proxy" is the real target: honest data


def prep(role, which="P"):
    c = caches[role]
    D = dict(X=c["X"], Y=c["Y"], P=c[which], role=role, L=None)
    E, valid = B.estimators(c["X"]); D.update(Em=E["mean"], El=E["last"], valid=valid)
    D["bins"] = B.rank_bins(B.sigma_local(c["X"]).astype(np.float32), valid, 6, 1)
    p1 = np.sqrt(((c["X"][:, 1:] - c["X"][:, :-1]) ** 2).mean(1)).astype(np.float32)
    D["p1b"] = B.rank_bins(p1, valid, 10, 1)
    D["h8"] = B.v8_half(c["X"], True, 1)
    return D


if TABLES_FROM:
    import importlib.util as _iu
    _s = _iu.spec_from_file_location("tf", TABLES_FROM); _m = _iu.module_from_spec(_s); _s.loader.exec_module(_m)
    pt, bt, H = np.array(_m.PRED_TAB), np.array(_m.BAND_TAB), np.array(_m.WIDTH_TAB)
    print("tables copied from", TABLES_FROM, "(this model saw SELECT/FINAL: FINAL numbers below are NOT honest)")
else:
    S = prep("select")
    pt = B.fit_pred(S, ("A",), False, True)
    bt = B.fit_band(S, ("A", "B"), False)
    H = B.fit_width(S, bt, ("AT", "BT"))
with open(os.path.join(BUILD, "_blend_tables.py"), "w") as f:
    f.write('"""Exp.1 build: tables fitted HONESTLY on SELECT (k=1) with per-window bins (experiments/clean_fno_build/build_exp1.py)."""\n')
    f.write(f"PRED_TAB = {np.round(pt, 6).tolist()!r}\nBAND_TAB = {np.round(bt, 6).tolist()!r}\n")
    f.write(f"WIDTH_TAB = {np.round(H, 7).tolist()!r}\nBAND_G = 1.0\nBAND_MODE = 'tab'\n")

# 4. FINAL report (honest for Exp.1; the official model's row is on data it may have seen)
W = dict(rel=.454, tke=.087, mvpe=.080, sps=.284)
sub = lambda e: 100.0 / (1.0 + 0.5 * e)          # error -> leaderboard subscore
for label, which in (("Exp.1 raw + v8 bands", "P"), ("official raw + v8 bands", "Pofficial")):
    F = prep("final", which)
    s = B.score(F["P"], F["Y"], F["P"], F["h8"])
    print(f"{label:28s} rel {sub(s['rel']):.3f} tke {sub(s['tke']):.3f} mvpe {sub(s['mvpe']):.3f} sps {s['sps']:.2f}", flush=True)
F = prep("final")
pred = B.boost(B.apply(F, F["P"], pt), F["X"]); c = B.apply(F, F["P"], bt); h = B.width_of(F, H)
s = B.score(pred, F["Y"], c, h)
print(f"{'model + post-proc':28s} rel {sub(s['rel']):.3f} tke {sub(s['tke']):.3f} mvpe {sub(s['mvpe']):.3f} sps {s['sps']:.2f}  (FINAL; honest only for a model that never saw SELECT/FINAL)")

# 4b. GATE (Jev d017): compare against the recipe that scored live (official FNO + v17pw tables) under the
#     live-calibrated proxy AT (k = 1.447 on FINAL). Same windows, same weights; time is equal (same arch).
comp = lambda z: sum(W[m] * sub(z[m]) for m in ("rel", "tke", "mvpe")) + W["sps"] * z["sps"]
import importlib.util as _iu2
_s2 = _iu2.spec_from_file_location("t17", os.path.join(LAB, "build_v17pw", "_blend_tables.py"))
_m2 = _iu2.module_from_spec(_s2); _s2.loader.exec_module(_m2)
O = prep("final", "Pofficial")
Yat = np.where(O["Y"] == 0.0, 0.0, O["P"] + 1.447 * (O["Y"] - O["P"])).astype(np.float32)
po = B.boost(B.apply(O, O["P"], np.array(_m2.PRED_TAB)), O["X"]); co = B.apply(O, O["P"], np.array(_m2.BAND_TAB))
so = B.score(po, Yat, co, B.width_of(O, np.array(_m2.WIDTH_TAB)))
print(f"{'official + v17pw (proxy AT)':28s} rel {sub(so['rel']):.3f} tke {sub(so['tke']):.3f} mvpe {sub(so['mvpe']):.3f} sps {so['sps']:.2f}")
d = comp(s) - comp(so)
print(f"GATE: composite(new model + post-proc, honest) - composite(live recipe, proxy AT) = {d:+.3f}"
      f"  -> {'PASS (candidate for slot 1)' if d > 0 else 'FAIL (use v19pw)'}"
      + ("" if not TABLES_FROM else "   [not meaningful: this model saw FINAL]"), flush=True)

# 5. zip (test_independence.py is not shipped)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for r, ds, fs in os.walk(BUILD):
        ds[:] = [x for x in ds if x != "__pycache__"]
        for fn in fs:
            if fn != "test_independence.py":
                p = os.path.join(r, fn); z.write(p, os.path.relpath(p, BUILD))
print("wrote", ZIP, round(os.path.getsize(ZIP) / 1e6, 1), "MB")
