"""Post-hoc, deterministic, mean-preserving fluctuation re-injection for the stock FNO.

P_new[t] = Pbar + a * (P[t] - Pbar),  Pbar = prediction's own 20-frame mean (per pixel/channel)
=> mvpe (probe TEMPORAL means) is invariant by construction, tke map scales by a^2,
   rel pays (a-1)^2 ||P'||^2 - 2(a-1)<P', Y-P>.

Gain families (all computed per call from X and P only -- nothing from targets at test time):
  G  global constant a
  L  lead-dependent a(t) (fitted on SELECT energy-ratio curve, exponent gamma)
  I  input-anchored per window: a_w = (sum KE(X)/sum KE(P))^(gamma/2), clipped
  IR input-anchored per coarse region (blocks), smoothed

Scored with the official v9 scorer + the shipped v8 band (experiments/interval_recipes/common.py), with
bands centred on the NEW prediction (honest) and on the OLD one.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(BASE, "lab", "bands"))
import common as C  # noqa: E402
sc = C.sc

W_PART = dict(rel=0.476, tke=0.086, mvpe=0.095, sps=0.243)   # sec 9 partials (sps explicit here)


def ke(a):
    f = a - a.mean(1, keepdims=True)
    return 0.5 * (f ** 2).mean(1).sum(-1)


def apply_gain(P, a):
    """a broadcastable to (N,T,H,W,1|2). Re-centres so the temporal mean is exactly Pbar."""
    Pb = P.mean(1, keepdims=True)
    Pn = Pb + a * (P - Pb)
    Pn = Pn - (Pn.mean(1, keepdims=True) - Pb)
    return Pn.astype(np.float32)


def evaluate(P, Y, h, Pband=None):
    rel = sc.score_error(float(sc.rel_l2_per_sample(P, Y, 2).mean()))
    tke = sc.score_error(float(sc.tke_rel_l2_per_sample(P, Y, 2).mean()))
    mvpe = sc.score_error(float(sc.mvpe_rel_l2_per_sample(P, Y).mean()))
    c = P if Pband is None else Pband
    hb = np.broadcast_to(h, P.shape).astype(np.float32)
    w, cov = sc.aggregate_sps(P, Y, 2, lower=c - hb, upper=c + hb)
    return dict(rel=rel, tke=tke, mvpe=mvpe, sps=100 * w, cov=cov)


def delta(s, b):
    d = {k: s[k] - b[k] for k in ("rel", "tke", "mvpe", "sps")}
    d["comp"] = sum(W_PART[k] * d[k] for k in W_PART)
    return d


def fmt(tag, s, b):
    d = delta(s, b)
    return (f"{tag:34} rel {s['rel']:7.3f} ({d['rel']:+.3f})  tke {s['tke']:7.3f} ({d['tke']:+.3f})  "
            f"mvpe {s['mvpe']:7.3f} ({d['mvpe']:+.3f})  sps {s['sps']:6.3f} ({d['sps']:+.3f})  comp {d['comp']:+.3f}")


# ---------- gain constructors (use only X, P) ----------
def gain_global(X, P, a):
    return a


def lead_curve_select():
    X, Y, P, _ = C.load("select")
    fp = P - P.mean(1, keepdims=True); fy = Y - Y.mean(1, keepdims=True)
    r = (fp ** 2).sum((0, 2, 3, 4)) / (fy ** 2).sum((0, 2, 3, 4))
    return r  # (20,)


def gain_lead(X, P, r, gamma, cap=2.0):
    a = np.clip(r ** (-gamma / 2), 1.0, cap)
    return a[None, :, None, None, None]


def gain_input(X, P, gamma, cap=1.6):
    kx = ke(X).reshape(len(X), -1).sum(1); kp = ke(P).reshape(len(P), -1).sum(1)
    a = np.clip((kx / np.maximum(kp, 1e-12)) ** (gamma / 2), 1.0, cap)
    return a[:, None, None, None, None]


def block_mean(K, by, bx):
    N, H, W = K.shape
    return K.reshape(N, H // by, by, W // bx, bx).mean((2, 4))


def gain_input_region(X, P, gamma, by=8, bx=16, cap=1.8):
    kx = block_mean(ke(X), by, bx); kp = block_mean(ke(P), by, bx)
    a = np.clip((kx / np.maximum(kp, 1e-12)) ** (gamma / 2), 1.0, cap)
    a = np.repeat(np.repeat(a, by, 1), bx, 2)
    return a[:, None, :, :, None]


def run(role, cfgs, r_sel):
    X, Y, P, _ = C.load(role)
    h = C.v8_half(X)
    base = evaluate(P, Y, h)
    out = {"base": base}
    print(f"\n==== {role} ({len(P)} windows)")
    print(fmt("stock", base, base))
    for tag, fn in cfgs:
        a = fn(X, P, r_sel)
        Pn = apply_gain(P, a)
        s = evaluate(Pn, Y, h)
        s_old = evaluate(Pn, Y, h, Pband=P)
        out[tag] = dict(new=s, oldband=s_old, d=delta(s, base), d_old=delta(s_old, base))
        print(fmt(tag, s, base))
        print(fmt("   (band centred on old P)", s_old, base))
    return out


if __name__ == "__main__":
    r_sel = lead_curve_select()
    print("SELECT lead energy ratio:", np.round(r_sel, 3))
    cfgs = []
    for a in (1.05, 1.1, 1.15, 1.2, 1.3):
        cfgs.append((f"G a={a}", lambda X, P, r, a=a: gain_global(X, P, a)))
    for g in (0.5, 1.0):
        cfgs.append((f"L gamma={g}", lambda X, P, r, g=g: gain_lead(X, P, r, g)))
    for g in (0.5, 0.75, 1.0):
        cfgs.append((f"I gamma={g}", lambda X, P, r, g=g: gain_input(X, P, g)))
    for g in (0.5, 1.0):
        cfgs.append((f"IR gamma={g}", lambda X, P, r, g=g: gain_input_region(X, P, g)))
    roles = sys.argv[1:] or ["select"]
    res = {r: run(r, cfgs, r_sel) for r in roles}
    json.dump(res, open(os.path.join(HERE, f"eval_rescale_{'_'.join(roles)}.json"), "w"), indent=1)
