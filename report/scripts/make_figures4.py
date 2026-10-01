"""Report v6 figure atlas, part B: our clean sim->real FNO (Run 1) vs the official FNO on the same windows.

Inputs: experiments/interval_recipes/cache_{select,final}_s20.npz (X, Y, P = official FNO) and report/data/run1_preds.npz.
Run 1 never saw SELECT or FINAL (honest); the official FNO probably did (Table 2), so its local errors are optimistic.
Processes one split at a time (low RAM).
"""
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_figures import AQUA, BLUE, INK, INK2, ORANGE, SEQ

HERE = Path(__file__).resolve().parent
FIG, DATA = HERE.parent / "figures", HERE.parent / "data"   # report/figures, report/data
ROOT = HERE.parents[1]
YEL = "#eda100"
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
R1 = np.load(DATA / "run1_preds.npz")
res = {}


def rel_by_lead(pred, Y, m):
    return np.sqrt((((pred - Y) ** 2) * m).sum((0, 2, 3, 4)) / ((Y ** 2) * m).sum((0, 2, 3, 4)))


def energy_ratio(pred, Y):
    fp = pred - pred.mean(1, keepdims=True); fy = Y - Y.mean(1, keepdims=True)
    return (fp ** 2).mean((0, 2, 3, 4)) / (fy ** 2).mean((0, 2, 3, 4))


def radial_spectrum(f):   # f: (n, H, W) -> radial power
    F = np.abs(np.fft.fft2(f - f.mean((1, 2), keepdims=True))) ** 2
    H, W = f.shape[1:]
    ky, kx = np.meshgrid(np.fft.fftfreq(H) * H, np.fft.fftfreq(W) * W, indexing="ij")
    k = np.sqrt(kx ** 2 + ky ** 2).astype(int)
    return np.bincount(k.ravel(), F.mean(0).ravel())[:16]


for role in ("select", "final"):
    d = np.load(ROOT / f"experiments/interval_recipes/cache_{role}_s20.npz")
    X, Y, P, T = d["X"], d["Y"], d["P"], d["traj"]
    Q = R1[role].astype(np.float32)
    m = (Y != 0).astype(np.float32)
    em = X.mean(1, keepdims=True); el = X[:, -1:]
    r = dict(off=rel_by_lead(P, Y, m), run1=rel_by_lead(Q, Y, m), mean=rel_by_lead(np.repeat(em, 20, 1), Y, m),
             last=rel_by_lead(np.repeat(el, 20, 1), Y, m), e_off=energy_ratio(P, Y), e_run1=energy_ratio(Q, Y))
    per = {}
    for t in sorted(set(T)):
        i = T == t
        e = lambda A: float(np.sqrt((((A[i] - Y[i]) ** 2) * m[i]).sum() / ((Y[i] ** 2) * m[i]).sum()))
        per[t] = (e(P), e(Q))
    r["per"] = per
    lead = 9
    r["spec"] = {k: radial_spectrum(A[:, lead, :, :, 0]) for k, A in (("target", Y), ("official", P), ("run1", Q))}
    if role == "select":
        tot = np.sqrt((((Q - Y) ** 2) * m).sum((1, 2, 3, 4)) / ((Y ** 2) * m).sum((1, 2, 3, 4)))
        w = int(np.argsort(tot)[len(tot) // 2])
        r["panel"] = dict(w=w, traj=str(T[w]), Y=Y[w], P=P[w], Q=Q[w], mask=m[w])
    res[role] = r
    del d, X, Y, P, Q, m, em, el; gc.collect()


def b1_error_by_lead():
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6), sharey=True)
    for ax, role, title in zip(axes, ("select", "final"), ("SELECT (AoA 15, live regime)", "FINAL (mixed AoA)")):
        r = res[role]; lead = np.arange(1, 21)
        for k, col, lab in (("off", ORANGE, "official FNO*"), ("run1", BLUE, "our clean FNO"),
                            ("last", INK2, "persistence"), ("mean", YEL, "window mean")):
            ax.plot(lead, r[k], color=col, lw=1.7, label=lab)
        ax.set_title(title, fontsize=7.8)
        ax.set_xlabel("forecast lead (frames)", fontsize=7)
    axes[0].set_ylabel("relative L2 error")
    axes[0].legend(frameon=False, fontsize=6.3)
    fig.text(0.5, -0.03, "*the official FNO very likely trained on these trajectories, so its local error is optimistic",
             ha="center", fontsize=6, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "fb01_error_by_lead.png"); plt.close(fig)


def b2_panels():
    p = res["select"]["panel"]; c, t = 0, 9
    Y, P, Q, m = p["Y"], p["P"], p["Q"], p["mask"][..., c] < 0.5
    var = lambda A: A[..., c].var(0)
    emax = np.percentile(np.abs(Q[t, ..., c] - Y[t, ..., c])[~m[t]], 99)
    vmax = np.percentile(var(Y)[~m.any(0)], 99)
    panels = [(Y[t, ..., c], "target u, lead 10", None), (np.abs(P[t, ..., c] - Y[t, ..., c]), "|official − target|*", emax),
              (np.abs(Q[t, ..., c] - Y[t, ..., c]), "|our clean FNO − target|", emax),
              (var(Y), "target temporal variance (tke map)", vmax), (var(P), "official FNO variance*", vmax),
              (var(Q), "our clean FNO variance", vmax)]
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 3.4))
    for k, (ax, (img, title, vm)) in enumerate(zip(axes.flat, panels)):
        mm = m[t] if k < 3 else m.any(0)
        im = ax.imshow(np.ma.array(img, mask=mm), cmap=CMAP, origin="lower", vmin=0 if vm else None, vmax=vm)
        ax.set_title(title, fontsize=6.8); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).ax.tick_params(labelsize=5)
    fig.suptitle(f"SELECT window {p['w']} ({p['traj']}, live regime); * official FNO probably saw this trajectory",
                 fontsize=7, color=INK2, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fb02_panels.png"); plt.close(fig)


def b3_per_traj():
    rows = [(t, *res[r]["per"][t], r) for r in ("select", "final") for t in res[r]["per"]]
    rows.sort(key=lambda z: (z[3] != "select", z[0]))
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.9, 2.5))
    ax.bar(x - 0.18, [z[1] for z in rows], 0.34, color=ORANGE, label="official FNO*")
    ax.bar(x + 0.18, [z[2] for z in rows], 0.34, color=BLUE, label="our clean FNO")
    ax.axvline(3.5, color=INK2, lw=0.8, ls=":")
    ax.text(1.5, ax.get_ylim()[1] * 0.95, "SELECT", ha="center", fontsize=6.5)
    ax.set_xticks(x, [z[0] for z in rows], fontsize=6, rotation=60)
    ax.set_ylabel("rel-L2 error (lower is better)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "fb03_per_traj.png"); plt.close(fig)


def b4_energy():
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    lead = np.arange(1, 21)
    for role, ls in (("select", "-"), ("final", "--")):
        ax.plot(lead, res[role]["e_off"], color=ORANGE, lw=1.6, ls=ls, label=f"official FNO, {role}")
        ax.plot(lead, res[role]["e_run1"], color=BLUE, lw=1.6, ls=ls, label=f"our clean FNO, {role}")
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("forecast lead (frames)"); ax.set_ylabel("fluctuation energy, pred / target")
    ax.legend(frameon=False, fontsize=6.2, ncol=2)
    fig.savefig(FIG / "fb04_energy.png"); plt.close(fig)


def b5_spectrum():
    s = res["select"]["spec"]
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    for k, col in (("target", INK), ("official", ORANGE), ("run1", BLUE)):
        ax.semilogy(np.arange(1, 16), s[k][1:16], color=col, lw=1.7, label={"target": "target", "official": "official FNO*",
                                                                            "run1": "our clean FNO"}[k])
    ax.set_xlabel("spatial wavenumber |k|"); ax.set_ylabel("power (u, lead 10)")
    ax.legend(frameon=False, fontsize=6.5)
    fig.savefig(FIG / "fb05_spectrum.png"); plt.close(fig)


for f in (b1_error_by_lead, b2_panels, b3_per_traj, b4_energy, b5_spectrum):
    f()
json.dump({r: {"per": res[r]["per"], "off": res[r]["off"].tolist(), "run1": res[r]["run1"].tolist()} for r in res},
          open(DATA / "partB_numbers.json", "w"), indent=1)
print("part B written")
