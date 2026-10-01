"""Data and experiment figures for report v2 (team Slash).

Rebuilds the shipped v17 forecast in numpy from the cached FNO predictions and the
shipped tables (submissions/v17/_blend_tables.py), so every figure shows what was
actually submitted. Run from the repo root:  python report/scripts/make_figures2.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG, DATA = HERE.parent / "figures", HERE.parent / "data"   # report/figures, report/data
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "experiments" / "estimator_blend"), str(ROOT / "experiments" / "interval_recipes"), str(ROOT / "submissions" / "v17")]
from base import boost, estimators, rank_bins  # noqa: E402
from common import sigma_local  # noqa: E402
from _blend_tables import PRED_TAB, WIDTH_TAB  # noqa: E402
from make_figures import AQUA, BLUE, INK, INK2, ORANGE, SEQ  # noqa: E402  (shared style)

K_PROXY = 1.447
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
DIV = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b8431a"])

d = np.load(ROOT / "experiments/interval_recipes/cache_final_s20.npz")
X, Y, P, traj = d["X"], d["Y"], d["P"], d["traj"]
_L_PATH = ROOT / "experiments/v19_ridge_partner/L_sxb3_final.npy"
L = np.load(_L_PATH) if _L_PATH.exists() else None   # only fig 6 needs it


def v17_forecast(X, P):
    est, valid = estimators(X)
    em, el = est["mean"][:, None], est["last"][:, None]
    sb = rank_bins(sigma_local(X), valid, 6)
    tab = np.asarray(PRED_TAB)                               # (T, C, 6, 2)
    ch = np.arange(2)[None, None, None, :]
    a = np.stack([tab[t][ch, sb] for t in range(20)], 1)  # (n,T,H,W,2,2)
    out = em + a[..., 0] * (P - em) + a[..., 1] * (el - em)
    out = np.where(valid[:, None], out, P).astype(np.float32)
    return boost(out, X), valid


def v17_halfwidth(X, valid):
    p1 = np.sqrt(((X[:, 1:] - X[:, :-1]) ** 2).mean(1)).astype(np.float32)
    pb = rank_bins(p1, valid, len(WIDTH_TAB[0][0]))
    H = np.asarray(WIDTH_TAB)
    ch = np.arange(2)[None, None, None, :]
    return np.stack([H[t // 5][ch, pb] for t in range(20)], 1)


V17, valid = v17_forecast(X, P)
Yp = P + K_PROXY * (Y - P)                                   # proxy A (live-calibrated) target
scored = Y != 0


def rel_by_lead(pred, tgt):
    num = (((pred - tgt) ** 2) * scored).sum(axis=(0, 2, 3, 4))
    den = ((tgt ** 2) * scored).sum(axis=(0, 2, 3, 4))
    return np.sqrt(num / den)


def fig_error_by_lead():
    est, _ = estimators(X)
    cands = {"FNO (official)": (P, BLUE), "v17 blend": (V17, AQUA), "ridge forecaster (v19)": (L, "#4a3aa7"),
             "persistence": (np.repeat(est["last"][:, None], 20, 1), ORANGE),
             "window mean": (np.repeat(est["mean"][:, None], 20, 1), "#eda100")}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=False)
    for ax, (tgt, title) in zip(axes, ((Y, "local target (FNO may have seen it in training)"),
                                       (Yp, f"proxy AT target, k = {K_PROXY}"))):
        ends = []
        for name, (pred, col) in cands.items():
            e = rel_by_lead(pred, tgt)
            ax.plot(np.arange(1, 21), e, color=col, lw=1.8)
            ends.append([e[-1], name])
        ends.sort()
        gap = 0.016
        for j in range(1, len(ends)):                       # de-collide end labels
            ends[j][0] = max(ends[j][0], ends[j - 1][0] + gap)
        for yv, name in ends:
            ax.text(20.4, yv, name, fontsize=6.3, color=INK, va="center")
        ax.set_title(title, fontsize=7.8, color=INK)
        ax.set_xlabel("forecast lead (frames)")
        ax.set_xlim(1, 27)
    axes[0].set_ylabel("relative L2 error (u, v)")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_error_by_lead.png")
    plt.close(fig)


def fig_field_panels():
    per = np.sqrt((((V17 - Y) ** 2) * scored).sum((1, 2, 3, 4)) / ((Y ** 2) * scored).sum((1, 2, 3, 4)))
    i = int(np.argsort(per)[len(per) // 2])                  # median-difficulty window
    t, c = 9, 0                                              # lead 10, u channel
    h = v17_halfwidth(X, valid)
    m = ~scored[i, t, ..., c]
    panels = [(X[i, -1, ..., c], "last input frame, u", CMAP, None),
              (Y[i, t, ..., c], "target u, lead 10", CMAP, None),
              (np.abs(P[i, t, ..., c] - Y[i, t, ..., c]), "|FNO − target|", CMAP, "err"),
              (np.abs(V17[i, t, ..., c] - Y[i, t, ..., c]), "|v17 − target|", CMAP, "err"),
              (np.abs(P[i, t, ..., c] - Y[i, t, ..., c]) - np.abs(V17[i, t, ..., c] - Y[i, t, ..., c]),
               "FNO err − v17 err (blue: v17 better)", DIV, "div"),
              (h[i, t, ..., c], "v17 interval half-width", CMAP, None)]
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.4))
    emax = np.percentile(np.abs(P[i, t, ..., c] - Y[i, t, ..., c])[~m], 99)
    for ax, (img, title, cmap, kind) in zip(axes.flat, panels):
        img = np.ma.array(img, mask=m)
        if kind == "err":
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=emax, origin="lower")
        elif kind == "div":
            im = ax.imshow(-img, cmap=DIV, vmin=-emax / 2, vmax=emax / 2, origin="lower")
        else:
            im = ax.imshow(img, cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=7.3, color=INK)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(labelsize=5.5)
    fig.suptitle(f"FINAL window {i} (trajectory {traj[i]}); masked cells blank", fontsize=7.5, color=INK2, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_field_panels.png")
    plt.close(fig)
    return i


def fig_transfer():
    rows = [("conditional-TKE map", 3.25, -1.02), ("DMD → FNO swap", 8.05, 0.56),
            ("log-POD denoise", 2.73, 0.03), ("rank band modifier (SPS)", 6.96, 0.15),
            ("v17 under proxy AT (SPS)", 4.01, 3.95)]
    fig, ax = plt.subplots(figsize=(6.8, 2.5))
    y = np.arange(len(rows))[::-1]
    for yi, (name, loc, live) in zip(y, rows):
        col = AQUA if "proxy" in name else ORANGE
        ax.plot([loc, live], [yi, yi], color="#c9c8c3", lw=2, zorder=1)
        ax.scatter([loc], [yi], s=34, color=BLUE, zorder=2, edgecolor="#fcfcfb", linewidth=1.2)
        ax.scatter([live], [yi], s=34, color=col, zorder=3, edgecolor="#fcfcfb", linewidth=1.2)
        ax.text(max(loc, live) + 0.25, yi, f"{100 * live / loc:.0f}% transfer", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("effect in composite or SPS points (as labelled): blue = predicted locally, orange/aqua = measured live", fontsize=7)
    ax.set_xlim(-2, 10.5)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "fig8_transfer.png")
    plt.close(fig)


def fig_generalisation():
    g = json.loads((ROOT / "results/generalisation_map.json").read_text())
    live = dict(rel=94.168, tke=74.026, mvpe=92.836)
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2))
    for ax, k in zip(axes, ("rel", "tke", "mvpe")):
        v = np.array([r[k] for r in g])
        ax.hist(v, bins=18, color=SEQ[3], edgecolor="#fcfcfb", linewidth=1)
        ax.axvline(live[k], color=ORANGE, lw=1.6, ls="--")
        ax.text(live[k], ax.get_ylim()[1] * 0.92, f" live {live[k]:.1f}", fontsize=6.8, color=INK)
        ax.set_title(f"{k} subscore", fontsize=7.8, color=INK)
        ax.set_xlabel("score on a training trajectory")
    axes[0].set_ylabel("trajectories (of 81)")
    fig.tight_layout()
    fig.savefig(FIG / "fig9_generalisation.png")
    plt.close(fig)


def fig_band_frontier():
    rows = [("shipped A + K·σ_local (v4–v6)", 0.4653), ("best flat width h = 0.009", 0.5217),
            ("ceiling: σ_local alone", 0.5417), ("ceiling: |prediction| alone", 0.5535),
            ("ceiling: all three features", 0.5663), ("oracle h = |error| (needs target)", 0.7851)]
    fig, ax = plt.subplots(figsize=(6.8, 2.2))
    y = np.arange(len(rows))[::-1]
    cols = [SEQ[3]] * 5 + ["#c9c8c3"]
    ax.barh(y, [r[1] for r in rows], color=cols, height=0.62, edgecolor="#fcfcfb", linewidth=2)
    for yi, (_, v) in zip(y, rows):
        ax.text(v + 0.006, yi, f"{v:.3f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("band factor T on 84 held-out windows (memorised split)")
    ax.set_xlim(0, 0.88)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "fig10_band_frontier.png")
    plt.close(fig)


if __name__ == "__main__":
    if L is not None:   # needs the v19 forecast cache
        fig_error_by_lead()
    w = fig_field_panels()
    fig_transfer()
    fig_generalisation()
    fig_band_frontier()
    print("done; field-panel window", w)
