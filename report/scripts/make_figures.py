"""Figures for the RealPDE Track 1 technical report (team Slash).

Every number plotted here is either a live Codabench result, the public
leaderboard snapshot in lb_snapshot.json / lb_analysis.json, or the shipped
v17 blend tables. Run from the repo root:  python report/scripts/make_figures.py
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
sys.path.insert(0, str(ROOT / "submissions" / "v17"))
from _blend_tables import PRED_TAB  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 200, "savefig.bbox": "tight",
})
lb = json.loads((DATA / "lb_analysis.json").read_text())


def fig_live_history():
    subs = [("S1", "DMD", 76.6845), ("S3", "DMD, tuned bands", 77.0117),
            ("S4", "+log-POD", 76.4927), ("S5", "FNO, no bands", 72.6542),
            ("S6", "FNO+corrector η=0.4", 77.9045), ("S7", "FNO+corrector η=1.4", 78.0331),
            ("S9", "bands ×0.85", 77.7459), ("v8", "stock FNO + bands", 78.1410),
            ("v9", "diagnostic", 78.1245), ("v12", "modifier off", 77.899), ("v17", "estimator blend", 79.2608), ("sf", "clean sim→real FNO", 79.2379)]
    x = np.arange(len(subs))
    y = [s[2] for s in subs]
    best = np.maximum.accumulate(y)
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.step(x, best, where="post", color=GRID, lw=2, zorder=1)
    ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=5, zorder=2,
            markeredgecolor="#fcfcfb", markeredgewidth=1.5)
    for i, (lab, desc, v) in enumerate(subs):
        if lab in ("S5", "v8", "v17", "S1", "sf"):
            off, ha = ((10, -4), "left") if lab == "S5" else (((0, -24), "center") if lab == "sf" else ((0, 8), "center"))
            ax.annotate(f"{desc}\n{v:.2f}", (i, v), xytext=off,
                        textcoords="offset points", ha=ha, fontsize=7, color=INK)
    ax.axhline(82.151, color=ORANGE, lw=1.2, ls="--")
    ax.text(len(subs) - 0.6, 82.151 + 0.15, "1st place (benchmark) 82.15",
            ha="right", fontsize=7, color=INK)
    ax.set_xticks(x, [s[0] for s in subs])
    ax.set_ylabel("live final_score")
    ax.set_ylim(71.5, 83)
    fig.savefig(FIG / "fig1_live_history.png")
    plt.close(fig)


def fig_T_by_rank():
    b = lb["buckets"]
    labels = [d["ranks"] for d in b]
    T = [d["T"] for d in b]
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    bars = ax.bar(labels, T, color=SEQ[3], width=0.62, edgecolor="#fcfcfb", linewidth=2)
    for r, v in zip(bars, T):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7.5, color=INK)
    for val, name, col in ((0.491, "ours, v8", ORANGE), (lb["T_v17"], "ours, v17", AQUA)):
        ax.axhline(val, color=col, lw=1.4, ls="--")
        ax.text(4.45, val + 0.008, f"{name}: {val:.3f}", ha="right", fontsize=7, color=INK)
    ax.set_ylabel("band factor T = SPS / (100·W̄)")
    ax.set_xlabel("leaderboard rank bucket (178 scored teams)")
    ax.set_ylim(0, 0.7)
    fig.savefig(FIG / "fig2_T_by_rank.png")
    plt.close(fig)


def fig_gap():
    g = lb["gap_contrib"]
    names = ["sps", "tke", "rel", "time", "mvpe"]
    vals = [g[n] for n in names]
    fig, ax = plt.subplots(figsize=(6.6, 2.0))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=BLUE, height=0.6, edgecolor="#fcfcfb", linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v + 0.03, yi, f"+{v:.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlabel("composite points needed to match the top-10 mean (fitted weights)")
    ax.set_xlim(0, 2.4)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "fig3_gap.png")
    plt.close(fig)


def fig_blend_weights():
    tab = np.asarray(PRED_TAB)                      # (lead 20, channel 2, sextile 6, [a, b])
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.2), sharex=True, sharey=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    for c in (0, 1):
        for k, (sym, what) in enumerate((("α", "FNO forecast"), ("β", "last input frame"))):
            ax = axes[k, c]
            im = ax.imshow(tab[:, c, :, k].T, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1,
                           extent=(0.5, 20.5, 0.5, 6.5))
            ax.set_title(f"{'u' if c == 0 else 'v'}: weight {sym} on the {what}", fontsize=7.6, color=INK)
            ax.grid(False)
    for ax in axes[1]:
        ax.set_xlabel("forecast lead (frames)")
    for ax in axes[:, 0]:
        ax.set_ylabel("σ_local sextile (1 = calmest)")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("weight (window mean gets 1 − α − β)", fontsize=7)
    fig.savefig(FIG / "fig4_blend_alpha.png")
    plt.close(fig)


def fig_proxy_vs_live():
    names = ["rel", "tke", "mvpe", "SPS"]
    proxy = [0.07, 1.75, 0.06, 4.01]    # proxy A, k=1.447 (experiments/estimator_blend report)
    live = [-0.064, 1.342, 0.249, 3.949]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    ax.bar(x - 0.18, proxy, 0.34, color=BLUE, edgecolor="#fcfcfb", linewidth=2, label="proxy AT (one of four pre-submission proxies; chosen after the live result)")
    ax.bar(x + 0.18, live, 0.34, color=ORANGE, edgecolor="#fcfcfb", linewidth=2, label="live (measured)")
    for xi, p, l in zip(x, proxy, live):
        ax.text(xi - 0.18, max(p, 0) + 0.08, f"{p:+.2f}", ha="center", fontsize=7, color=INK)
        ax.text(xi + 0.18, max(l, 0) + 0.08, f"{l:+.2f}", ha="center", fontsize=7, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, names)
    ax.set_ylabel("Δ subscore, v17 − v8")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_ylim(-0.5, 4.8)
    fig.savefig(FIG / "fig5_proxy_vs_live.png")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_live_history, fig_blend_weights, fig_proxy_vs_live):
        f()
    print("figures written to", HERE)
