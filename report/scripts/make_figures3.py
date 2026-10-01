"""Report v6 figure atlas, part A: figures from our own recorded results (no model inference).

Every number is copied from a named source: CAMPAIGN_REPORT.md §3 ledger, results/live_logs/*.txt (live scores and Kaggle
runs), experiments/estimator_blend/controls2_out.txt, experiments/robustness/swap_ablation_out.txt, report/data/experiments.json, the build tables.
The only other team used is the 1st-place score, as the maximum achieved on the board (benchmark).
"""
import os
import importlib.util as iu
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
YEL, VIO, GRN = "#eda100", "#4a3aa7", "#008300"
FIRST = dict(final=82.151, rel=94.746, tke=80.191, mvpe=94.243, time=92.030, sps=44.967)   # 1st place, 26 Sep
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)


# our live submissions with full subscores (CAMPAIGN_REPORT §3; results/live_logs)
LIVE = [  # label, rel, tke, mvpe, time, sps, final, ms/sample
    ("S3 DMD", 92.501, 75.129, 92.629, 90.866, 31.879, 77.012, 7.37),
    ("S5 FNO, no bands", 94.168, 74.026, 92.836, None, 14.080, 72.654, None),
    ("S6 +bands, η0.4", 94.189, 74.770, 92.870, 90.590, 32.397, 77.905, 7.87),
    ("S7/v4 η1.4 + modifier", 94.020, 75.725, 92.807, 91.405, 32.549, 78.033, 6.45),
    ("S9/v6 bands ×0.85", 94.020, 75.725, 92.807, 91.139, 31.559, 77.746, 6.89),
    ("v8 FNO + bands", 94.168, 74.026, 92.836, 91.409, 33.338, 78.141, 6.40),
    ("v9 diagnostics", 94.168, 74.026, 92.836, 91.235, 33.338, 78.124, 6.70),
    ("v17 blend", 94.104, 75.368, 93.085, 91.130, 37.287, 79.261, 6.90),
    ("exp1_sf clean FNO", 93.728, 77.348, 92.378, 91.701, 37.160, 79.238, 6.00),
]


def a1_subscores():
    keys = [("rel", 1), ("tke", 2), ("mvpe", 3), ("time", 4), ("sps", 5), ("final", 6)]
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.2))
    x = np.arange(len(LIVE))
    for ax, (k, j) in zip(axes.flat, keys):
        y = np.array([np.nan if r[j] is None else r[j] for r in LIVE], float)
        ax.plot(x, y, color=BLUE, lw=1.6, marker="o", ms=3.5)
        ax.axhline(FIRST[k], color=ORANGE, lw=1.2, ls="--")
        ax.text(len(LIVE) - 1, FIRST[k], "1st place ", fontsize=6, color=ORANGE, va="top", ha="right")
        ax.set_title(k, fontsize=8, color=INK)
        ax.set_xticks(x, [r[0].split()[0] for r in LIVE], fontsize=5.5, rotation=90)
        ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    save(fig, "fa01_live_subscores.png")


def a2_vs_first():
    names = ["rel", "tke", "mvpe", "time", "sps"]
    best = dict(rel=94.104, tke=75.368, mvpe=93.085, time=91.130, sps=37.287)
    clean = dict(rel=93.728, tke=77.348, mvpe=92.378, time=91.701, sps=37.160)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    for off, d, col, lab in ((-0.27, best, BLUE, "v17 (best, 79.261)"), (0.0, clean, AQUA, "exp1_sf clean model (79.238)"),
                             (0.27, FIRST, ORANGE, "1st place benchmark (82.151)")):
        v = [d[n] for n in names]
        ax.bar(x + off, v, 0.26, color=col, edgecolor="#fcfcfb", linewidth=1.5, label=lab)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 0.8, f"{vi:.1f}", ha="center", fontsize=5.8, color=INK)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="x", visible=False)
    save(fig, "fa02_vs_first.png")


def a3_score_over_time():
    import datetime as dt
    import matplotlib.dates as mdates
    days = [dt.date(2026, 8, d) for d in (8, 10, 13)] + [dt.date(2026, 9, d) for d in (25, 26)]
    ours = [78.033, 78.033, 78.141, 79.261, 79.261]
    first = [80.400, 80.792, 81.157, 82.151, 82.151]
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    ax.step(days, ours, where="post", color=BLUE, lw=2, marker="o", ms=4, label="our best live score")
    ax.plot(days, first, color=ORANGE, lw=1.5, ls="--", marker="s", ms=3.5, label="1st place over time (board snapshots)")
    for d, o in zip(days[2:4], ours[2:4]):
        ax.text(d, o - 0.55, f"{o:.2f}", ha="center", fontsize=6.5, color=INK)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", labelsize=6, rotation=45)
    ax.set_xlabel("date (2026; points are the recorded snapshots)", fontsize=6.5)
    ax.set_ylabel("final_score")
    ax.set_ylim(75.5, 83)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "fa03_score_over_time.png")


def a4_runtime():
    rows = [r for r in LIVE if r[7] is not None]
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    x = np.arange(len(rows))
    ax.bar(x, [r[7] for r in rows], color=SEQ[3], edgecolor="#fcfcfb", linewidth=1.5)
    for xi, r in zip(x, rows):
        ax.text(xi, r[7] + 0.1, f"{r[7]:.2f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=6, rotation=30, ha="right")
    ax.set_ylabel("ms per sample (board)")
    ax.set_ylim(0, 9)
    ax.grid(axis="x", visible=False)
    save(fig, "fa04_runtime.png")


RUN1 = dict(sel=[263.971, 264.652, 265.207, 266.173, 265.482, 266.069, 266.313, 266.286, 266.808, 266.470, 266.244, 266.739],
            loss=[0.1888, 0.1388, 0.1316, 0.1275, 0.1258, 0.1237, 0.1215, 0.1190, 0.1188, 0.1177, 0.1163, 0.1163],
            rel=[92.326, 92.560, 92.770, 92.950, 92.947, 93.032, 93.150, 93.170, 93.219, 93.220, 93.188, 93.236],
            tke=[77.204, 77.813, 77.931, 78.177, 78.176, 78.371, 78.284, 78.535, 78.533, 78.429, 78.417, 78.611],
            mvpe=[94.441, 94.279, 94.506, 95.045, 94.359, 94.665, 94.880, 94.580, 95.056, 94.821, 94.639, 94.892])
RUN2 = dict(sel=[264.084, 264.716, 265.311, 266.267, 265.624, 266.138, 266.356, 266.312, 266.820, 266.474, 266.273, 266.758],
            loss=[0.1852, 0.1390, 0.1314, 0.1272, 0.1255, 0.1234, 0.1212, 0.1186, 0.1184, 0.1173, 0.1159, 0.1159],
            rel=[92.393, 92.612, 92.836, 92.990, 92.990, 93.077, 93.185, 93.205, 93.254, 93.249, 93.223, 93.267],
            tke=[77.185, 77.776, 77.899, 78.180, 78.205, 78.348, 78.289, 78.521, 78.523, 78.421, 78.412, 78.600],
            mvpe=[94.506, 94.328, 94.576, 95.097, 94.429, 94.714, 94.882, 94.587, 95.043, 94.803, 94.639, 94.891])


def a5_training_curves():
    ep = np.arange(12)
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4))
    for d, col, lab in ((RUN1, BLUE, "Run 1"), (RUN2, AQUA, "Run 2 (residual)")):
        axes[0].plot(ep, d["loss"], color=col, lw=1.6, label=lab)
        axes[1].plot(ep, d["sel"], color=col, lw=1.6, marker="o", ms=2.5, label=lab)
        axes[2].plot(ep, d["tke"], color=col, lw=1.4, label=f"{lab} tke")
        axes[2].plot(ep, d["rel"], color=col, lw=1.4, ls="--", label=f"{lab} rel")
    axes[0].set_title("training loss", fontsize=7.5)
    axes[1].set_title("SELECT sum (unseen)", fontsize=7.5)
    axes[1].axvline(8, color=INK2, lw=0.8, ls=":")
    axes[1].text(8.2, 264.2, "best epoch 8", fontsize=6)
    axes[2].set_title("SELECT rel (dashed), tke (solid)", fontsize=7.5)
    for ax in axes:
        ax.set_xlabel("epoch", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, "fa05_training_curves.png")


GATE = {  # epoch -> (u, v) at leads 1, 2, 5, 10, 20 (Run 2 log)
    0: ([.15, .13, .11, .09, .10], [.46, .41, .18, .11, .05]), 1: ([.11, .08, .05, .04, .05], [.54, .47, .21, .15, .08]),
    2: ([.09, .06, .04, .03, .03], [.55, .47, .18, .13, .07]), 3: ([.08, .06, .04, .03, .03], [.57, .49, .21, .13, .08]),
    4: ([.08, .05, .03, .02, .02], [.57, .49, .19, .14, .08]), 5: ([.08, .05, .02, .01, .02], [.57, .49, .19, .13, .06]),
    6: ([.07, .05, .02, .01, .01], [.58, .50, .19, .12, .07]), 7: ([.07, .05, .02, .01, .02], [.58, .50, .19, .12, .05]),
    8: ([.07, .04, .02, .01, .01], [.58, .50, .19, .11, .06]), 9: ([.07, .04, .02, .01, .01], [.57, .49, .18, .12, .05]),
    10: ([.06, .04, .02, .01, .01], [.57, .49, .19, .12, .06]), 11: ([.06, .04, .02, .01, .01], [.57, .49, .19, .12, .05])}


def a6_gate():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    for ax, c, name in zip(axes, (0, 1), ("u", "v")):
        m = np.array([GATE[e][c] for e in range(12)]).T
        im = ax.imshow(m, aspect="auto", cmap=CMAP, vmin=0, vmax=0.6, origin="lower")
        ax.set_yticks(range(5), ["1", "2", "5", "10", "20"])
        ax.set_xlabel("epoch", fontsize=7)
        ax.set_title(f"{name}: learned weight on the last frame", fontsize=7.5)
        ax.grid(False)
    axes[0].set_ylabel("forecast lead (selected, not to scale)", fontsize=6.5)
    fig.colorbar(im, ax=axes, fraction=0.03)
    save(fig, "fa06_resid_gate.png")


def a7_finetune_deltas():
    rows = [("full fine-tune, stride 20", -0.889, 2.66, -0.618), ("9.3× windows, rel-heavy", -0.063, -0.934, -0.352),
            ("+ amplitude jitter", -0.111, -0.464, -0.403), ("Exp. 2 from official", -0.480, 2.099, -0.396),
            ("freeze-bn-fc from official", -0.032, -0.923, -0.028), ("learned corrector η1.4", -0.226, 2.711, -0.190)]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    for off, j, col, lab in ((-0.25, 1, BLUE, "Δ rel"), (0, 2, AQUA, "Δ tke"), (0.25, 3, YEL, "Δ mvpe")):
        ax.barh(y + off, [r[j] for r in rows], 0.24, color=col, label=lab)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6.8)
    ax.set_xlabel("Δ subscore vs frozen official FNO (local FINAL)")
    ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", visible=False)
    save(fig, "fa07_finetune_deltas.png")


def a8_waterfall():
    steps = [("prediction blend\n+ energy corr.", 0.31), ("interval-centre\ntable", 1.037 - 0.31), ("width table\n(= v17)", 1.33 - 1.037)]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    base = 0.0
    for i, (n, d) in enumerate(steps):
        ax.bar(i, d, bottom=base, color=[BLUE, AQUA, YEL][i], width=0.55, edgecolor="#fcfcfb")
        ax.text(i, base + d + 0.03, f"+{d:.2f}", ha="center", fontsize=7, color=INK)
        base += d
    ax.bar(3, base, color=SEQ[5], width=0.55)
    ax.text(3, base + 0.03, f"{base:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(4), [s[0] for s in steps] + ["total\n(proxy AT)"], fontsize=6.8)
    ax.set_ylabel("Δ composite vs v8")
    ax.grid(axis="x", visible=False)
    save(fig, "fa08_waterfall.png")


def a9_swap():
    rows = [("all SELECT", 1.356, 1.356), ("swap prediction", 1.241, 1.221), ("swap centre", 1.195, 1.188),
            ("swap width", 1.002, 0.593), ("swap centre+width", 0.823, 0.385), ("all random", 0.710, 0.255)]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    ax.bar(x - 0.18, [r[1] for r in rows], 0.34, color=BLUE, label="random split, k refit")
    ax.bar(x + 0.18, [r[2] for r in rows], 0.34, color=ORANGE, label="random split, SELECT's k")
    ax.set_xticks(x, [r[0] for r in rows], fontsize=6.8)
    ax.set_ylabel("Δ composite vs v8 (FINAL, AT)")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(axis="x", visible=False)
    save(fig, "fa09_swap.png")


def a10_sps_parts():
    def W(r, t, m):
        f = lambda s: 1 - (2 * (100 / s - 1)) / (0.5 + 2 * (100 / s - 1))
        return 0.5 * f(r) + 0.3 * f(t) + 0.2 * f(m)
    rows = [r for r in LIVE if r[5] is not None]
    w = [W(r[1], r[2], r[3]) for r in rows]
    T = [r[5] / 100 / wi for r, wi in zip(rows, w)]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    ax.bar(x - 0.18, w, 0.34, color=BLUE, label="accuracy factor W̄")
    ax.bar(x + 0.18, T, 0.34, color=AQUA, label="band factor T")
    for xi, ti in zip(x, T):
        ax.text(xi + 0.18, ti + 0.01, f"{ti:.2f}", ha="center", fontsize=6, color=INK)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=6, rotation=30, ha="right")
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "fa10_sps_parts.png")


def load_tabs(path):
    s = iu.spec_from_file_location("t", path); m = iu.module_from_spec(s); s.loader.exec_module(m)
    return m


def a11_width_tables():
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.6), sharex=True, sharey=True)
    for row, (path, name) in enumerate(((ROOT / "submissions/v17_per_window/_blend_tables.py", "v17pw (official FNO)"),
                                        (ROOT / "submissions/exp1_sf/_blend_tables.py", "exp1_sf (clean FNO, ×√scale)"))):
        H = np.asarray(load_tabs(path).WIDTH_TAB)
        for c in (0, 1):
            ax = axes[row, c]
            im = ax.imshow(H[:, c, :], aspect="auto", cmap=CMAP, origin="lower",
                           extent=(0.5, H.shape[2] + 0.5, -0.5, 3.5))
            ax.set_title(f"{name}: {'u' if c == 0 else 'v'}", fontsize=7)
            ax.set_yticks(range(4), ["1–5", "6–10", "11–15", "16–20"], fontsize=6)
            ax.grid(False)
            cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.ax.tick_params(labelsize=5.5)
            cb.set_label("half-width h (velocity units)" if row == 0 else "c in h = c·√s_w (s_w: window median σ_local)",
                         fontsize=5.2)
    for ax in axes[1]:
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("input frame-difference decile (1 = calmest, 10 = most active)", fontsize=6.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("lead group", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fa11_width_tables.png")


def a12_transfer_scatter():
    pts = [("cond.-TKE map", 3.25, -1.02, (4, -9), "left"), ("DMD→FNO swap", 8.05, 0.56, (-4, 6), "right"),
           ("log-POD", 2.73, 0.03, (3, 5), "left"), ("rank modifier (SPS)", 6.96, 0.15, (-4, -10), "right"),
           ("v17, proxy AT (SPS; proxy chosen post hoc)", 4.01, 3.95, (-5, 4), "right"),
           ("exp1_sf, SELECT", -0.75, -0.02, (0, 8), "center"), ("exp1_sf, FINAL", 0.80, -0.02, (2, -11), "left")]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot([-2, 9], [-2, 9], color=INK2, lw=0.8, ls=":")
    for n, lx, ly, off, ha in pts:
        col = AQUA if "exp1" in n else (GRN if "v17" in n else BLUE)
        ax.scatter(lx, ly, s=30, color=col, edgecolor="#fcfcfb", zorder=3)
        ax.annotate(n, (lx, ly), xytext=off, textcoords="offset points", fontsize=6.3, color=INK, ha=ha)
    ax.set_xlabel("predicted locally")
    ax.set_ylabel("measured live")
    ax.set_xlim(-2, 9); ax.set_ylim(-2, 5)
    save(fig, "fa12_transfer_scatter.png")


def a13_splits():
    split = json.loads((ROOT / "data/splits/holdout_v12_3.json").read_text())
    pool = set(json.loads((ROOT / "data/splits/train_pool_v12_3.json").read_text())["pool"])
    names = [f[:-3] for f in __import__("os").listdir(os.environ.get("RPDE_TRAIN_REAL", "data/train_real")) if f.endswith(".h5")]
    res = sorted({int(n.split("_")[0]) for n in names}); aoas = [0, 5, 10, 15, 20]
    col = {"train": SEQ[2], "select": ORANGE, "final": AQUA, "dup": INK2}
    fig, ax = plt.subplots(figsize=(6.8, 2.2))
    for n in names:
        re_, a = int(n.split("_")[0]), int(n.split("_")[1])
        k = "dup" if n == "7575_0" else ("select" if n in split["select"] else ("final" if n in split["final"] else "train"))
        ax.scatter(res.index(re_), aoas.index(a), s=48, marker="s", color=col[k])
    ax.axvspan(res.index(21600) - 0.5, res.index(25425) + 0.5, ymin=0.6, ymax=0.8, color=ORANGE, alpha=0.08)
    ax.set_xticks(range(len(res)), res, fontsize=5.5, rotation=90)
    ax.set_yticks(range(5), [f"AoA {a}" for a in aoas], fontsize=6.5)
    for k, lab in (("train", "training (67)"), ("select", "SELECT (4, live-like)"), ("final", "FINAL (10)"), ("dup", "duplicate 7575_0")):
        ax.scatter([], [], s=30, marker="s", color=col[k], label=lab)
    ax.legend(frameon=False, fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    ax.set_xlabel("Reynolds-number label (file name)", fontsize=6.5)
    ax.grid(False)
    save(fig, "fa13_splits.png")


def a14_final_to_live():
    names = ["rel", "tke", "mvpe"]
    off = [-2.21, -7.01, -4.23]; clean = [-0.43, -1.83, -3.13]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(5.0, 2.3))
    ax.bar(x - 0.18, off, 0.34, color=ORANGE, label="official FNO (saw FINAL)")
    ax.bar(x + 0.18, clean, 0.34, color=AQUA, label="our clean FNO (never saw FINAL)")
    for xi, a, b in zip(x, off, clean):
        ax.text(xi - 0.18, a - 0.45, f"{a:.2f}", ha="center", fontsize=6.5, color=INK)
        ax.text(xi + 0.18, b - 0.45, f"{b:.2f}", ha="center", fontsize=6.5, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, names)
    ax.set_ylabel("live − local FINAL")
    ax.set_ylim(-8.2, 0.8)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", visible=False)
    save(fig, "fa14_final_to_live.png")


if __name__ == "__main__":
    for f in (a1_subscores, a2_vs_first, a4_runtime, a5_training_curves, a6_gate, a7_finetune_deltas,
              a8_waterfall, a9_swap, a10_sps_parts, a11_width_tables, a12_transfer_scatter, a13_splits, a14_final_to_live):
        f()
    print("part A written")
