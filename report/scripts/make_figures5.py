"""Report v6, part D: data figures from our own audit (report/data/h5_meta.json, read from the released
train_real .h5 files) and our logged live-input diagnostics (CAMPAIGN_REPORT.md §5, §13.2–13.3)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_figures import AQUA, BLUE, INK, INK2, ORANGE

HERE = Path(__file__).resolve().parent
FIG, DATA = HERE.parent / "figures", HERE.parent / "data"   # report/figures, report/data
M = json.loads((DATA / "h5_meta.json").read_text())
AOA_COL = {0: "#2a78d6", 5: "#1baf7a", 10: "#eda100", 15: "#eb6834", 20: "#4a3aa7"}


def d1_lengths():
    fig, ax = plt.subplots(figsize=(6.8, 2.3))
    for r in M:
        lab = int(r["name"].split("_")[0])
        ax.scatter(lab, r["T"], s=18, color=AOA_COL[int(r["aoa"])], zorder=3)
    for a, c in AOA_COL.items():
        ax.scatter([], [], s=18, color=c, label=f"AoA {a}")
    ax.set_xlabel("Reynolds label (file name)"); ax.set_ylabel("frames per trajectory")
    ax.legend(frameon=False, fontsize=6.3, ncol=5, loc="lower right")
    fig.savefig(FIG / "fd01_lengths.png"); plt.close(fig)


def d2_re_mismatch():
    lab = np.array([int(r["name"].split("_")[0]) for r in M]); att = np.array([r["re"] for r in M])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
    axes[0].scatter(lab, att, s=14, color=BLUE)
    axes[0].plot([3000, 27500], [3000, 27500], color=INK2, lw=0.8, ls=":")
    axes[0].set_xlabel("Re in file name"); axes[0].set_ylabel("Re attribute in file")
    axes[1].scatter(lab, att - lab, s=14, color=ORANGE)
    axes[1].axhline(0, color=INK2, lw=0.8)
    axes[1].set_xlabel("Re in file name"); axes[1].set_ylabel("attribute − name")
    axes[1].set_title(f"{int((abs(att - lab) > 1).sum())} of {len(M)} disagree", fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "fd02_re_mismatch.png"); plt.close(fig)


def d3_mask():
    fig, ax = plt.subplots(figsize=(6.8, 2.2))
    for r in M:
        ax.scatter(int(r["name"].split("_")[0]), 100 * r["zero_frac"], s=16, color=AOA_COL[int(r["aoa"])])
    ax.axhline(100 * np.mean([r["zero_frac"] for r in M]), color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("Reynolds label"); ax.set_ylabel("masked pixels, first frame (%)")
    fig.savefig(FIG / "fd03_mask.png"); plt.close(fig)


def d4_fingerprint():
    rows = [("u mean", 0.21810, 0.15194), ("u std", 0.15039, 0.10972), ("v std", 0.02323, 0.01506),
            ("σ_local u", 0.016263, 0.00809), ("σ_local v", 0.007310, 0.00480), ("frame diff", 0.014485, 0.00782),
            ("τ", 2.110, 2.345), ("n_eff", 4.95, 4.264), ("zero fraction", 0.0900, 0.0938)]
    x = np.arange(len(rows)); ratio = [a / b for _, a, b in rows]
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    ax.bar(x, ratio, color=[ORANGE if r > 1.3 else BLUE for r in ratio], edgecolor="#fcfcfb")
    for xi, r in zip(x, ratio):
        ax.text(xi, r + 0.03, f"{r:.2f}", ha="center", fontsize=6.5, color=INK)
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xticks(x, [r[0] for r in rows], fontsize=6.5, rotation=20)
    ax.set_ylabel("live / local reference")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "fd04_fingerprint.png"); plt.close(fig)


def d5_live_diagnostics():
    spec_live = [1477.5, 129.3, 30.0, 14.7]; gain = [1.00, 1.08, 0.98, 0.95]
    ac_live = [0.686, 0.417, 0.146]; ac_loc = [0.697, 0.426, 0.147]; ar1 = [0.686, 0.686 ** 2, 0.686 ** 3]
    sig = [0.252, 0.281, 0.151, 0.121, 0.097, 0.098]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3))
    axes[0].semilogy(range(1, 5), spec_live, color=ORANGE, marker="o", label="live")
    axes[0].semilogy(range(1, 5), [s / g for s, g in zip(spec_live, gain)], color=BLUE, marker="s", ls="--", label="local")
    axes[0].set_title("u radial spectrum (4 bins)", fontsize=7); axes[0].legend(frameon=False, fontsize=6)
    axes[1].plot([1, 2, 3], ac_live, color=ORANGE, marker="o", label="live")
    axes[1].plot([1, 2, 3], ac_loc, color=BLUE, marker="s", ls="--", label="local")
    axes[1].plot([1, 2, 3], ar1, color=INK2, ls=":", label="AR(1) from lag 1")
    axes[1].set_title("temporal autocorrelation", fontsize=7); axes[1].legend(frameon=False, fontsize=6)
    axes[1].set_xlabel("lag (frames)", fontsize=6.5)
    axes[2].bar(range(6), sig, color=AQUA)
    axes[2].set_xticks(range(6), ["<.003", ".008", ".016", ".028", ".045", ">"], fontsize=5.5)
    axes[2].set_title("live σ_local(u) histogram", fontsize=7)
    for ax in axes:
        ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(FIG / "fd05_live_diag.png"); plt.close(fig)


for f in (d1_lengths, d2_re_mismatch, d3_mask, d4_fingerprint, d5_live_diagnostics):
    f()
print("part D written")
