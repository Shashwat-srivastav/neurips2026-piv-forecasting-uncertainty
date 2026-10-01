"""Evidence figures from experiments/robustness: K9 over many random splits (k9_multi.json) and baseline arms with
trajectory-bootstrap CIs (arms_bootstrap.json). A figure whose JSON is missing is skipped."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_figures import AQUA, BLUE, INK, INK2, ORANGE

HERE = Path(__file__).resolve().parent
FIG, DATA = HERE.parent / "figures", HERE.parent / "data"   # report/figures, report/data
A2 = HERE.parents[1] / "experiments" / "robustness"
YEL, GREY = "#eda100", "#8d8b85"


def k9():
    p = A2 / "k9_multi.json"
    if not p.exists():
        return print("k9_multi.json missing, skipped")
    R = json.loads(p.read_text())
    ref = np.array([r["refit_k"]["comp"] for r in R.values()]); sel = np.array([r["select_k"]["comp"] for r in R.values()])
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.5), gridspec_kw=dict(width_ratios=[1.55, 1]))
    rng = np.random.default_rng(1)
    rows = [("tables fitted on SELECT\n(fingerprint-selected)", np.array([1.356]), ORANGE),
            ("random split,\nproxy k refitted", ref, BLUE), ("random split,\nSELECT's k", sel, AQUA)]
    for i, (lab, v, c) in enumerate(rows):
        y = 2 - i
        a.scatter(v, y + rng.uniform(-0.13, 0.13, len(v)), s=18, color=c, edgecolor="white", lw=0.5, zorder=3)
        if len(v) > 1:
            a.plot([v.min(), v.max()], [y - 0.3, y - 0.3], color=c, lw=1.2)
            a.scatter([v.mean()], [y - 0.3], marker="|", s=90, color=INK, zorder=4)
            a.text(v.max() + 0.05, y - 0.3, f"mean {v.mean():+.2f}  (n={len(v)})", va="center", fontsize=6, color=INK2)
        else:
            a.text(v[0] + 0.06, y, f"{v[0]:+.2f}", va="center", fontsize=6.5, color=INK)
    a.axvline(0, color=INK2, lw=0.8, ls=":")
    a.set_yticks([2, 1, 0], [r[0] for r in rows], fontsize=6.4)
    a.set_xlabel("Δ composite vs v8 on FINAL (proxy AT)", fontsize=6.8); a.set_ylim(-0.7, 2.5)
    a.grid(axis="y", visible=False)
    sh_r = 1 - ref / 1.356; sh_s = 1 - sel / 1.356
    bins = np.linspace(0, max(1.3, sh_s.max() + 0.05), 14)
    b.hist(sh_r, bins=bins, color=BLUE, alpha=0.85, label="k refitted")
    b.hist(sh_s, bins=bins, color=AQUA, alpha=0.6, label="SELECT's k")
    b.set_xlabel("share of v17's local gain that\nneeds the regime-matched fit", fontsize=6.6)
    b.set_ylabel("random splits", fontsize=6.6); b.legend(frameon=False, fontsize=6)
    b.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "fe01_k9_multi.png"); plt.close(fig)
    print(f"k9: n={len(ref)} share refit {sh_r.min():.2f}-{sh_r.max():.2f} (median {np.median(sh_r):.2f}); "
          f"SELECT k {sh_s.min():.2f}-{sh_s.max():.2f} (median {np.median(sh_s):.2f})")


def arms():
    p = A2 / "arms_bootstrap.json"
    if not p.exists():
        return print("arms_bootstrap.json missing, skipped")
    R = json.loads(p.read_text())
    names = [("official", "official FNO*", ORANGE), ("clean", "our clean FNO", AQUA),
             ("persistence", "persistence", GREY), ("window_mean", "window mean", YEL)]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.45), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    for ax, role, title in zip(axes[:2], ("select", "final"), ("SELECT (live regime)", "FINAL")):
        r = R[role]; x = np.arange(3)
        for j, (k, lab, c) in enumerate(names):
            pt = np.array(r["point"][k]); lo, hi = (np.array(z) for z in r["ci"][k])
            ax.bar(x + (j - 1.5) * 0.2, pt, 0.19, color=c, label=lab)
            ax.errorbar(x + (j - 1.5) * 0.2, pt, yerr=[pt - lo, hi - pt], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5)
        ax.set_xticks(x, ["rel", "tke", "mvpe"], fontsize=6.8); ax.set_ylim(40, 100)
        ax.set_title(f"{title}: {r['n_traj']} trajectories", fontsize=7); ax.tick_params(labelsize=6)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("score (raw forecast)", fontsize=6.8)
    m = R["final"]["memorisation_differential"]; pt = np.array(m["point"]); lo, hi = (np.array(z) for z in m["ci"])
    ax = axes[2]; x = np.arange(3)
    ax.bar(x, pt, 0.55, color=BLUE)
    ax.errorbar(x, pt, yerr=[pt - lo, hi - pt], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    for xi, v in zip(x, pt):
        ax.text(xi + 0.31, v, f"{v:+.2f}", ha="left", va="center", fontsize=6.2, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, ["rel", "tke", "mvpe"], fontsize=6.8); ax.tick_params(labelsize=6); ax.set_xlim(-0.5, 2.85)
    ax.set_title("FINAL→live change,\nclean minus official (95% CI)", fontsize=6.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.36, 1.0))
    fig.savefig(FIG / "fe02_arms.png"); plt.close(fig)
    print("arms figure written")


def loo():
    p = A2 / "loo.json"
    if not p.exists():
        return print("loo.json missing, skipped")
    R = json.loads(p.read_text())
    folds = list(R)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 0.36 * len(folds) + 1.0), sharey=True,
                               gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(len(folds))[::-1]
    for yi, f in zip(y, folds):
        r = R[f]; tr = [t["comp"] for t in r["transfer"]]
        a.barh(yi + 0.18, r["in_fold"]["comp"], 0.32, color=GREY, label="fitted on the held-out fold itself (reference)" if yi == y[0] else None)
        a.barh(yi - 0.18, np.mean(tr), 0.32, color=BLUE, label="fitted on other folds (3 draws, mean)" if yi == y[0] else None)
        a.scatter(tr, [yi - 0.18] * len(tr), s=10, color=INK, zorder=3)
        b.scatter([r["v8_cov"]], [yi], marker="s", s=16, color=GREY, zorder=3, label="v8 widths" if yi == y[0] else None)
        b.scatter([r["in_fold"]["cov"]], [yi + 0.12], marker="^", s=18, color=INK2, zorder=3, label="in-fold fit" if yi == y[0] else None)
        b.scatter([t["cov"] for t in r["transfer"]], [yi - 0.12] * len(tr), s=12, color=BLUE, zorder=3,
                  label="transferred fit" if yi == y[0] else None)
    a.axvline(0, color=INK2, lw=0.8)
    if any(f.startswith("Re") for f in folds) and any(f.startswith("AoA") for f in folds):
        cut = sum(f.startswith("AoA") for f in folds)
        for ax in (a, b):
            ax.axhline(len(folds) - cut - 0.5, color=INK2, lw=0.6, ls=":")
    a.set_yticks(y, [f"{f}  (n={R[f]['n_traj']})" for f in folds], fontsize=6.4)
    a.set_xlabel("Δ composite vs v8 on the held-out fold (proxy AT)", fontsize=6.6)
    a.grid(axis="y", visible=False)
    b.axvspan(0.70, 0.80, color=AQUA, alpha=0.15, lw=0)
    b.text(0.75, y[0] + 0.5, "optimal 0.70–0.80", ha="center", fontsize=5.8, color=INK2)
    b.set_xlabel("interval coverage on the held-out fold", fontsize=6.6)
    b.grid(axis="y", visible=False)
    for ax in (a, b):
        ax.tick_params(labelsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(*a.get_legend_handles_labels(), frameon=False, fontsize=5.8, ncol=1, loc="upper left", bbox_to_anchor=(0.12, 1.0))
    fig.legend(*b.get_legend_handles_labels(), frameon=False, fontsize=5.8, ncol=3, loc="upper left", bbox_to_anchor=(0.62, 1.0))
    fig.savefig(FIG / "fe03_loo.png"); plt.close(fig)
    print("loo figure written")


k9()
arms()
loo()
