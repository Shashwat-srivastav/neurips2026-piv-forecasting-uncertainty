"""Fig. 14: the clean sim->real FNO (Run 1, 26 Sep) with honestly fitted interval tables, composite vs v17 LIVE,
by flow regime (trajectories it never saw). Values from experiments/clean_fno_build/aoa_split.py, select_score.py, experiments/clean_fno_regimes/."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from make_figures import BLUE, ORANGE, AQUA, INK, INK2
HERE = Path(__file__).resolve().parent
FIG, DATA = HERE.parent / "figures", HERE.parent / "data"   # report/figures, report/data
rows = [("FINAL AoA 15\nRe 17.8–20.3k", 1.382, 1.500), ("FINAL AoA 10\nRe 21.6–25.4k", 0.871, None),
        ("FINAL AoA 20\nRe 19.1–21.6k", -0.287, None), ("FINAL all", 0.668, 0.796),
        ("SELECT AoA 15\nRe 21.6–25.4k\n(= live regime)", -0.760, -0.754)]
fig, ax = plt.subplots(figsize=(6.8, 2.6))
x = np.arange(len(rows))
ax.bar(x - 0.18, [r[1] for r in rows], 0.34, color=[ORANGE if i == 4 else BLUE for i in x], edgecolor="#fcfcfb", linewidth=2, label="absolute widths")
sf = [r[2] if r[2] is not None else np.nan for r in rows]
ax.bar(x + 0.18, sf, 0.34, color=AQUA, edgecolor="#fcfcfb", linewidth=2, label="√-scaled widths (exp1_sf)")
for xi, r in zip(x, rows):
    ax.text(xi - 0.18, r[1] + (0.06 if r[1] >= 0 else -0.16), f"{r[1]:+.2f}", ha="center", fontsize=6.8, color=INK)
    if r[2] is not None:
        ax.text(xi + 0.18, r[2] + (0.06 if r[2] >= 0 else -0.16), f"{r[2]:+.2f}", ha="center", fontsize=6.8, color=INK)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(x, [r[0] for r in rows], fontsize=6.6)
ax.set_ylabel("Δ composite vs v17 live")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="absolute widths"), Patch(color=ORANGE, label="absolute widths, live regime (SELECT)"),
                   Patch(color=AQUA, label="√-scaled widths (exp1_sf)")], frameon=False, fontsize=6.8, loc="upper right")
ax.set_ylim(-1.2, 1.9); ax.grid(axis="x", visible=False)
fig.savefig(FIG / "fig14_regime.png"); print("ok")
