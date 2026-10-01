"""Page-1 overview: the pipeline we built and the live results, in one figure (fig0_overview.png)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from make_figures import AQUA, BLUE, INK, INK2, ORANGE

HERE = Path(__file__).resolve().parent
FIG, DATA = HERE.parent / "figures", HERE.parent / "data"   # report/figures, report/data
LIGHT = {"blue": "#e6effa", "aqua": "#e3f5ee", "orange": "#fdeee4", "grey": "#f3f2ee"}

fig = plt.figure(figsize=(7.2, 3.35))
ax = fig.add_axes([0, 0.30, 1, 0.70]); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")


def box(x, y, w, h, title, lines, edge, face):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6", lw=1.2, ec=edge, fc=face))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=7.6, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 7.2, "\n".join(lines), ha="center", va="top", fontsize=6.1, color=INK2, linespacing=1.35)


def arrow(x0, x1, y=21):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=INK2))


box(0.5, 6, 16, 30, "Input window", ["20 PIV frames", "u, v on 32×64", "masked cells = 0"], INK2, LIGHT["grey"])
box(21, 21.5, 19, 14.5, "Official FNO", ["sim-pretrained,", "real-fine-tuned (v17)"], ORANGE, LIGHT["orange"])
box(21, 6, 19, 13.5, "Our clean FNO", ["sim-only → 67 real traj.", "never saw hold-outs"], AQUA, LIGHT["aqua"])
box(45, 17, 24, 19, "Estimator blend", ["weights on FNO / window mean /", "last frame per lead × channel", "× σ_local sextile; energy corr."], BLUE, LIGHT["blue"])
box(74, 6, 25.5, 30, "Interval tables", ["centre: same blend form", "half-width: lead group ×", "frame-difference decile",
                                          "fitted per model", "(per-window in exp1_sf)"], BLUE, LIGHT["blue"])
arrow(17.3, 20.5, 28.5); arrow(17.3, 20.5, 13)
arrow(40.6, 44.5, 28.5); arrow(69.6, 73.5, 28.5); arrow(40.6, 73.5, 11)
ax.text(57, 12.2, "raw forecast (exp1_sf)", ha="center", fontsize=6, color=INK2, style="italic")
ax.text(57, 38.2, "v17 only", ha="center", fontsize=6, color=INK2, style="italic")

# results strip
bx = fig.add_axes([0.07, 0.06, 0.88, 0.22])
rows = [("exp1_sf  (our clean FNO + interval tables, per-window)", 79.238, AQUA),
        ("v17  (official FNO + blend + tables, pooled bins)", 79.261, ORANGE),
        ("1st place  (benchmark)", 82.151, INK2)]
for i, (lab, v, c) in enumerate(rows):
    y = 2 - i
    bx.barh(y, v - 77, left=77, height=0.62, color=c, alpha=0.9)
    bx.text(v + 0.06, y, f"{v:.3f}", va="center", fontsize=6.8, color=INK, fontweight="bold")
    bx.text(77.05, y, lab, va="center", fontsize=6.1, color="white" if c != INK2 else "white")
bx.set_xlim(77, 82.9); bx.text(82.9, -1.05, "axis starts at 77", ha="right", fontsize=5.5, color=INK2); bx.set_yticks([]); bx.set_xlabel("live final score (development phase)", fontsize=6.5)
bx.tick_params(labelsize=6)
for s in ("left", "right", "top"):
    bx.spines[s].set_visible(False)
bx.grid(axis="y", visible=False)
fig.savefig(FIG / "fig0_overview.png")
print("overview written")
