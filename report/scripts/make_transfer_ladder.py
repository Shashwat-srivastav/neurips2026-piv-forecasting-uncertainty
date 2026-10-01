"""Fig. 12: sim-to-real transfer ladder on the local FINAL split (10 trajectories).
Values from report/data/experiments.json (EM06, FT08, FT10, FT09) and the live board."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from make_figures import AQUA, BLUE, INK, INK2, ORANGE

HERE = Path(__file__).resolve().parent
FIG, DATA = HERE.parent / "figures", HERE.parent / "data"   # report/figures, report/data
ROWS = [  # label, rel, tke, mvpe, style
    ("simulation-only FNO\n(never saw real data)", 79.709, 66.872, 75.199, "clean"),
    ("ours: freeze-all-but-fc/BN\nfrom sim-only (FT10)", 92.462, 71.939, 93.527, "clean"),
    ("ours: full fine-tune from\nsim-only, 67 traj. (Exp. 1)", 94.266, 78.782, 95.505, "clean"),
    ("official sim+real FNO\n(trained on these traj.)", 96.379, 81.039, 97.064, "memorised"),
    ("ours: fine-tune of official\n(Exp. 2, v18ft)", 95.899, 83.137, 96.668, "memorised"),
]
LIVE = (94.168, 74.026, 92.836)
fig, axes = plt.subplots(1, 3, figsize=(6.9, 3.0), sharey=True)
y = np.arange(len(ROWS))[::-1]
for j, (ax, name) in enumerate(zip(axes, ("rel", "tke", "mvpe"))):
    for yi, r in zip(y, ROWS):
        clean = r[4] == "clean"
        ax.barh(yi, r[1 + j], height=0.62, color=BLUE if clean else "#c9c8c3",
                hatch=None if clean else "////", edgecolor="#fcfcfb", linewidth=1.5)
        ax.text(r[1 + j] + 0.4, yi, f"{r[1 + j]:.1f}", va="center", fontsize=6.6, color=INK)
    ax.axvline(LIVE[j], color=ORANGE, lw=1.4, ls="--")
    ax.text(LIVE[j] - 0.6, -0.72, f"official, live {LIVE[j]:.1f}", fontsize=6.3, color=INK, ha="right")
    ax.set_ylim(-1.0, len(ROWS) - 0.5)
    ax.set_xlim(60, 102)
    ax.set_title(f"{name} subscore (local FINAL)", fontsize=7.6, color=INK)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(y, [r[0] for r in ROWS], fontsize=6.6)
fig.tight_layout()
fig.savefig(FIG / "fig12_transfer_ladder.png")
print("ok")
