"""Fig. 13: per-trajectory gain of the v17 recipe over v8 (proxy AT, per-window bins), tables fitted
on the fingerprint-selected SELECT split vs a random AoA-stratified split. Parsed from
experiments/robustness/eval_out.txt."""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from make_figures import AQUA, BLUE, INK, INK2

HERE = Path(__file__).resolve().parent
FIG, DATA = HERE.parent / "figures", HERE.parent / "data"   # report/figures, report/data
txt = (HERE.parents[1] / "experiments" / "robustness" / "eval_out.txt").read_text()
blocks = {}
for name in ("select_pw", "rand_pw"):
    seg = txt.split(f"== {name}:")[1].split("==")[0]
    blocks[name] = {m.group(1): float(m.group(2)) for m in re.finditer(r"(\d+_\d+)\s+n=\s*\d+\s+comp ([+-][\d.]+)", seg)}
trajs = sorted(blocks["select_pw"], key=lambda t: blocks["select_pw"][t])
y = np.arange(len(trajs))
fig, ax = plt.subplots(figsize=(6.6, 2.9))
for yi, t in zip(y, trajs):
    a, b = blocks["select_pw"][t], blocks["rand_pw"][t]
    ax.plot([b, a], [yi, yi], color="#c9c8c3", lw=2, zorder=1)
ax.scatter([blocks["rand_pw"][t] for t in trajs], y, s=30, color=AQUA, zorder=2, edgecolor="#fcfcfb", label="tables fitted on random split")
ax.scatter([blocks["select_pw"][t] for t in trajs], y, s=30, color=BLUE, zorder=3, edgecolor="#fcfcfb", label="tables fitted on fingerprint-selected SELECT")
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(y, trajs, fontsize=6.8)
ax.set_xlabel("Δ composite vs v8 on each FINAL trajectory (proxy AT, per-window bins)")
ax.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
ax.set_xlim(-0.1, 1.8)
ax.grid(axis="y", visible=False)
fig.savefig(FIG / "fig13_robustness.png")
print("ok", len(trajs))
