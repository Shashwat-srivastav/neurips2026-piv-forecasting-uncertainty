"""Method x subscore outcome matrix (report v2, Fig. 11). Values from report/data/experiments.json
records (ids in the ROWS list). Colour is clipped at +-3 points; the printed number is exact."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_figures import INK, INK2

HERE = Path(__file__).resolve().parent
FIG, DATA = HERE.parent / "figures", HERE.parent / "data"   # report/figures, report/data
N = np.nan
# (label, evidence, d_rel, d_tke, d_mvpe, d_sps, register id)
ROWS = [
    ("add heteroscedastic intervals (+η=0.4 corrector)", "live", 0.021, 0.744, 0.034, 18.317, "SUB02"),
    ("corrector η 0.4 → 1.4 + rank band modifier", "live", -0.169, 0.955, -0.063, 0.152, "SUB-v4"),
    ("remove corrector (v4 → v8)", "live", 0.148, -1.699, 0.029, 0.789, "SUB-v8"),
    ("bands ×0.85, fp16 bounds, modifier off (v4 → v6)", "live", 0, 0, 0, -0.990, "SUB-v6"),
    ("v17: estimator blend + interval tables + tke boost", "live", -0.064, 1.342, 0.249, 3.949, "SUB-v17"),
    ("fine-tune FNO, stride 20, sensitivity weights", "local", -0.889, 2.66, -0.618, -2.01, "FT01"),
    ("fine-tune, 9.3× windows, rel-heavy loss", "local", -0.063, -0.934, -0.352, N, "FT02"),
    ("fine-tune + amplitude jitter augmentation", "local", -0.111, -0.464, -0.403, N, "FT02"),
    ("fine-tune from sim_real_fno, lr 2e-5 (Exp. 2)", "local", -0.480, 2.099, -0.396, N, "FT09"),
    ("freeze all but fc/BN, from sim_real_fno", "local", -0.032, -0.923, -0.028, N, "FT11"),
    ("learned corrector η=1.4", "local", -0.226, 2.711, -0.190, -0.318, "FT03"),
    ("FNO/CNO proportion blend (0.95/0.05, η 0.8)", "local", -0.019, 1.832, 0.038, 0.378, "BL02"),
    ("mean-preserving tke boost (v16)", "local", -0.053, 1.619, 0, 0.391, "TK04"),
    ("v17 recipe vs v8", "proxy", 0.07, 1.75, 0.06, 4.006, "BL04"),
    ("v19: + ridge forecaster partner vs v17", "proxy", 0.060, 0.149, 0.023, 0.509, "BL06"),
]


def main():
    vals = np.array([r[2:6] for r in ROWS], float)
    fig, ax = plt.subplots(figsize=(6.9, 5.0))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", ["#b8431a", "#ec835a", "#f0efec", "#6da7ec", "#184f95"])
    shown = np.ma.masked_invalid(np.clip(vals, -3, 3))
    cmap.set_bad("#e4e3df")
    ax.imshow(shown, cmap=cmap, vmin=-3, vmax=3, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            txt = "n/a" if np.isnan(v) else f"{v:+.2f}"
            dark = not np.isnan(v) and abs(v) > 1.8
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.8,
                    color="#ffffff" if dark else INK)
    ax.set_xticks(range(4), ["Δ rel", "Δ tke", "Δ mvpe", "Δ SPS"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(ROWS)), [f"{r[0]}  [{r[1]}]" for r in ROWS], fontsize=6.8)
    for b in (4.5, 12.5):
        ax.axhline(b, color=INK2, lw=1)
    ax.grid(False)
    ax.set_title("Δ subscore points vs the preceding build or frozen baseline (blue = better; colour clipped at ±3)",
                 fontsize=7.4, color=INK, pad=26)
    fig.savefig(FIG / "fig11_method_matrix.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
    print("matrix written")
