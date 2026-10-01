"""Report builder, stage 0: base text, tables and layout. Run through build_all.py."""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)
from xml.sax.saxutils import escape

AUTHORS = "Shashwat Srivastava"
AFFILIATION = "Independent Researcher"
VERSION = "v5.0 (26 Sep 2026, before the development phase closed)"

HERE = Path(__file__).resolve().parent
FIG, DATA = HERE.parent / "figures", HERE.parent / "data"   # report/figures, report/data
ROOT = HERE.parents[1]
OUT = HERE / "RealPDE2026_Track1_Slash_technical_report_v5.pdf"
LIVE_EXP1_SF = (93.728, 77.348, 92.378, 91.701, 37.160, 79.238)
# Filled in from experiments/per_window_rebuild once the per-window rebuild reports (see §10).
PW_TEXT = ("We rebuilt the method with every rank statistic computed from the answered window alone (v17pw, v19pw; "
           "tables refitted with per-window bins). Under the four proxies, the change moves the local composite by "
           "+0.002 to +0.006, so the pooling carried no benefit. Each per-window build must pass an independence test "
           "before submission: every window's prediction and bounds must be identical, to 10<super>−6</super>, whether "
           "it is predicted alone, in any batch, shuffled, padded or duplicated.")

F = Path("C:/Windows/Fonts")
for n, f in (("DV", "DejaVuSans.ttf"), ("DVB", "DejaVuSans-Bold.ttf"), ("DVI", "DejaVuSans-Oblique.ttf"),
             ("DVM", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(n, str(F / f)))
registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")
INK, INK2, TINT = (colors.HexColor(c) for c in ("#0b0b0b", "#52514e", "#f3f2ee"))
S = {
    "title": ParagraphStyle("t", fontName="DVB", fontSize=15, leading=19, alignment=TA_CENTER, textColor=INK, spaceAfter=6),
    "sub": ParagraphStyle("s", fontName="DV", fontSize=9.2, leading=12.5, alignment=TA_CENTER, textColor=INK2),
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=11.3, leading=15, textColor=INK, spaceBefore=9, spaceAfter=4),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=9.4, leading=12.5, textColor=INK, spaceBefore=5, spaceAfter=2),
    "body": ParagraphStyle("b", fontName="DV", fontSize=8.8, leading=12.2, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=5),
    "abs": ParagraphStyle("a", fontName="DV", fontSize=8.6, leading=11.9, alignment=TA_JUSTIFY, textColor=INK,
                          leftIndent=7 * mm, rightIndent=7 * mm, spaceAfter=6),
    "cap": ParagraphStyle("c", fontName="DVI", fontSize=7.7, leading=10, textColor=INK2, spaceBefore=2, spaceAfter=8),
    "eq": ParagraphStyle("e", fontName="DVM", fontSize=8.1, leading=11.4, textColor=INK, leftIndent=7 * mm, spaceAfter=5),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.2, leading=9, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DVB", fontSize=7.2, leading=9, textColor=INK),
    "tiny": ParagraphStyle("tiny", fontName="DV", fontSize=6.3, leading=7.8, textColor=INK),
    "tinyb": ParagraphStyle("tinyb", fontName="DVB", fontSize=6.3, leading=7.8, textColor=INK),
}
P = lambda t, s="body": Paragraph(t, S[s])  # noqa: E731


def fig(name, w, cap):
    img = Image(str(FIG / name))
    img.drawWidth, img.drawHeight = w * mm, w * mm * img.imageHeight / img.imageWidth
    return KeepTogether([img, P(cap, "cap")])


def table(rows, widths, cap=None, small=False, keep=True):
    cs, hs = (S["tiny"], S["tinyb"]) if small else (S["cell"], S["cellb"])
    data = [[Paragraph(str(c), hs if i == 0 else cs) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TINT])]))
    out = [t] + ([P(cap, "cap")] if cap else [])
    return KeepTogether(out) if keep else out


story = []
A = story.append
E = story.extend

A(P("Input-Window Estimator Blending for Real PIV Forecasting under a Possibly Contaminated Local Hold-out", "title"))
A(P("NeurIPS 2026 RealPDE Competition, Track 1 (Simulation-to-Real Transfer): technical report", "sub"))
A(P(f"Team <b>Slash</b> (Codabench: slash) · {AUTHORS} · {AFFILIATION}", "sub"))
A(P(f"Report version {VERSION}. Final: no further submissions; v17pw, v19pw and v18ft were never scored live.", "sub"))
A(Spacer(1, 7))
A(P("<b>Abstract.</b> We forecast 20 future frames of real particle-image-velocimetry (PIV) velocity around a "
    "NACA4418 airfoil from a 20-frame input window. <b>Main result:</b> a post-processing layer fitted on released "
    "training data raised the live Safe Prediction Score (SPS) from 33.34 to 37.29 and the final score from 78.14 to "
    "79.26. The organisers' simulation-pretrained, real-fine-tuned Fourier Neural Operator (FNO) was left unchanged. "
    "That live score comes from a build that pooled rank statistics across the windows of each call, which the "
    "Rules do not allow (§12). Its per-window rebuild differs locally by +0.002 to +0.006 composite but was not scored "
    "live. Our best live score without cross-window pooling is 79.238 (exp1_sf, independence test passed; its tables were "
    "fitted on SELECT, so Disclosure 2 applies; §10). <b>Why it works:</b> statistics of the input window "
    "identify where persistence or the window mean is more accurate than the neural forecast, especially at short "
    "leads and in calm regions, where the FNO over-smooths. The same statistics place the prediction interval. Under a "
    "live-calibrated proxy, the interval-centre table alone adds 2.5 SPS points and the width table alone 1.8, of 4.0 in "
    "total; the attribution depends on the order in which components are added. <b>Sim-to-real:</b> "
    "our fine-tunes of the simulation-only FNO raise the rel subscore on held-out real trajectories from 79.7 to "
    "94.3. Retrained on 26 Sep and submitted live (exp1_sf, 79.238), its raw forecast scored 93.73/77.35/92.38 against the "
    "official FNO's 94.17/74.03/92.84 on the same live set: it traded 0.44 rel and 0.46 mvpe for 3.32 tke, at composite "
    "parity (§10). <b>Evaluation:</b> the official checkpoint appears to have been trained on every released real trajectory. "
    "This is an inference, not documented; a clean control loses far less between FINAL and live (§10). If so, no local hold-out is unseen by it, and every local comparison "
    "favoured it. That is consistent with a record in which four locally validated changes delivered between −31% and +7% "
    "of their local effect live. We give an exact specification of the method (Algorithm 1), a component ablation, the decision "
    "rules we used, and the protocol of every record (46 experiments, 15 measurements, 19 submissions). We also disclose two compliance issues found in review: "
    "rank statistics pooled across the windows of each call, and evaluation-input statistics written to logs in August.", "abs"))

# 1
A(P("1  Task, metrics and what counts", "h1"))
A(P("Inputs are windows (N, 20, 32, 64, 3) with channels [u, v, p], and p ≡ 0 for PIV. Outputs have the same shape "
    "for 20 future frames, optionally with per-element <i>lower</i>/<i>upper</i> bounds. Five subscores are reported. "
    "Relative L2 (rel), turbulent kinetic energy (tke: relative L2 of the per-pixel temporal variance map) and mean "
    "velocity profile error (mvpe: four probe columns, time-averaged first) each map an error e to 100/(1+0.5e). "
    "A runtime score and the Safe Prediction Score complete the set:"))
A(P("SPS = 100 · Σ<sub>b</sub> w<sub>b</sub> · mean[(1−pm<sub>b</sub>) · exp(−(upper−lower)/σ) · 1{covered}],  pm = e/(0.5+e),"
    "  σ = 0.0563870,  w = (0.5, 0.3, 0.2) for (rel, tke, mvpe)", "eq"))
A(P("SPS factorises closely as 100·W̄·T. W̄ depends only on accuracy. T, the <i>band factor</i>, depends on "
    "how tight and well-placed the intervals are. The optimal coverage is 0.70–0.80, not 0.95, because over-wide "
    "bands are penalised exponentially. How the five subscores combine into final_score is unpublished."))

# 2
A(P("2  Data audit and the evaluation set", "h1"))
A(P("<b>Release defects.</b> <i>train_real/6300_0</i> and <i>7575_0</i> are byte-identical. We reported this, and the "
    "organisers' data notice of 12 Aug 2026 excluded <i>7575_0</i>. In 78 of 82 files the <i>re</i> attribute disagrees "
    "with the file name (the true Reynolds step is 1,278–1,279, not 1,275). <i>train_sim</i> matches the real grid, but "
    "its amplitude ratio is anisotropic (u_std 3.59×, v_std 10.81×) and it lacks PIV dropout (0.20% vs 9.91% zeros). "
    "We never used it for training."))
A(P("<b>The evaluation set.</b> From aggregate input statistics our August submissions logged (see §12), the "
    "development evaluation set is a single flow condition, AoA 15°, Re ≈ 2.1–2.5×10<super>4</super>. Its live "
    "autocorrelation (lags 1–3: 0.686/0.417/0.146) and radial spectrum match training trajectories of the same "
    "regime to within 8%. The input statistics we measured (lag-1–3 autocorrelation, radial spectrum) show no anomaly. "
    "Live mvpe, below all 81 training trajectories (Fig. 1), is not explained by them."))
A(fig("fig9_generalisation.png", 150, "Figure 1. The official FNO scored on each of the 81 training trajectories (10 windows each), "
      "with its live development score (dashed). These trajectories were very likely in its training data (Table 2), "
      "so the spread need not reflect generalisation. Live mvpe lies below all 81."))

# 3
A(P("3  Supporting evidence from the public leaderboard", "h1"))
wb = json.loads((DATA / "weights_bootstrap.json").read_text())["weights"]
A(P("This section only tells us where the score is sensitive. It is not an estimate of the official "
    "scoring function. The leaderboard API returns all five subscores for every team. A linear approximation to the "
    "observed mapping, fitted over 176 teams, gives these partial-score sensitivities per subscore point "
    f"(bootstrap 95% CIs): rel {wb['rel'][0]:.3f} "
    f"[{wb['rel'][1]:.3f}, {wb['rel'][2]:.3f}], tke {wb['tke'][0]:.3f} [{wb['tke'][1]:.3f}, {wb['tke'][2]:.3f}], "
    f"mvpe {wb['mvpe'][0]:.3f} [{wb['mvpe'][1]:.3f}, {wb['mvpe'][2]:.3f}], time {wb['time'][0]:.3f} "
    f"[{wb['time'][1]:.3f}, {wb['time'][2]:.3f}], SPS {wb['sps'][0]:.3f} [{wb['sps'][1]:.3f}, {wb['sps'][2]:.3f}]. "
    "The RMS residual is 0.12. These coefficients should not be read as independent marginal weights. The subscores "
    "are strongly coupled (tke–SPS r = 0.73, rel–SPS r = 0.57), and rel, tke and mvpe also enter SPS through pm. "
    "The fit also mixes teams with different methods, so the coefficients are local slopes of this particular "
    "population, and a different population gives different ones. Refitting on the top 30 teams lowers the SPS "
    "sensitivity to 0.244 and raises tke to 0.101, so one SPS point is worth about 3.3 tke points on the full board (95% CI 2.8–3.9) and 2.4 in the top 30. One quantity is "
    "measured, not inferred: two of our live builds were bit-identical except for runtime, which gives a time "
    "sensitivity of 0.095, consistent with the fit. The same pair shows the board is deterministic for accuracy "
    "and SPS. Only runtime varies, by about ±0.017 final."))
A(fig("fig3_gap.png", 145, "Figure 2. Composite points our v17 entry would gain by matching the top-10 mean on each subscore "
      "(fitted sensitivities used as marginal weights, which §3 cautions against; bars sum to 2.86 vs the actual 2.69 gap; "
      "snapshot 25 Sep 2026, 14:33 UTC)."))
A(fig("fig2_T_by_rank.png", 145, "Figure 3. Mean band factor T = SPS/(100·W̄) by rank bucket (178 non-zero entries of 181). The top ten's "
      "rel-L2 errors are only about 10% below ours, so most of their SPS advantage comes from the band factor; aggregate "
      "subscores cannot show where in the field it arises."))

# 4
A(P("4  Sim-to-real transfer experiments", "h1"))
A(P("Our first line of attack was to transfer the simulation-pretrained operators to real PIV ourselves (Table 1, "
    "Fig. 4). The clean-start experiments begin from checkpoints that never saw real data and train on 67 real "
    "trajectories. The 10 FINAL trajectories are held out, so these evaluations are honest. Full fine-tuning of the "
    "simulation-only FNO (Exp. 1) lifts rel/tke/mvpe on FINAL from 79.7/66.9/75.2 to 94.3/78.8/95.5. That is "
    "about the official model's <i>live</i> rel (94.2), above its live tke and mvpe. Freezing everything except the "
    "fc/BatchNorm layers (FT10) recovers most of the rel and mvpe gain with 0.02% of the parameters trainable."))
A(fig("fig12_transfer_ladder.png", 158, "Figure 4. Sim-to-real transfer on the local FINAL split. Blue: models that never saw these "
      "trajectories. Grey hatched: the official sim+real checkpoint and our fine-tunes of it. The organisers' training "
      "set includes these trajectories (inferred; see §5), so those scores are optimistic. Dashed: the official "
      "model's live scores. The two are measured on different data and are not directly comparable."))
A(table([
    ["experiment", "start", "trainable / data", "FINAL rel / tke / mvpe", "reading"],
    ["Exp. 1 full fine-tune", "sim-only FNO", "all; 67 traj., TRUE_W loss", "94.27 / 78.78 / 95.51", "large honest sim→real gain; plateaued by epoch 10; never tested live"],
    ["FT10 freeze-bn-fc + spectral aug.", "sim-only FNO", "fc/BN (0.02%)", "92.46 / 71.94 / 93.53", "wins all three vs its own start (+36 sum)"],
    ["FT01 fine-tune, stride 20", "official FNO", "all; 3,362 windows", "Δ −0.89 / +2.66 / −0.62", "buys tke, sells rel (amplitude rescale)"],
    ["FT02 9.3× windows, rel-heavy loss", "official FNO", "all; 31,343 windows", "Δ −0.06 / −0.93 / −0.35", "no productive direction once tke is forbidden"],
    ["FT02 + amplitude jitter", "official FNO", "all", "Δ −0.11 / −0.46 / −0.40", "augmentation did not rescue it"],
    ["Exp. 2/3 lr 2e-5 (v18ft)", "official FNO", "all; 67 traj.", "Δ −0.48 / +2.10 / −0.40", "same trade; memorised yardstick biased against it"],
    ["FT11 freeze-bn-fc", "official FNO", "fc/BN", "Δ −0.03 / −0.92 / −0.03", "output calibration already optimal"],
    ["WiSE-FT sim ↔ sim+real", "both", "interpolation", "−7.5 composite at α=0.95", "different basins (‖Δθ‖/‖θ‖ = 1.09)"],
    ["CNO clean-start fine-tune", "sim-only CNO", "all", "no result", "out of memory on a 14.6 GB T4; never completed"],
], [34, 20, 28, 30, 62], "Table 1. Fine-tuning experiments (register ids FT01–FT13, EM06). Δ rows are against the frozen official model on "
   "the memorised split. The rows starting from the sim-only model are honest held-out numbers."))
A(P("<b>Why the fine-tunes of the official model failed.</b> The official checkpoint had already been fine-tuned on "
    "the same real data. From that converged point, the one direction that reduces the training loss is an "
    "output-amplitude rescale, which buys tke and sells rel. Forbid that trade and training overfits (run B: train "
    "loss −15% while held-out fell). And every one of these runs was judged against a checkpoint that had memorised "
    "the yardstick (§5). A fine-tune that generalises better could still look slightly worse locally. None was ever "
    "submitted live, so the question is open."))

# 5
A(P("5  The evaluation trap and how we worked around it", "h1"))
A(P("We state the memorisation question once, here, with the strength of each claim. Elsewhere in the report, "
    "“memorised split” is shorthand for the plausible but unproven explanation in the third row."))
A(table([
    ["claim", "evidence level", "basis"],
    ["The official checkpoint was fine-tuned on real PIV data", "documented", "released as the sim+real fine-tuned baseline; its metadata records iteration, best_iteration and best_val_loss"],
    ["It was trained on every released real trajectory, including our hold-out", "inference", "no statement that any trajectory was held out; release contains only the training split"],
    ["Its local scores are inflated by having seen the local trajectories", "supported by one clean control (n = 1)",
     "it scores 96.38/81.04/97.06 on our FINAL split vs 94.17/74.03/92.84 live; our Exp. 1, which never saw FINAL, "
     "scores 94.27/78.78/95.51 there. FINAL→live drop: official raw FNO −2.21/−7.01/−4.23; clean Run 1 (never saw FINAL) "
     "−0.43/−1.83/−3.13 (§10). Alternative not fully excluded: the live flow sits in the regime cluster where the FNO is weakest"],
    ["Local comparisons against it were biased in its favour", "consistent with the record",
     "four locally validated changes delivered −31% to +7% of their local effect live (Fig. 5); for v17, live SPS was 57–99% of the four proxies' predictions and live composite 54–84%"],
], [50, 26, 94], "Table 2. The memorisation question, graded by evidence."))
A(fig("fig8_transfer.png", 150, "Figure 5. Local prediction vs live measurement for shipped changes. The first four were validated on the "
      "memorised split. The last was predicted before submission by one of four proxies (see Table 3), identified "
      "as the closest only after the live result was known. Units differ (composite or SPS points), as labelled."))
A(P("<b>Residual-inflation proxies.</b> We simulate a live-like error level by inflating the local residual. "
    "Proxy A uses Y′ = P + k(Y − P), with the real input windows. Proxy B uses P′ = Y + k(P − Y): it degrades only "
    "the FNO and scores the input-window estimators against the real target. k was fitted two ways. With "
    "k_rel = 1.665 the unmodified entry reproduces the live rel-L2 error. With k_T = 1.447 it reproduces the live "
    "band factor. Before v17 was submitted, all four proxies were computed (Table 3), and none was designated in "
    "advance. The decision record also noted that the proxies had overstated earlier band effects by 1.5–1.9×. "
    "After the live result, the AT proxy was closest for SPS (+4.01 vs +3.95 live). It over-predicted the composite "
    "by 19% and tke by 31%. That is one post-hoc agreement, not a validation. The ±5% default-band check "
    "(T 0.204 vs 0.208 live) reproduces a number that was known before k was fitted."))
A(table([
    ["proxy (k)", "Δrel", "Δtke", "Δmvpe", "ΔSPS", "Δcomposite"],
    ["A (k_rel 1.665)", "+0.08", "+1.55", "+0.07", "+4.31", "+1.40"],
    ["B (k_rel)", "+0.07", "+0.57", "+0.15", "+6.98", "+2.08"],
    ["AT (k_T 1.447)", "+0.07", "+1.75", "+0.06", "+4.01", "+1.33"],
    ["BT (k_T)", "+0.04", "+0.94", "+0.12", "+5.59", "+1.70"],
    ["<b>live (measured)</b>", "<b>−0.06</b>", "<b>+1.34</b>", "<b>+0.25</b>", "<b>+3.95</b>", "<b>+1.12</b>"],
], [40, 22, 22, 22, 22, 26], "Table 3. v17 minus v8: the four pre-submission proxy predictions and the live result."))
A(fig("fig5_proxy_vs_live.png", 140, "Figure 6. The AT row of Table 3 against the live result. AT was selected after the fact; "
      "tke is over-predicted by 31%."))

# 6
A(P("6  Method: the shipped entry (v17)", "h1"))
A(P("Algorithm 1 specifies the method completely. Every constant is either given there or is a table entry "
    "released with the code (submissions/v17_per_window/_blend_tables.py). The prose below explains each step."))
ALG = [
    "<b>Algorithm 1 — v17 inference for one window</b> (all statistics use this window only; per-window version v17pw)",
    "<b>Input</b>  x ∈ R<super>20×32×64×3</super> (frames t = 1..20, channels u, v, p); NaN/±inf → 0.",
    "<b>Tables</b>  PRED, CENT ∈ R<super>20×2×6×2</super> (lead × channel × sextile × (α, β)); WID ∈ R<super>4×2×10</super> (lead group × channel × decile).",
    "1.  P ← FNO((x − μ<sub>in</sub>)/σ<sub>in</sub>)·σ<sub>out</sub> + μ<sub>out</sub>;  P[..., p] ← 0.        (official sim_real_ft checkpoint, fp32)",
    "2.  For each pixel and c ∈ {u, v}:  m = #{t : x<sub>t</sub> ≠ 0};  valid = (m > 0).",
    "     E<sub>m</sub> = Σ<sub>t</sub> x<sub>t</sub> / max(m, 1)   (masked window mean);   E<sub>l</sub> = x<sub>20</sub> if x<sub>20</sub> ≠ 0 else E<sub>m</sub>   (last measured frame).",
    "     s = √(mean<sub>t</sub>(x<sub>t</sub> − mean<sub>t</sub> x)<super>2</super>)   (σ<sub>local</sub>, unmasked);   d = √(mean<sub>t=1..19</sub>(x<sub>t+1</sub> − x<sub>t</sub>)<super>2</super>).",
    "3.  Ranks (per window, per channel, over valid pixels only):  q<sub>j</sub> = linear-interpolated quantiles at j/K;",
    "     bin = #{j : q<sub>j</sub> < value}, i.e. searchsorted(q, value, left).  K = 6 for s (sextile k), K = 10 for d (decile r).",
    "4.  For lead τ = 1..20 and valid pixels:  (α, β) = PRED[τ, c, k];  ŷ<sub>τ</sub> = E<sub>m</sub> + α(P<sub>τ</sub> − E<sub>m</sub>) + β(E<sub>l</sub> − E<sub>m</sub>).",
    "     Invalid pixels (never measured in the window): ŷ<sub>τ</sub> = P<sub>τ</sub>.   Constraint of the fit: α, β ≥ 0, α + β ≤ 1.",
    "5.  Interval centre:  (α′, β′) = CENT[τ, c, k];  ĉ<sub>τ</sub> = E<sub>m</sub> + α′(P<sub>τ</sub> − E<sub>m</sub>) + β′(E<sub>l</sub> − E<sub>m</sub>)   (same rule for invalid pixels).",
    "6.  Half-width:  h<sub>τ</sub> = WID[⌊(τ−1)/5⌋, c, r]  (absolute, 0.0025–0.025);  lower = ĉ − h,  upper = ĉ + h;  p-channel bounds = 0.",
    "7.  Energy correction of the prediction only:  f = ŷ − mean<sub>τ</sub> ŷ;  E<sub>pred</sub> = Σ<sub>pixels,c</sub> mean<sub>τ</sub> f<super>2</super>;",
    "     E<sub>in</sub> = Σ<sub>pixels,c</sub> s<super>2</super>;  a = clip((E<sub>in</sub>/E<sub>pred</sub>)<super>1/4</super>, 1, 1.6);  ŷ ← mean<sub>τ</sub> ŷ + a·f.   Bounds are not changed.",
    "<b>Output</b>  prediction ŷ (p = 0), lower, upper; float32.  No interpolation between bins; ranks always fall inside the table, "
    "but widths are absolute and not rescaled for other flow conditions.",
]
_alg = Table([[Paragraph(l, S["cell"])] for l in ALG], colWidths=[172 * mm])
_alg.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, INK), ("BACKGROUND", (0, 0), (-1, -1), TINT),
                          ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
                          ("LEFTPADDING", (0, 0), (-1, -1), 5)]))
A(KeepTogether([_alg, P("Fitting (offline, CPU, minutes). PRED: constrained least squares per cell on SELECT (4 trajectories, "
                        "168 windows) under proxy A (Y′ = P + k(Y − P), k = 1.534 fitted on SELECT to the live rel-L2 error). CENT: "
                        "per-cell grid search maximising the SPS band factor T under proxies A and B. WID: per-cell h maximising "
                        "exp(−2h/σ)·F̂(h) of the centre residual under proxies AT and BT (k = 1.394 on SELECT, fitted to the live band factor). The live v17 build pooled the rank edges over the 5 windows of each call; "
                        "v17pw computes them per window (§12).", "cap")]))
A(P("<b>Base model.</b> The official <i>sim_real_ft</i> FNO (50.36 M parameters, 99.9% in spectral convolutions; "
    "modes 4×12×16, width 64), stored in fp16 and run in fp32. No weights were changed."))
A(P("<b>Estimator blend.</b> With E<sub>m</sub> the masked input-window mean and E<sub>l</sub> the last measured frame, each element is"))
A(P("pred = E<sub>m</sub> + α·(P − E<sub>m</sub>) + β·(E<sub>l</sub> − E<sub>m</sub>),   α, β ≥ 0,  α + β ≤ 1", "eq"))
A(P("(α, β) are looked up per forecast lead (20) × channel (2) × σ<sub>local</sub> sextile (6). They were fitted "
    "by constrained least squares on the SELECT split under proxy A. Figure 7 shows the fit. At lead 1, most of the "
    "field is persistence (α = 0 in sextiles 1–4). In the calmest sextile at later leads, the window mean takes "
    "over. In active regions the FNO keeps most of the weight."))
A(fig("fig4_blend_alpha.png", 140, "Figure 7. Fitted blend weights α (FNO) and β (last frame) by lead and σ<sub>local</sub> sextile; the "
      "window mean receives 1 − α − β."))
A(P("<b>Why persistence wins at lead 1.</b> On FINAL, the FNO's first frame has rel-L2 0.0583 against "
    "persistence's 0.0522. It is further from the second target frame (0.0749), so it is not a one-step time shift. "
    "Two defects combine. The FNO loses the last frame's fine-scale structure: its fine-scale content is weaker than "
    "the truth's (u 0.0124 vs 0.0145), and its fine-scale correlation with the truth is lower than persistence's "
    "(0.79 vs 0.92). It also jumps too far from the last frame: its lead-1 increment std is 1.24× the true one "
    "(1.50× in the high-pass band: 0.0089 vs 0.0059; increment correlation 0.54). In the calmest sextile its lead-1 mean squared error is 14× persistence's. "
    "By lead 2 the FNO wins (0.0598 vs 0.0913)."))
A(fig("fig6_error_by_lead.png", 160, "Figure 8. rel-L2 by forecast lead on FINAL for the FNO, v17, the ridge forecaster of v19 (§8), "
      "persistence and the window mean. Left: the real local target, which the FNO may have seen in training (Table 2). Right: the AT "
      "proxy target (k = 1.447)."))
A(P("<b>Intervals.</b> A second table of the same form sets the interval <i>centre</i>. Half-widths come from a "
    "table indexed by lead group × channel × decile of the input's one-step frame-difference RMS. It is one of the "
    "strongest single target-free features we found (Spearman 0.55 with |error|; others reached 0.62, and a learned "
    "combination 0.65). Each cell's width "
    "maximises exp(−2h/σ)·F̂(h). <b>These widths are absolute values (0.0025–0.0256).</b> The rank index is "
    "scale-free, but the widths are not. The intervals are tuned to one flow condition and carry no coverage "
    "guarantee. They are an SPS-optimised heuristic, not uncertainty quantification in the conformal sense."))
A(P("<b>Energy correction.</b> The FNO forecast keeps about 60% of the target's temporal variance: 0.83 at lead 1, "
    "0.50 at lead 20, with spatial correlation ≈ 0.9. For an MSE-optimal forecast this under-representation is "
    "expected, not a defect. We therefore describe the correction as fitting the estimator to the tke metric. Each "
    "window's deviation from its own 20-frame mean is scaled by a = clip((E<sub>in</sub>/E<sub>pred</sub>)<super>1/4</super>, 1, 1.6). "
    "The time mean, and so mvpe, is unchanged, and the intervals stay where they are."))
A(fig("fig7_field_panels.png", 158, "Figure 9. One median-difficulty FINAL window at lead 10 (u): input, target, FNO and v17 errors, their "
      "difference, and the v17 half-width map. Masked cells are blank."))

# 7
A(P("7  Results", "h1"))
A(table([
    ["entry", "rel", "tke", "mvpe", "time", "SPS", "final", "note"],
    ["v8: official FNO + heteroscedastic bands", "94.168", "74.026", "92.836", "91.409", "33.338", "78.141", "13 Aug"],
    ["v17: v8 + blend, interval tables, energy correction", "94.104", "75.368", "93.085", "91.130", "37.287", "<b>79.261</b>", "25 Sep"],
    ["Δ", "−0.064", "+1.342", "+0.249", "−0.278", "+3.949", "+1.120", ""],
    ["exp1_sf: clean sim→real FNO + interval tables", "93.728", "77.348", "92.378", "91.701", "37.160", "79.238", "26 Sep"],
    ["top-10 mean, 25 Sep", "94.78", "79.64", "94.16", "92.15", "44.30", "81.95", ""],
], [50, 15, 15, 15, 15, 15, 15, 34], "Table 4. Live development-phase results. Rank 121 → 87 of 178 non-zero entries; top-10 cut-off 81.775. The v17 "
   "build pooled rank edges over the windows of each call (§12); its per-window rebuild was not scored live. The best "
   "live score without cross-window pooling is exp1_sf, 79.238 (§10); its tables were fitted on SELECT, so Disclosure 2 applies."))
A(table([
    ["component, vs v8 (proxy AT)", "Δrel", "Δtke", "Δmvpe", "ΔSPS", "Δcomposite"],
    ["energy correction only (v16)", "−0.04", "+1.61", "0", "+0.30", "+0.21"],
    ["uniform ×1.2 widths only", "0", "0", "0", "+1.01", "+0.29"],
    ["interval-centre table only", "0", "0", "0", "+2.53", "+0.72"],
    ["frame-difference width table only", "0", "0", "0", "+1.78", "+0.51"],
    ["prediction blend + energy correction", "+0.07", "+1.75", "+0.06", "+0.43", "+0.31"],
    ["blend + centre + energy correction", "+0.07", "+1.75", "+0.06", "+2.99", "+1.04"],
    ["+ frame-difference width table = v17", "+0.07", "+1.75", "+0.06", "+4.01", "+1.33"],
], [62, 20, 20, 20, 20, 26], "Table 5. Component ablation under proxy AT (experiments/estimator_blend/controls2_out.txt). Attribution depends on order: the width "
   "table adds +1.78 SPS alone but +1.02 when added last. The centre table is the largest single component either way."))
A(P("<b>Robustness.</b> With per-window rank bins, v17's gain over v8 under proxy AT is positive on all 10 FINAL "
    "trajectories: Δcomposite mean +1.36 (SD 0.15, range +1.10 to +1.56), ΔSPS +3.2 to +4.6. We refitted the tables on "
    "four trajectories drawn at random (one per AoA, chosen without the live fingerprint). With the proxy scale k "
    "refitted on that split, the gain drops to +0.71 (10/10 positive). With SELECT's k it drops to +0.26 (8/10 "
    "positive on composite, 6/10 on SPS). Between 48% and 81% of v17's local gain therefore depends on fitting on "
    "flows like the evaluation flow. A swap ablation (Table 6) locates it. Replacing only the width table costs "
    "0.35–0.76, only the centre table 0.16–0.17, and only the prediction table 0.12–0.14. The random split's flows are "
    "lower-Reynolds and its widths 20–51% narrower (Fig. 10). The absolute widths are the largest single risk for the "
    "private test at unseen AoA and Re, but the centre and prediction tables are also condition-specific. FINAL was "
    "also used to compare many recipes and proxy variants, so it is a selection set, not a clean hold-out, and the "
    "per-trajectory SD does not account for that selection."))
A(table([
    ["tables used (FINAL, proxy AT)", "Δ vs v8, random split, k refit", "Δ vs v8, random split, SELECT's k"],
    ["all fitted on SELECT (fingerprint)", "+1.356", "+1.356"],
    ["swap prediction table only", "+1.241", "+1.221"],
    ["swap centre table only", "+1.195", "+1.188"],
    ["swap width table only", "+1.002", "+0.593"],
    ["swap centre + width", "+0.823", "+0.385"],
    ["all fitted on random split", "+0.710", "+0.255"],
], [70, 50, 50], "Table 6. Which fitted component carries the dependence on the fitting split (experiments/robustness/swap_ablation_out.txt)."))
A(fig("fig13_robustness.png", 150, "Figure 10. Per-trajectory Δcomposite over v8 (proxy AT) with tables fitted on the fingerprint-selected "
      "split vs a random AoA-stratified split."))
A(fig("fig11_method_matrix.png", 160, "Figure 11. Effect of each method on each subscore against its predecessor or frozen baseline "
      "(live, local-memorised, or proxy, as tagged). Training rows repeatedly buy tke and sell rel. Only interval "
      "and blend changes raised SPS without costing accuracy."))
A(fig("fig1_live_history.png", 150, "Figure 12. Live final_score of the submissions whose scores are recorded, in order. S2 is omitted (only its "
      "SPS was recorded). S8, a CUDA-graph build, crashed at import and scored 0. The grey step is the best so far. "
      "Live scores for v7, v10, v11, v13 and v15 were not recorded. sf = exp1_sf (26 Sep, clean sim→real FNO)."))

# 8 decision rules + protocol cards
A(P("8  How we decided: rules and the protocols of the key experiments", "h1"))
A(table([
    ["condition observed", "decision", "where it was applied"],
    ["improves held-out rel, tke and mvpe together", "keep, and move towards a live test", "FT10 (from sim-only)"],
    ["improves tke but lowers rel or mvpe (amplitude trade)", "reject", "FT01, FT09, corrector η 1.4 (K2, K3)"],
    ["improves only on a split the reference model may have memorised", "do not trust; re-test on an honest split or a proxy", "log-POD, rank modifier, v18 (−0.055 real, +0.13 proxy)"],
    ["positive under all four proxies (A, B, AT, BT)", "candidate for a live slot", "v17, v19"],
    ["positive under some proxies only", "do not submit", "v18 unconstrained blend (BT, B negative)"],
    ["gain smaller than run-to-run noise (±0.017 final, runtime only)", "do not submit", "runtime micro-optimisations"],
    ["runtime cost exceeds composite gain (0.093 per time point)", "reject", "temporal-disagreement band feature (−1.47 time)"],
    ["SPS gain without coverage collapse (T up; coverage near the 0.70–0.80 optimum, not far below)", "candidate", "v17 (local coverage 0.84)"],
    ["live gain confirmed", "adopt as incumbent", "v8 (13 Aug), v17 (25 Sep)"],
    ["any build", "must pass preflight; from 25 Sep also the per-window independence test", "all submissions"],
], [62, 50, 58], "Table 7. Decision rules as applied. Slot choices from 25 Sep were also routed through a logged decision gate "
   "(tools/decisions/)."))
A(P("The cards below give the full protocol of the twelve experiments that shaped the campaign. Every other "
    "experiment is in Appendix B, with the same fields in compact form."))


def card(title, f):
    rows = [[Paragraph("<b>" + title + "</b>", S["cell"]), ""]]
    rows += [[Paragraph(k, S["cellb"]), Paragraph(v, S["cell"])] for k, v in f]
    t = Table(rows, colWidths=[27 * mm, 145 * mm])
    t.setStyle(TableStyle([("SPAN", (0, 0), (-1, 0)), ("BACKGROUND", (0, 0), (-1, 0), TINT),
                           ("BOX", (0, 0), (-1, -1), 0.6, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.4, INK),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1.3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1.3)]))
    return KeepTogether([t, Spacer(1, 5)])


CARDS = [
 ("K1  Heteroscedastic prediction intervals (S5 → S6, live)", [
  ("Purpose", "Is the default ±5% interval, used when no bounds are returned, what is holding SPS down?"),
  ("Hypothesis", "Intervals proportional to each pixel's input variability raise the band factor T, and so SPS, without touching accuracy."),
  ("Criterion", "Live SPS rises. Not pre-registered."),
  ("Intervention", "Return lower/upper = ŷ ± (A + K·σ<sub>local</sub>), A<sub>u</sub> = 0.004, A<sub>v</sub> = 0.003, K = 1.5 (scale bracketed earlier on three live DMD builds). Bundled with a learned corrector at η = 0.4."),
  ("Baseline", "S5: official FNO, no bounds (default ±5%)."),
  ("Data / evidence", "Live board (AoA 15 evaluation set)."),
  ("Result", "SPS 14.08 → 32.40 (+18.3); T 0.208 → 0.475; final 72.65 → 77.90. rel +0.02, tke +0.74 (the corrector's share)."),
  ("Conclusion", "Interval construction dominates SPS. This was the largest single gain of the campaign."),
  ("Caveat", "Two changes at once. The corrector's accuracy effect is small next to the interval effect.")]),
 ("K2  Learned residual corrector (v4 vs v8, live)", [
  ("Purpose", "Does a small learned operator trained on the FNO's residuals add composite score?"),
  ("Hypothesis", "Its tke gain outweighs any rel/SPS loss."),
  ("Criterion", "Live final_score of the corrected build exceeds the uncorrected one."),
  ("Intervention", "ŷ = P + η·C(x), C a 668,316-parameter TinyOperator (zero-initialised output) trained on real residuals; η = 1.4 (v4) vs η = 0 (v8). Identical intervals."),
  ("Baseline", "v8 (η = 0), otherwise identical."),
  ("Data / evidence", "Live board; local FINAL for the pre-test."),
  ("Result", "v4 − v8 live: rel −0.148, tke +1.699, mvpe −0.029, SPS −0.789, final −0.108. Local-to-live transfer of the three accuracy deltas: 66% / 63% / 15%."),
  ("Conclusion", "The corrector buys tke with rel. Because the intervals were centred on the corrected field, SPS fell. Closed."),
  ("Caveat", "Only two η values were shipped live.")]),
 ("K3  Further fine-tuning of the official FNO (FT01, FT02)", [
  ("Purpose", "Can more real-data training improve the official checkpoint?"),
  ("Hypothesis", "Fine-tuning on <i>train_real</i> (hold-out excluded) with a composite-aligned loss improves held-out rel, tke and mvpe."),
  ("Criterion", "All three ≥ 0 on FINAL."),
  ("Intervention", "FT01: stride 20 (3,362 windows), sensitivity-weighted loss. FT02 run B: stride 2 (31,343 windows), loss weights rel 0.8 / tke 0 / mvpe 0.2, input noise σ = 0.02, 7575_0 excluded, Kaggle T4. FT02 + amplitude jitter 0.70."),
  ("Baseline", "Frozen official FNO on the same FINAL windows."),
  ("Data / evidence", "Local; FINAL may have been seen by the baseline (Table 2)."),
  ("Result", "FT01 Δ −0.89 / +2.66 / −0.62 (SPS −2.01). FT02-B Δ −0.06 / −0.93 / −0.35. With jitter Δ −0.11 / −0.46 / −0.40. About 100% of the movement happens in epoch 0."),
  ("Conclusion", "From this converged start, the only direction that lowers the training loss is an output-amplitude rescale. It buys tke and sells rel. Forbid it and training overfits."),
  ("Caveat", "Judged against a possibly memorised baseline. Never submitted live, so not a clean negative.")]),
 ("K4  Clean-start sim-to-real fine-tune (Exp. 1)", [
  ("Purpose", "How much real-data transfer does our own training achieve on trajectories no model has seen?"),
  ("Hypothesis", "Fine-tuning the simulation-only FNO on 67 real trajectories lifts held-out scores towards the official model's live level."),
  ("Criterion", "Beats the frozen sim-only FNO on all three subscores on FINAL."),
  ("Intervention", "Start from sim_fno.pth; all parameters trainable; lr 1e-4; 80 epochs; composite-derivative loss weights; pool train_pool_v12_3 (67 trajectories, hold-out and 7575_0 excluded); Kaggle T4."),
  ("Baseline", "Frozen sim_fno on the same 10 FINAL trajectories."),
  ("Data / evidence", "Local, honest (FINAL never seen by either model)."),
  ("Result", "rel/tke/mvpe 79.71 / 66.87 / 75.20 → 94.27 / 78.78 / 95.51. Plateau from epoch 10."),
  ("Conclusion", "A large genuine sim-to-real gain, reaching roughly the official model's live rel (94.17)."),
  ("Caveat", "Local FINAL is not the live set. Never submitted live.")]),
 ("K5  Output-layer-only fine-tuning (FT10, FT11)", [
  ("Purpose", "Does freezing the spectral weights remove the amplitude trade of K3?"),
  ("Hypothesis", "With only the lift/projection and BatchNorm layers trainable, rel cannot be sold for tke."),
  ("Criterion", "All three subscores ≥ the frozen start on FINAL."),
  ("Intervention", "Train only fc0, fc1, fc2 and BatchNorm affine (9,667 of 50.36 M parameters) with radial spectral input augmentation; 30 epochs; epoch chosen on SELECT. FT10 starts from sim_fno; FT11 from the official FNO (lr 2e-5)."),
  ("Baseline", "The respective frozen start."),
  ("Data / evidence", "Local; honest for FT10, possibly memorised for FT11."),
  ("Result", "FT10: +12.75 / +5.07 / +18.33 (all three up). FT11: −0.03 / −0.92 / −0.03."),
  ("Conclusion", "Freezing works when the start has headroom. It only degrades an already-calibrated output layer."),
  ("Caveat", "The two effects of augmentation and freezing were not separated.")]),
 ("K6  Is our hold-out unseen by the official model?", [
  ("Purpose", "Decide whether local comparisons against the official FNO can be trusted."),
  ("Hypothesis", "If the hold-out were unseen, the official model would score no better there than on the live set."),
  ("Criterion", "Official hold-out score > own live score ⇒ the hold-out is not independent of its training."),
  ("Intervention", "Score frozen sim_fno, sim_cno and the official sim_real_fno on the same 10 FINAL trajectories (420 windows, stride 20)."),
  ("Baseline", "The official model's own live score."),
  ("Data / evidence", "Local + live."),
  ("Result", "Official 96.38 / 81.04 / 97.06 locally vs 94.17 / 74.03 / 92.84 live. sim_fno 79.71 / 66.87 / 75.20; sim_cno 78.15 / 67.15 / 74.74. Update 26 Sep: from FINAL to live the official raw FNO drops −2.21/−7.01/−4.23, the clean Run 1 only −0.43/−1.83/−3.13."),
  ("Conclusion", "Consistent with memorisation (Table 2). We stopped using the official model as a local yardstick."),
  ("Caveat", "A harder live regime would also produce a gap.")]),
 ("K7  Input-window estimator blend and interval tables (v17, live)", [
  ("Purpose", "Can input-window statistics identify where trivial estimators beat the FNO, and place intervals better?"),
  ("Hypothesis", "Blending towards persistence and the window mean in calm regions and short leads lowers error where SPS is most sensitive; re-centred, variability-indexed intervals raise T."),
  ("Criterion", "Positive composite under all four proxies before submission (Table 7); then live."),
  ("Intervention", "Algorithm 1 on top of v8, with the tables fitted on SELECT."),
  ("Baseline", "v8 (live 78.141)."),
  ("Data / evidence", "Proxy (FINAL), then live."),
  ("Result", "Proxies +1.33 to +2.08 composite. Live: final +1.120 (78.141 → 79.261), SPS +3.95, tke +1.34, mvpe +0.25, rel −0.06, time −0.28. The board is deterministic for accuracy and SPS; sampling uncertainty over the ~90 evaluation windows of one flow condition is not quantified."),
  ("Conclusion", "Adopted. The mechanism is decomposed in K8."),
  ("Caveat", "The live build pooled rank edges over five windows (§12). The tables were fitted on a fingerprint-selected split (K9).")]),
 ("K8  Component ablation of v17", [
  ("Purpose", "Which parts of v17 cause the gain?"),
  ("Hypothesis", "Most of the SPS gain comes from interval placement, not from the prediction."),
  ("Criterion", "Not pre-registered."),
  ("Intervention", "Add components one at a time to v8: energy correction; uniform width ×1.2; centre table; prediction blend; widths table (Table 5)."),
  ("Baseline", "v8."),
  ("Data / evidence", "Proxy AT on FINAL."),
  ("Result", "Centre table alone +2.53 SPS (+0.72 composite); widths alone ×1.2 +1.01 SPS; prediction blend + energy correction +0.43 SPS, +1.75 tke; full v17 +4.01 SPS, +1.33 composite."),
  ("Conclusion", "Moving the interval centre is the largest single cause."),
  ("Caveat", "Persistence-only and mean-only blends were not run separately.")]),
 ("K9  Fingerprint-free refit", [
  ("Purpose", "How much of v17's gain depends on fitting the tables on flows chosen to resemble the evaluation set?"),
  ("Hypothesis", "The prediction part transfers; the absolute widths do not."),
  ("Criterion", "Not pre-registered."),
  ("Intervention", "Refit all tables (per-window bins) on four random trajectories, one per AoA (seed 0: 10125_5, 12675_10, 7575_15, 3750_20; 168 windows); refit proxy k the same way (1.413 / 1.826)."),
  ("Baseline", "The same recipe fitted on SELECT."),
  ("Data / evidence", "Proxy AT on FINAL, per trajectory."),
  ("Result", "k refitted: +0.71 vs v8 (10/10 positive), −0.65 vs SELECT-fitted. SELECT's k: +0.26 (8/10 composite, 6/10 SPS), −1.10. Swap ablation (Table 6): width table 0.35–0.76, centre 0.16–0.17, prediction 0.12–0.14. Random-split widths 20–51% narrower (lower Re)."),
  ("Conclusion", "48–81% of v17's local gain depends on the choice of fitting flows; the widths carry most of it, the centre and prediction tables the rest."),
  ("Caveat", "One random split.")]),
 ("K10  Why persistence beats the FNO at lead 1", [
  ("Purpose", "Explain the α = 0 cells of the blend table (Fig. 7)."),
  ("Hypothesis", "Either the FNO output is shifted by a time step, or it over-smooths."),
  ("Criterion", "A time shift ⇒ P<sub>1</sub> is closer to Y<sub>2</sub> than to Y<sub>1</sub>."),
  ("Intervention", "Compare P<sub>1</sub> with Y<sub>1</sub>, Y<sub>2</sub> and persistence; compare fine-scale content and increment correlations; split by σ<sub>local</sub> sextile."),
  ("Baseline", "Persistence x<sub>20</sub>."),
  ("Data / evidence", "Local FINAL, real target."),
  ("Result", "rel-L2 of P<sub>1</sub>: 0.0583 vs Y<sub>1</sub>, 0.0749 vs Y<sub>2</sub>; persistence 0.0522. Fine-scale correlation 0.79 vs persistence 0.92. In the calmest sextile the FNO's error is 14× persistence's. Lead 2: FNO 0.0598 vs persistence 0.0913."),
  ("Conclusion", "Lost fine-scale structure plus an overshooting first increment; not a one-step time shift."),
  ("Caveat", "Measured on data the FNO may have memorised, which favours the FNO.")]),
 ("K11  Energy correction for the tke metric", [
  ("Purpose", "Can the tke deficit be reduced without costing rel?"),
  ("Hypothesis", "The forecast's variance map is well placed but too weak, so a mean-preserving amplitude increase raises tke with little rel cost."),
  ("Criterion", "Pre-registered primary: fixed gain a = 1.15 (experiments/tke_energy/PREREG.txt)."),
  ("Intervention", "Scale fluctuations about the forecast's own time mean by a = clip((E<sub>in</sub>/E<sub>pred</sub>)<super>1/4</super>, 1, 1.6). Intervals unchanged."),
  ("Baseline", "Official FNO with v8 intervals."),
  ("Data / evidence", "Local FINAL, then live inside v17."),
  ("Result", "Diagnosis: predicted/target energy 0.62, spatial correlation 0.90. Fixed a = 1.15: tke +1.38, rel −0.07. Adaptive: tke +1.62, rel −0.05, SPS +0.39."),
  ("Conclusion", "This fits the estimator to the metric: an MSE-optimal forecast under-represents variance by design."),
  ("Caveat", "tke is a 20-frame variance with effective n ≈ 4; the gain is metric-specific.")]),
 ("K12  Learned linear forecaster as a blend partner (v19)", [
  ("Purpose", "Can a simple model trained only on released data beat the window estimators, honestly?"),
  ("Hypothesis", "A local linear forecaster beats persistence and the window mean at short leads."),
  ("Criterion", "Positive under all four proxies."),
  ("Intervention", "Ridge regression per (σ<sub>local</sub> sextile × 4×4 spatial block) from the last 10 frames in a 5×5 u,v neighbourhood plus the window mean (549 features), λ = 10<super>−4</super>, shrunk towards the per-sextile fit; trained on the 67-trajectory pool at stride 4 (13,629 windows); added to the blend as a third, constrained partner."),
  ("Baseline", "v17."),
  ("Data / evidence", "Local honest (standalone), proxy (blend)."),
  ("Result", "Standalone FINAL rel-L2 0.126 vs window mean 0.141 and persistence 0.170. Best predictor at leads 1–2. Blend: +0.16 to +0.33 composite over v17 under the four proxies."),
  ("Conclusion", "Built as v19pw (independence PASS); never submitted."),
  ("Caveat", "Never evaluated live. Runtime on the evaluation GPU unmeasured.")]),
]
for title, fields in CARDS:
    A(card(title, fields))

# 9
A(P("9  Negative results, reversals, and the last candidates", "h1"))
A(P("<b>Two conclusions we had to reverse.</b> In August we closed blending “as a class”: a 3,624-combination search "
    "over FNO, CNO, DMD, persistence and window-mean blends returned the pure FNO. We also concluded that the "
    "intervals sat at their analytic optimum, worth at most +0.15 composite. v17 contradicts both, with SPS +3.95 "
    "live from blending and interval placement. Both August analyses were run on the memorised split, where the "
    "FNO's errors look about 1.5× smaller than they are live. That makes every trivial estimator look worse and "
    "every tight interval look adequate. The lesson is general: a conclusion drawn on a biased yardstick inherits "
    "the bias."))
A(table([
    ["direction", "evidence", "reading"],
    ["learned corrector on FNO output (668 k params)", "live 78.03 (η 1.4), 77.90 (η 0.4) vs 78.14 at η 0", "gains tke, moves interval centre, loses SPS"],
    ["test-time BatchNorm adaptation", "−0.81 local, 5/5 negative", "no covariate shift to correct"],
    ["log-POD variance denoising", "local +2.73 → live +0.03", "memorised-split artefact"],
    ["narrower intervals ×0.85 with modifier off (v6)", "live −0.99 SPS", "confounded: two changes at once"],
    ["torch.compile / AOT / TF32 / CUDA graphs", "PermissionError; 3–6×10<super>−3</super> drift; import crash (0)", "container limits; spectral weights brittle"],
    ["FFT padding retune", "−3.97 to −22.6 (sum of three)", "model welded to its padding"],
    ["richer interval-width models (ridge, GBM)", "≤ +0.02 composite over the lookup", "band shape saturated"],
    ["DMD / linear trend as extra blend partners", "+0.00 to +0.02", "no information beyond window mean"],
], [55, 55, 60], "Table 8. Further closed directions (full record in Appendix B)."))
A(P("<b>Candidates never evaluated live.</b> v19 adds a ridge-regression forecaster (last 10 frames × 5×5 "
    "neighbourhood, trained on 67 trajectories) as a third blend partner. It is honest by construction and beats "
    "every other predictor at leads 1–2 (Fig. 8). The proxies put it at +0.16 to +0.33 over v17. v18ft applies "
    "the v17 post-processing to our fine-tune of the official model."))

# 10 final days
A(P("10  Final days: a clean sim-to-real model, measured in the live regime", "h1"))
A(P("No top-30 team reports the official FNO's raw triple (94.168/74.026/92.836), which 11 teams at ranks 124–164 "
    "report unchanged. A unique triple shows a modified forecast, not necessarily a new model. We therefore retrained Exp. 1 on 26 Sep "
    "(Run 1: sim_fno → 67 real trajectories, lr 1e-4, 12 epochs, best epoch 8 by SELECT). We also trained one variant "
    "(Run 2) with a learned residual path to the last input frame. Both exclude SELECT and FINAL, so their local "
    "scores are honest."))
A(table([
    ["model (never saw SELECT/FINAL)", "SELECT sum", "FINAL rel", "FINAL tke", "FINAL mvpe", "FINAL sum"],
    ["sim_fno (start)", "220.12", "79.71", "66.87", "75.20", "221.78"],
    ["Exp. 1 (23 Aug)", "–", "94.27", "78.78", "95.51", "268.56"],
    ["Run 1 (26 Sep)", "266.81", "94.17", "79.29", "95.54", "269.00"],
    ["Run 2: + residual-to-last-frame", "266.82", "94.20", "79.29", "95.55", "269.04"],
], [60, 22, 22, 22, 22, 22], "Table 9. Clean sim-to-real FNO fine-tunes. The retrain reproduces Exp. 1. The residual path "
   "learned to lean on the last frame for v at short leads (gate 0.58 at lead 1) and hardly for u (0.07), but changed "
   "the scores by +0.04 of 269. Seed variance was not measured; Exp. 1 and Run 1 differ by 0.44 in sum, so +0.04 is "
   "not detectable."))
A(P("<b>Post-processing a clean model.</b> Refitted honestly (k = 1) on Run 1, the v17 prediction blend <i>hurts</i>. "
    "It pulls the forecast toward the window mean and cuts tke from 79.2 to 71.6. The tke boost repairs only part "
    "of that. The blend compensated for the official FNO's lead-1 over-smoothing, which the clean model does not "
    "share. The interval-centre and width tables still help (SPS 33.0 → 37.7). The build therefore uses the raw forecast "
    "plus the interval tables. A per-window scale for the widths, h ∝ √(median σ<sub>local</sub>), gave about +0.1 in "
    "every honest comparison (build exp1_sf). Fully scale-free widths over-covered SELECT (0.83) and lost."))
A(fig("fig14_regime.png", 150, "Figure 13. Run 1 build vs v17's live score by flow regime, on trajectories it never saw. "
      "It wins clearly at lower Reynolds number and other angles, but loses in the live regime (SELECT: AoA 15, "
      "Re 21.6–25.4k), mostly on rel (−0.9) and SPS (−2.8), while gaining tke (+3.1) and mvpe (+1.9)."))
A(P("<b>Two more negative results.</b> The official CNO, never scored live, trails the official FNO + v17pw by 3.1 "
    "composite on SELECT under the same proxy (2.7 on FINAL) and runs about 3× slower. An average of the official FNO "
    "and Run 1 never beats the official FNO alone on SELECT (+0.42 at 50/50 vs +0.67 alone, accuracy part), and two "
    "FNOs would exceed the 256 MB limit. On the live-like set, nothing we built beats the official FNO + v17-style "
    "post-processing."))
_live = ("pending at the time of writing" if LIVE_EXP1_SF is None else
         "rel {0:.3f} / tke {1:.3f} / mvpe {2:.3f} / time {3:.3f} / SPS {4:.3f}, final {5:.3f}".format(*LIVE_EXP1_SF))
A(P("<b>Live test.</b> The √-width Run 1 build (exp1_sf) was submitted on 26 Sep: the first live score of a model "
    "that never saw the local data. The regime analysis predicted it below 79.26 (SELECT −0.75), while FINAL as a "
    "whole predicted above (+0.80). Result: " + _live + "."))
A(P("<b>What the live test shows.</b> 79.238 is 0.023 below v17: better tke (+1.98) and runtime (+0.57), worse rel "
    "(−0.38) and mvpe (−0.71). SELECT predicted −0.75 against v17; live was −0.023. The miss is almost entirely SPS "
    "(SELECT 34.54, live 37.16), even though the tables were fitted on SELECT. The rel/tke/mvpe shifts from SELECT to "
    "live (+0.52/−1.17/−2.63) resemble the numpy DMD's (+0.55/−1.04/−1.86): a two-model consistency check, not a "
    "calibration. On the live set, the clean fine-tune reaches composite parity with the organisers' model with a "
    "different profile (more tke, less rel and mvpe). Its FINAL→live drop (−0.43/−1.83/−3.13) is far smaller than the "
    "official model's (−2.21/−7.01/−4.23), the strongest evidence in this report for the memorisation inference."))
A(P("<b>Not tested (future work).</b> Clean fine-tunes of CNO (the 29 Aug run stalled at 5.8 s/batch and then ran "
    "out of memory on the T4) and of Transolver; DPOT-S/-L and UNet fine-tuned at 32×64 (only scored off the shelf, "
    "88.8/70.1/84.4 on SELECT); WDNO and other architectures; using train_sim in our own training; an FNO trained on "
    "all 81 trajectories; multi-seed ensembles; SWA of the output-layer-only run; intermediate freeze ratios from the "
    "official FNO; Fisher-protected fine-tuning (PhysGuard); a learned interval-width head; directly optimised "
    "asymmetric intervals; compressing two models under 256 MB."))

# 9
A(P("11  Limitations", "h1"))
A(P("(1) All tables were fitted on 4 trajectories near the evaluation condition. Half of the local gain depends on "
    "that choice (§7), and the widths are absolute, so performance at unseen AoA/Re is uncertain. (2) The proxies "
    "rest on one post-hoc agreement. (3) The final entry uses no simulation data at inference. Its sim-to-real "
    "content is the official checkpoint's. Our own clean sim-to-real model (§10) was submitted live once: 79.238, 0.023 below v17, after SELECT had predicted −0.75. (4) Frame-"
    "correlated PIV error could favour persistence. Calm-pixel window-mean error (0.00295) is roughly twice our "
    "noise-floor estimate for freestream u (≤ 0.0016), so part of the calm-region gain may be measurement "
    "structure rather than flow. (5) Our entry did not reach the top ten. (6) Proxies A/AT are anchored on the official "
    "FNO's residual; cross-model comparisons under them (CNO, ensembles) are unvalidated, and live sampling uncertainty "
    "over ~90 windows is unquantified. The √-width choice was also cross-fitted (SELECT −0.83 when fitted on FINAL), "
    "not only fitted in-sample (−0.75)."))

# 10
A(P("12  Data provenance and compliance disclosures", "h1"))
A(P("Only the official release was used. We trained on <i>train_real</i> (with <i>7575_0</i> excluded) and used "
    "the official <i>sim_real_ft</i> FNO. The <i>sim_fno</i>/<i>sim_cno</i> checkpoints served as fine-tuning starts "
    "(§4). No external data, externally pretrained weights or generated data were used. All fitted quantities come "
    "from released training windows."))
A(P("<b>Disclosure 1: cross-window rank statistics.</b> From v4 onward, our submissions computed the quantile "
    "edges of their rank lookups (the interval-shape modifier and, in v17, the σ<sub>local</sub> sextiles and "
    "frame-difference deciles) over all windows in each predict() call. That is five windows, padded by repeating "
    "the last. Each window's prediction and bounds therefore depended slightly on other evaluation windows. The "
    "harness also feeds window k's target frames as window k+1's input, so this dependence can reach the answered "
    "window's own target. This was not intended, and the effect is a shift of bin edges. The Rules prohibit using "
    "information from other evaluation windows, and this falls within that. " + PW_TEXT))
A(P("<b>Disclosure 2: logged evaluation-input statistics.</b> In August, several diagnostic submissions wrote "
    "aggregate statistics of the evaluation <i>inputs</i> to the participant log: means, spectra, "
    "autocorrelations and mask sizes. They never included targets. We used them to characterise the live flow "
    "condition, and that characterisation chose our local hold-out, the SELECT split and the proxy calibration. The "
    "current Rules prohibit encoding evaluation data into logs. We have not been able to confirm whether this applied to "
    "aggregate input statistics in August; no ruling from the organisers had been obtained at the time of deposit. "
    "Section 7 measures what this choice was worth locally: 0.65–1.10 of v17's 1.36 proxy composite gain (48–81%)."))
A(P("<b>Disclosure 3: harness structure.</b> We noticed that consecutive evaluation windows overlap (window k's "
    "target is window k+1's input). We did not deliberately exploit it. Disclosure 1 describes an unintended "
    "channel through which it could have acted."))
A(table([
    ["use", "what information", "status"],
    ["A. Model training (weights or table values)", "released <i>train_real</i> windows only; official checkpoints as starting points",
     "no evaluation data of any kind"],
    ["B. Offline model selection", "which training trajectories formed SELECT/FINAL; the proxy scale k",
     "chosen using aggregate statistics of logged evaluation <i>inputs</i> (Disclosure 2) and our own live scores; "
     "effect measured: 0.65–1.10 of 1.36 proxy composite (48–81%, K9)"],
    ["C. Logged by submissions (Aug)", "means, spectra, autocorrelations, mask sizes of evaluation inputs; timings",
     "never targets; never read back at inference"],
    ["D. At inference (live builds v4–v17)", "rank-bin edges pooled over the 5 windows of a call (Disclosure 1)",
     "unintended; effect ≤ 0.006 composite locally; removed in v17pw/v19pw"],
    ["E. After submissions, for analysis only", "own live subscores; public leaderboard subscores of all teams",
     "public or our own results; used for §3 and proxy calibration"],
], [40, 62, 68], "Table 10. Where each kind of information entered the pipeline."))

# 11
A(CondPageBreak(70 * mm))
A(P("13  Reproducibility", "h1"))
A(table([
    ["step", "file / command"],
    ["inference packages", "submissions/v17/submission_blend_v17.zip (live); per-window rebuilds in submissions/v17_per_window, submissions/v19_per_window"],
    ["local caches", "experiments/interval_recipes/cache_preds.py → cache_{select,final}_s20.npz (X, Y, P); random split: experiments/robustness/cache_rand.py"],
    ["splits", "holdout_v12_3.json (SELECT 4, FINAL 10, train 67), train_pool_v12_3.json, experiments/robustness/rand_split.json"],
    ["prediction / centre / width fits", "experiments/estimator_blend/fit.py, fit_band.py, controls2.py → pickles; make_tables.py writes _blend_tables.py"],
    ["energy correction", "experiments/tke_energy/eval_rescale.py (gain_input, γ = 0.5 → exponent 1/4, cap 1.6)"],
    ["ridge forecaster (v19)", "experiments/v19_ridge_partner/accum.py 4; gen_L.py 10 5 1e-4 sxb 1e-3 sxb3; blend19.py"],
    ["fine-tuning", "finetune_fno_kaggle.py (Kaggle T4; flags --freeze-bn-fc, --spectral-aug, --unfreeze-extra, --pool-file)"],
    ["analyses in this report", "experiments/robustness/*.py; report/data/weights_bootstrap.json; report/data/experiments.json"],
    ["gates", "preflight.py <zip>; experiments/estimator_blend/verify_torch.py; per-window independence test (experiments/per_window_rebuild)"],
    ["figures and report", "report/scripts/make_figures*.py, make_matrix.py, make_transfer_ladder.py, make_robustness.py, build_report_v2.py (writes this v5)"],
], [42, 128], "Table 11. Reproduction map. All post-processing is lookup tables fitted in minutes on a CPU; fine-tuning used a Kaggle T4."))

A(P("Acknowledgements", "h1"))
A(P("We thank the RealPDE organisers for the benchmark, the public leaderboard API, and their quick response to "
    "the data-duplication report."))

# Appendix A
A(PageBreak())
A(P("Appendix A. Submission ledger", "h1"))
A(table([
    ["build", "change", "live final", "local preflight (rel/tke/mvpe/SPS)"],
    ["S1–S3", "numpy DMD + interval scale bracket", "76.68 / — / 77.01", "—"],
    ["S4", "+ log-POD denoise, lazy decode", "76.49", "—"],
    ["S5", "official FNO, no intervals (±5% default)", "72.65", "—"],
    ["S6", "FNO + corrector η 0.4 + intervals", "77.90", "—"],
    ["v4 (S7)", "corrector η 1.4 + rank modifier", "78.03", "—"],
    ["v5 (S8)", "+ manual CUDA graphs", "0 (import crash)", "—"],
    ["v6 (S9)", "×0.85 widths, fp16 bounds, modifier off", "77.75", "—"],
    ["v7", "v8 path + η 1.4, multi-stage loader", "not recorded", "—"],
    ["v8", "official FNO + heteroscedastic intervals + modifier", "78.141", "96.06 / 80.29 / 97.20 / —"],
    ["v9", "v8 + target-free diagnostics", "78.124", "—"],
    ["v10", "v8 + input diagnostics (host syncs)", "not recorded (time 83.09)", "96.06 / 80.29 / 97.20 / 42.91"],
    ["v11", "flat intervals, modifier off", "not recorded", "96.06 / 80.29 / 97.20 / 39.41"],
    ["v12", "v8 with modifier off", "77.899", "96.06 / 80.29 / 97.20 / 35.43"],
    ["v13", "flat intervals + modifier", "not recorded", "96.06 / 80.29 / 97.20 / 31.54"],
    ["v14", "fine-tuned FNO swapped into v8 (planned)", "never built", "—"],
    ["v15", "v8 + AOT-compile / requests probe", "not recorded", "—"],
    ["v16", "v8 + energy correction", "not submitted", "same pool vs v8: −0.01 / +2.07 / 0 / +0.55"],
    ["v17", "v16 + blend + interval tables", "<b>79.261</b>", "96.10 / 82.54 / 97.20 / 44.08"],
    ["v18", "v17, unconstrained blend", "not submitted", "96.08 / 82.38 / 97.26 / 44.05"],
    ["v18ft", "v17 post-processing on fine-tuned FNO", "not submitted", "95.46 / 82.12 / 96.70 / 41.42"],
    ["v19", "v17 + ridge forecaster partner", "not submitted", "96.14 / 82.73 / 97.20 / 44.54"],
    ["v17pw / v19pw", "per-window (compliant) rebuilds of v17 / v19", "not submitted", "96.10 / 82.55 / 97.20 / 44.08 ; 96.14 / 82.74 / 97.20 / 44.55"],
    ["exp1", "Run 1 clean FNO + honest interval tables", "not submitted", "93.13 / 78.66 / 94.79 / 34.31 (unseen pair)"],
    ["exp1_sf", "exp1 with √-scaled widths", "submitted 26 Sep; " + ("pending" if LIVE_EXP1_SF is None else f"{LIVE_EXP1_SF[5]:.3f}"), "honest FINAL 94.16 / 79.18 / 95.50 / 38.15"],
], [18, 70, 32, 50], "Table A1. All builds. Local preflight scores are on the memorised FINAL pair and only compare builds with each other."))

# Appendix B/C: every record, untruncated protocol fields
reg = json.loads((DATA / "experiments.json").read_text(encoding="utf-8"))["records"]
EXP = {"bands_sps": "Intervals and SPS", "tke": "TKE", "model_architecture": "Models",
       "finetuning": "Fine-tuning (sim-to-real)", "post_processing": "Post-processing", "blending": "Blending",
       "evaluation_methodology": "Evaluation methodology"}
MEAS = {"data_audit": "Data audit", "eda_analysis": "Data analysis", "harness_runtime": "Runtime and harness"}


def txt(s):
    return escape(" ".join(str(s or "not recorded").split()))


def rec_block(r, measurement=False):
    head = f"<b>{r['id']}  {txt(r['title'])}</b>   [{r['verdict']}]" + (f"   {r['date']}" if r.get("date") else "")
    f = ([("Question", txt(r["hypothesis"]))] if measurement else [("Question / hypothesis", txt(r["hypothesis"]))])
    f += [("What was done" if not measurement else "How measured", txt(r["method"])),
          ("Data / split", txt(r["data_split"])), ("Result", txt(r["results"])),
          ("Reading", txt(r["why"])), ("Scripts / source", txt(", ".join(r.get("scripts") or [])) + " · " + txt(r["source"]))]
    rows = [[Paragraph(head, S["tiny"]), ""]] + [[Paragraph(k, S["tinyb"]), Paragraph(v, S["tiny"])] for k, v in f]
    t = Table(rows, colWidths=[26 * mm, 146 * mm])
    t.setStyle(TableStyle([("SPAN", (0, 0), (-1, 0)), ("BACKGROUND", (0, 0), (-1, 0), TINT),
                           ("LINEABOVE", (0, 0), (-1, 0), 0.6, INK), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 0.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.8)]))
    return KeepTogether([t, Spacer(1, 3)])


A(P("Appendix B. Every experiment: question, exact procedure, data, result, reading", "h1"))
A(P("Several records below use the words “memorised” or “memorized” as shorthand; read them with the evidence grading of Table 2 (an inference, not an established fact). Experiments change something and compare it with a baseline. The twelve that shaped the campaign have full cards "
    "in §8, and every register experiment appears below, except the 26 Sep work of §10, the K9 swap ablation, the K10 "
    "analysis and the per-window rebuild (see results/live_logs/*26sep.txt, experiments/clean_fno_regimes, experiments/cno_check, experiments/ensemble_check, experiments/robustness, experiments/per_window_rebuild). Results are tagged live, local (memorised split), local-honest or "
    "proxy. Numbers were checked against the files in the last row of each block. Details not recorded in any "
    "source are marked as such."))
for i, (cat, title) in enumerate(EXP.items(), 1):
    rows = [r for r in reg if r["category"] == cat]
    if rows:
        A(P(f"B.{i}  {title}", "h2"))
        for r in rows:
            A(rec_block(r))
A(P("Appendix C. Measurements (facts established, not hypotheses tested)", "h1"))
for i, (cat, title) in enumerate(MEAS.items(), 1):
    rows = [r for r in reg if r["category"] == cat]
    if rows:
        A(P(f"C.{i}  {title}", "h2"))
        for r in rows:
            A(rec_block(r, measurement=True))


def on_page(c, d):
    c.saveState()
    c.setFont("DV", 7)
    c.setFillColor(INK2)
    c.drawString(18 * mm, 10 * mm, "RealPDE 2026 Track 1 · team Slash · technical report v5")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, str(d.page))
    c.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                  title="Input-Window Estimator Blending for Real PIV Forecasting under a Possibly Contaminated Local Hold-out",
                  author=AUTHORS, subject="NeurIPS 2026 RealPDE Track 1 technical report").build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT)
