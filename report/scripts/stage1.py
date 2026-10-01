"""Stage 1: report_stage0.py -> report_stage1.py. Own-work framing with 1st place as the only
benchmark, figures inline and in an atlas, automatic figure renumbering."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
t = (HERE / "report_stage0.py").read_text(encoding="utf-8")


def sub(a, b, count=1):
    global t
    n = t.count(a)
    assert n == count, (n, a[:90])
    t = t.replace(a, b)


sub('VERSION = "v5.0 (26 Sep 2026, before the development phase closed)"', 'VERSION = "v6.0 (26 Sep 2026, final)"')
sub("technical_report_v5.pdf", "technical_report_v6.pdf")
sub('technical report v5")', 'technical report v6")')

# ---- §2: add split map
sub('"Live mvpe, below all 81 training trajectories (Fig. 1), is not explained by them."))',
    '"Live mvpe, below all 81 training trajectories (Fig. 1), is not explained by them."))\n'
    'A(fig("fa13_splits.png", 150, "Figure 99. Our split of the 81 usable real trajectories: 67 for training, SELECT (4 AoA-15 '
    'trajectories at Re 21.6–25.4k, the regime of the live set, shaded) and FINAL (10). The duplicate 7575_0 is excluded."))')

# ---- §3: replace the other-teams section with our live record + 1st-place benchmark
i = t.index('# 3\nA(P("3  Supporting evidence from the public leaderboard", "h1"))')
j = t.index('# 4', i)
t = t[:i] + '''# 3
A(P("3  Our live record and the benchmark", "h1"))
A(P("We made at least 16 live submissions (S1–S9, v8, v9, v10, v12, v15, v17, exp1_sf); nine have fully recorded "
    "subscores and are plotted below. The only external number used in this report is the "
    "1st-place score on the development board, 82.151 (rel 94.746, tke 80.191, mvpe 94.243, time 92.030, SPS 44.967), as "
    "the maximum achieved. Our best is 79.261 (v17), 2.89 below it. Our best build without cross-window pooling is "
    "79.238 (exp1_sf, §10)."))
A(fig("fa03_score_over_time.png", 140, "Figure 99. Our best live score over the development phase, against the 1st-place "
      "benchmark."))
A(fig("fa01_live_subscores.png", 158, "Figure 99. Every live submission with recorded subscores, per metric, against the "
      "1st-place benchmark. S5 shipped without intervals (SPS 14.1). v17 and exp1_sf are the two final approaches: the "
      "official FNO with our post-processing, and our own clean sim-to-real FNO."))
A(fig("fa02_vs_first.png", 150, "Figure 99. Our two final approaches against the 1st-place benchmark, per subscore. The "
      "largest distance is SPS (−7.7) and tke (−2.8 for the clean model, −4.8 for v17)."))
A(fig("fa10_sps_parts.png", 150, "Figure 99. SPS of each live build split into the accuracy factor W̄ and the band factor "
      "T (SPS ≈ 100·W̄·T). Accuracy barely moved across our builds; every SPS gain came from the intervals: default bands "
      "→ heteroscedastic (T 0.21 → 0.48), then v17's centre and width tables (0.49 → 0.55)."))
A(fig("fa04_runtime.png", 150, "Figure 99. Measured runtime per sample on the evaluation server for each build."))
A(P("<b>Scoring sensitivities (method note).</b> To rank candidates, we use a linear approximation of how final_score "
    "responds to each subscore, estimated from the public leaderboard: per subscore point rel 0.453, tke 0.087, "
    "mvpe 0.081, time 0.093, SPS 0.284 (bootstrap CIs in Appendix A of report/data/weights_bootstrap.json). Two of our "
    "own bit-identical builds that differed only in runtime give a directly measured time sensitivity of 0.095, and show "
    "that the board is deterministic for accuracy and SPS (only runtime varies, ±0.017 final)."))

''' + t[j:]

# ---- Table 4: 1st place instead of top-10 mean
sub('    ["top-10 mean, 25 Sep", "94.78", "79.64", "94.16", "92.15", "44.30", "81.95", ""],',
    '    ["1st place (benchmark), 26 Sep", "94.746", "80.191", "94.243", "92.030", "44.967", "82.151", ""],')
sub('"Table 4. Live development-phase results. Rank 121 → 87 of 178 non-zero entries; top-10 cut-off 81.775. The v17 "',
    '"Table 4. Live development-phase results against the 1st-place benchmark. The v17 "')

# ---- §10 opening
sub('A(P("No top-30 team reports the official FNO\'s raw triple (94.168/74.026/92.836), which 11 teams at ranks 124–164 "\n'
    '    "report unchanged. A unique triple shows a modified forecast, not necessarily a new model. We therefore retrained Exp. 1 on 26 Sep "',
    'A(P("Everything above post-processes the official FNO, and its local scores cannot be trusted (§5). To have a model "\n'
    '    "we can evaluate honestly, we retrained Exp. 1 on 26 Sep "')
sub('"regime, where it trails on rel and SPS (§10)."', '"regime, where it trails on rel and SPS (§10)."', 0) if False else None

# ---- §10: inline clean-vs-official figures after the regime figure
sub('A(P("<b>Two more negative results.</b>',
    'A(fig("fb01_error_by_lead.png", 158, "Figure 99. Error by forecast lead of our clean FNO against the official FNO, '
    'persistence and the window mean, on SELECT (live regime) and FINAL. Our model never saw these trajectories; the '
    'official one very likely did."))\n'
    'A(fig("fb02_panels.png", 158, "Figure 99. One live-regime SELECT window (u): target at lead 10, the two models\' '
    'errors, and the temporal-variance (tke) maps. Our clean FNO\'s variance map is stronger but more diffuse, and its '
    'pointwise errors in the wake are larger."))\n'
    'A(fig("fa14_final_to_live.png", 120, "Figure 99. Score change from local FINAL to live. The official FNO, which '
    'very likely trained on FINAL, loses far more than our clean FNO, which never saw it."))\n'
    'A(P("<b>Two more negative results.</b>')

# ---- atlas section before Limitations; renumber later sections
sub('A(P("11  Limitations", "h1"))', '''A(P("11  Figure atlas of our work", "h1"))
A(P("The remaining figures document the experiments of §4–§10 visually. Each caption says what was run and what it shows."))
A(fig("fa05_training_curves.png", 158, "Figure 99. Kaggle training curves of our two clean sim-to-real runs (26 Sep): "
      "training loss, SELECT sum (unseen) and SELECT rel/tke by epoch. Nearly all the gain comes in epoch 0; both "
      "plateau from epoch 3 and peak at epoch 8."))
A(fig("fa06_resid_gate.png", 140, "Figure 99. Run 2's learned weight on the last input frame by lead and epoch. It uses the "
      "last frame strongly for v at short leads and hardly for u. It converges by epoch 3."))
A(fig("fa07_finetune_deltas.png", 150, "Figure 99. Our fine-tunes of the official FNO and the learned corrector, against "
      "the frozen official FNO on local FINAL. Every configuration that buys tke sells rel and mvpe."))
A(fig("fa08_waterfall.png", 130, "Figure 99. How v17's gain over v8 builds up under the live-calibrated proxy: prediction "
      "blend with energy correction, then the interval-centre table, then the width table."))
A(fig("fa09_swap.png", 150, "Figure 99. Swap ablation: v17's tables fitted on SELECT, with one component at a time "
      "replaced by its randomly-fitted counterpart. The width table carries most of the dependence on the fitting split."))
A(fig("fa11_width_tables.png", 150, "Figure 99. The fitted interval half-width tables of our two final builds, by lead "
      "group and input frame-difference decile. Widths grow with input activity and lead; exp1_sf's are scaled per window."))
A(fig("fa12_transfer_scatter.png", 115, "Figure 99. Local prediction against live result for every change we could check. "
      "The dotted line is perfect transfer. Early changes validated on possibly-memorised data fell far below it; v17 "
      "(proxy AT) and our clean model sit close to it."))
A(fig("fb03_per_traj.png", 158, "Figure 99. Relative L2 error per trajectory of our clean FNO and the official FNO on "
      "SELECT (left of the dotted line) and FINAL. Our model never saw these trajectories; the official one very likely did."))
A(fig("fb04_energy.png", 125, "Figure 99. Fluctuation energy of the forecast relative to the target, by lead. Our clean FNO "
      "over-shoots at short leads (up to 1.38× at lead 1) and falls to the official model's level after lead 6; both lose "
      "energy at long leads, the source of the tke deficit."))
A(fig("fb05_spectrum.png", 115, "Figure 99. Spatial power spectrum of u at lead 10 on SELECT. Both models under-represent "
      "fine scales by a similar amount; the spectra are nearly identical."))

A(P("12  Limitations", "h1"))''')
sub('A(P("12  Data provenance and compliance disclosures", "h1"))', 'A(P("13  Data provenance and compliance disclosures", "h1"))')
sub('A(P("13  Reproducibility", "h1"))', 'A(P("14  Reproducibility", "h1"))')
t = t.replace("§12", "§13")
sub('"structure rather than flow. (5) Our entry did not reach the top ten.',
    '"structure rather than flow. (5) Our best score (79.261) is 2.89 below the 1st-place benchmark (82.151).')

# ---- §2 pointer to the full details
sub('A(fig("fa13_splits.png"',
    'A(P("Full data, harness, training and testing details, with figures, are in Appendix D."))\nA(fig("fa13_splits.png"')

# ---- Appendix D: data, training and testing details (from CAMPAIGN_REPORT.md §4-5 and §13, docs/KAGGLE_RUNS.md,
#      docs/FOLLOW_UP_EXPERIMENTS.md, the Kaggle logs, finetune_fno_kaggle.py defaults and our own h5 audit)
sub("def on_page(c, d):", '''A(PageBreak())
A(P("Appendix D. Data, evaluation harness, training and testing: full details", "h1"))
A(P("D.1  The released real data (our audit of train_real)", "h2"))
E(table([
    ["item", "value"],
    ["files", "82 .h5 files (datasets aoa, re, t, u, v, x, y); 81 usable: 7575_0 is byte-identical to 6300_0 and excluded"],
    ["conditions", "AoA 0, 5, 10, 15, 20°; 18 Reynolds labels 3,750–26,700 in the file names"],
    ["Reynolds attribute", "78 of 82 files store a Re that differs from the file name; true step 1,278–1,279 (3750, 5028, 6306, …, 26761)"],
    ["length and time step", "282–868 frames per trajectory; Δt = 0.02 s"],
    ["field size", "raw 64 × 128; used at 32 × 64 via [::2, ::2]; channels u, v (p ≡ 0 for PIV)"],
    ["masked vectors", "exact zeros where PIV has no data (airfoil body, laser shadow): 9.5% of pixels on average (9–11%)"],
    ["normalisation", "input mean (0.15496, −0.00051, 0), std (0.09681, 0.01596, 1); target mean (0.15496, −0.00052, 0), std (0.09681, 0.01596, 1)"],
    ["train_sim", "same grid; u_std 3.59× and v_std 10.81× the real data (anisotropic); KE-map correlation 0.69; 0.20% zeros vs 9.91%; not used for training"],
], [38, 132], keep=False))
A(fig("fd01_lengths.png", 150, "Figure 99. Length of every released real trajectory by Reynolds label and angle of attack."))
A(fig("fd02_re_mismatch.png", 150, "Figure 99. The Reynolds number stored in each file against the one in its name: 78 of "
      "82 disagree, following a 1,278–1,279 step rather than 1,275."))
A(fig("fd03_mask.png", 150, "Figure 99. Fraction of masked (zero) PIV vectors in the first frame of each trajectory; "
      "dotted line: mean 9.5%."))
A(P("D.2  Windows and splits", "h2"))
E(table([
    ["set", "trajectories", "windows (stride)", "use"],
    ["training pool", "67", "27,193 (stride 2, Kaggle)", "fine-tuning; ridge forecaster (13,629 windows, stride 4)"],
    ["SELECT", "4: 21600_15, 22875_15, 24150_15, 25425_15", "168 (stride 20) · 1,660 (stride 2)", "epoch choice, table fitting; live-regime check"],
    ["FINAL", "10 (AoA 10/15/20, Re 17.8–25.4k)", "420 (stride 20) · 4,150 (stride 2)", "touched once per experiment"],
    ["preflight pair", "21600_15, 25425_15 (part of SELECT)", "–", "packaging gate score"],
    ["random split (K9)", "10125_5, 12675_10, 7575_15, 3750_20", "168 (stride 20)", "fingerprint-free refit"],
], [26, 52, 42, 50], keep=False))
A(P("Each window is 20 input frames followed by 20 target frames. Only pixels whose target is non-zero are scored."))
A(P("D.3  The evaluation harness (measured from our submissions)", "h2"))
E(table([
    ["item", "value"],
    ["hardware / software", "NVIDIA A800-SXM4-80GB; torch 2.2.2; numpy 1.26.4"],
    ["process layout", "17 processes, each: import, one untimed synthetic warm-up call, one timed call of n = 5 windows (~90 scored)"],
    ["what is timed", "only predict() of the timed call; import, checkpoint load and warm-up are free; ~2.9–3.6 ms/sample charged outside predict()"],
    ["unavailable", "torch.compile and AOT compilation (PermissionError); TF32 (deviation 3–19×10⁻³, auto-reverted); fp16 FFT (needs power-of-two sizes; ours 26/38/70)"],
    ["limits", "256 MB zip; 5 min per run; 10,240 B of stderr shown; bounds must be finite with lower ≤ upper or every subscore is zero"],
    ["contention", "host synchronisations inside predict() cost up to 7.8× under the 17 concurrent processes"],
], [38, 132], keep=False))
A(P("D.4  The live evaluation set (from logged input statistics; see Disclosure 2)", "h2"))
A(P("A single flow condition, AoA 15°, Re ≈ 20,600–25,200, with masked cells exactly zero (9–11%). Its temporal "
    "autocorrelation and spectrum match local trajectories of the same regime; its amplitude statistics are larger."))
A(fig("fd04_fingerprint.png", 145, "Figure 99. Live evaluation-set statistics relative to our local reference pool."))
A(fig("fd05_live_diag.png", 158, "Figure 99. Live input diagnostics: u radial spectrum (live vs local), temporal "
      "autocorrelation (faster decay than AR(1), the vortex-shedding signature, identical live and local), and the live σ_local(u) "
      "histogram."))
A(CondPageBreak(60 * mm))
A(P("D.5  Training (Kaggle Tesla T4, 14.56 GB)", "h2"))
E(table([
    ["setting", "value"],
    ["FNO fine-tune (Run 1/2, Exp. 1)", "start sim_fno.pth; AdamW lr 1e-4, weight decay 1e-5, cosine decay; batch 4 × accumulation 4; "
     "1,200 steps/epoch (4,800 windows); 12 epochs; input noise σ 0.02; fp32; ~222 ms/batch, ~300 s/epoch; best epoch by SELECT sum"],
    ["loss", "weighted rel-L2 / tke / mvpe errors, weights 0.734 / 0.106 / 0.160 (score sensitivity × composite weight at the start errors)"],
    ["Run 2 extra", "residual-to-last-frame gate (20 leads × 2 channels), initialised at 0, lr 1e-2"],
    ["official-FNO fine-tunes", "lr 2e-5 (Exp. 2/3), stride 20 or 2, loss weights varied (FT01/FT02), amplitude jitter 0.70 (FT02)"],
    ["output-layer-only", "9,667 of 50,357,955 parameters trainable (fc0, fc1, fc2, BatchNorm affine); spectral augmentation; 30 epochs"],
    ["CNO fine-tune", "batch 16 × accumulation 8 with AMP (batch 64/32 ran out of memory); ~5.8 s/batch; never completed"],
    ["ridge forecaster (v19)", "last 10 frames × 5×5 u,v neighbourhood + window mean (549 features); λ 1e-4; per σ-sextile × 4×4 block"],
    ["post-processing tables", "fitted on SELECT in minutes on a CPU; blend by constrained least squares; widths by maximising exp(−2h/σ)·F̂(h)"],
], [38, 132], keep=False))
A(P("D.6  Testing and validation", "h2"))
E(table([
    ["check", "what it does"],
    ["official scorer", "scoring.py from starting kit v9, run locally on cached windows"],
    ["honest hold-out", "SELECT/FINAL for models that never saw them (sim_fno start); FINAL touched once"],
    ["proxies (official FNO)", "A: Y′ = P + k(Y − P), k_rel 1.534 (SELECT) / 1.665 (FINAL); AT: k_T 1.394 / 1.447; B/BT degrade P instead"],
    ["preflight.py", "zip size and layout, fresh-process import, warm-up + real call, dtypes, stderr budget, failure-injection fallback, scorer acceptance"],
    ["independence test", "every window's prediction and bounds identical (< 10⁻⁶) whether predicted alone, in batches, shuffled, padded or duplicated"],
    ["torch vs numpy", "the shipped submission.py reproduces the numpy evaluation to ≤ 1.5×10⁻⁷ given the same bins"],
    ["live board", "deterministic for rel/tke/mvpe/SPS; runtime varies ±0.017 final"],
], [38, 132], keep=False))


def on_page(c, d):''')

# ---- appendix: drop the record that analyses other teams (EM09)
t = t.replace('rows = [r for r in reg if r["category"] == cat]', 'rows = [r for r in reg if r["category"] == cat and r["id"] != "EM09"]')

# ---- automatic figure renumbering: captions "Figure N." and text references "Fig. N"
calls = list(re.finditer(r'fig\("([^"]+)", \d+, "Figure (\d+)\.', t))
old_to_file = {int(m.group(2)): m.group(1) for m in calls if m.group(2) != "99"}
order = {}
for m in calls:
    order.setdefault(m.group(1), len(order) + 1)
file_to_new = order


def cap(m):
    return f'fig("{m.group(1)}", {m.group(2)}, "Figure {file_to_new[m.group(1)]}.'


t = re.sub(r'fig\("([^"]+)", (\d+), "Figure \d+\.', cap, t)


def ref(m):
    old = int(m.group(1))
    f = old_to_file.get(old)
    return f"Fig. {file_to_new[f]}" if f else m.group(0)


t = re.sub(r"Fig\. (\d+)", ref, t)
(HERE / "report_stage1.py").write_text(t, encoding="utf-8")
print("figures:", len(file_to_new), "| old refs mapped:", sorted(old_to_file))
