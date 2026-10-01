"""One command for the report.

    python report/scripts/build_all.py              # report from the committed figures (seconds, no data needed)
    python report/scripts/build_all.py --figures    # regenerate every figure first, then the report

Figure scripts read the released train_real data, the local forecast caches (experiments/interval_recipes/cache_*_s20.npz, rebuilt by
experiments/interval_recipes/cache_preds.py) and the analysis outputs in experiments/. A figure script whose inputs are missing is reported
and its committed PNG is kept, so the report always builds.
The text is assembled in stages: report_stage0.py -> stage1.py -> stage2.py -> stage3.py -> build_report.py.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = ["make_figures.py", "make_figures2.py", "make_figures3.py", "make_figures4.py", "make_figures5.py",
           "make_regime.py", "make_matrix.py", "make_transfer_ladder.py", "make_robustness.py", "make_overview.py",
           "make_evidence.py"]
REPORT = ["stage1.py", "stage2.py", "stage3.py", "build_report.py"]


def run(script):
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    ok = r.returncode == 0
    last = (r.stdout.strip().splitlines() or [""])[-1] if ok else (r.stderr.strip().splitlines() or ["?"])[-1]
    print(f"  {'ok  ' if ok else 'FAIL'} {script:26} {time.time() - t0:5.1f}s  {last[:90]}")
    return ok


if "--figures" in sys.argv:
    print("figures:")
    failed = [s for s in FIGURES if (HERE / s).exists() and not run(s)]
    if failed:
        print("  kept the committed PNGs of:", ", ".join(failed))
print("report:")
for s in REPORT:
    if not run(s):
        sys.exit(f"report stage {s} failed")
print("done:", HERE.parent / "RealPDE2026_Track1_Slash_technical_report.pdf")
