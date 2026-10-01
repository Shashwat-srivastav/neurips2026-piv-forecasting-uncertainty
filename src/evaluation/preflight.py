#!/usr/bin/env python3
"""Pre-submission gate. Run this BEFORE spending a daily slot.

    python preflight.py submission_fno_boosted_v6.zip

Exits 0 only if every HARD gate passes. Anything else means do not submit.

Why this exists: submission 884075 scored 0.000000 on all five metrics because
an exception escaped during module import and the process died before serving a
single batch. Per-function try/except was not enough, and none of the local
scoring checks caught it -- they all ran in an already-warm interpreter that had
imported the module successfully in a different configuration.

So the central test here is TEST 3: import the module in a FRESH SUBPROCESS,
exactly as the harness does, and see whether it survives. Everything else is
secondary to that.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'src', d) for d in ('training', 'evaluation', 'splits')]
KIT = os.path.join(REPO, "realpde_t1_starting_kit_v9", "realpde_t1_starting_kit_v9")
SCORING = os.path.join(KIT, "scoring.py")
SPLITS = os.path.join(REPO, "data/splits/matched_splits.json")
LIMIT_MB = 256
LOG_CAP = 10240          # bytes of participant stderr the harness shows
N_PROC = 17              # processes the harness runs
BATCH = 5                # samples per predict() call

# Known-good reference on the FINAL pool. Update deliberately, never casually.
BASELINE = dict(rel=95.8382, tke=82.9980, mvpe=97.0126)

CHILD = r'''
import io, json, os, sys, contextlib, traceback
import numpy as np
res = {"ok": False, "stage": "start", "log_bytes": 0, "err": None}
buf = io.StringIO()
try:
    res["stage"] = "import"
    with contextlib.redirect_stderr(buf):
        import submission as S
    res["log_bytes"] = len(buf.getvalue().encode())
    res["stage"] = "has_predict"
    assert callable(getattr(S, "predict", None)), "predict is not callable"

    rs = np.random.RandomState(0)
    SH = (5, 20, 32, 64, 3)

    # the harness sends a synthetic Gaussian warmup first, then real PIV
    res["stage"] = "call_synthetic"
    synth = (0.153 + 0.096 * rs.randn(*SH)).astype(np.float32)
    with contextlib.redirect_stderr(buf):
        o1 = S.predict(synth)

    res["stage"] = "call_real"
    real = (0.218 + 0.150 * rs.randn(*SH)).astype(np.float32)
    real[:, :, :4, :6, :] = 0.0
    real[..., 2] = 0.0
    with contextlib.redirect_stderr(buf):
        o2 = S.predict(real)

    res["stage"] = "contract"
    for k in ("prediction", "lower", "upper"):
        assert k in o2, "missing key " + k
    p = np.asarray(o2["prediction"]); lo = np.asarray(o2["lower"]); up = np.asarray(o2["upper"])
    assert p.shape == SH, f"prediction shape {p.shape} != {SH}"
    assert lo.shape == p.shape and up.shape == p.shape, "bound shape mismatch"
    assert np.isfinite(p).all(), "prediction has non-finite values"
    assert np.isfinite(lo).all() and np.isfinite(up).all(), "bounds have non-finite values"
    assert (lo <= up).all(), "lower > upper somewhere"
    res["dtypes"] = [str(p.dtype), str(lo.dtype), str(up.dtype)]

    res["stage"] = "ragged"
    for n in (1, 2, 3, 4, 5, 7, 11):
        q = np.asarray(S.predict(real[:1].repeat(n, 0))["prediction"])
        assert q.shape[0] == n, f"n={n} returned {q.shape[0]}"

    res["stage"] = "dirty_input"
    bad = real.copy(); bad[0, 0, 0, 0, 0] = np.nan; bad[1, 2, 3, 4, 1] = np.inf
    inp = bad.copy()
    q = np.asarray(S.predict(bad)["prediction"])
    assert np.isfinite(q).all(), "NaN input produced non-finite output"
    fin = np.isfinite(inp)
    assert np.array_equal(bad[fin], inp[fin]), "predict() MUTATED the caller's array"

    res["stage"] = "repeat"
    for _ in range(4):
        S.predict(real)

    res["log_bytes"] = len(buf.getvalue().encode())
    res["ok"] = True
    res["stage"] = "done"
except Exception as e:
    res["err"] = f"{type(e).__name__}: {e}"
    res["tb"] = traceback.format_exc()[-800:]
print("PREFLIGHT_JSON " + json.dumps(res))
'''

SCORE_CHILD = r'''
import io, json, os, sys, contextlib, importlib.util
import numpy as np, h5py
res = {"ok": False}
buf = io.StringIO()
try:
    with contextlib.redirect_stderr(buf):
        import submission as S
    sp = importlib.util.spec_from_file_location("sc9", sys.argv[1])
    sc = importlib.util.module_from_spec(sp); sp.loader.exec_module(sc)
    root = json.load(open(sys.argv[2]))["root"]
    X, Y = [], []
    for t in ("21600_15", "25425_15"):
        with h5py.File(os.path.join(root, t + ".h5")) as f:
            u = np.asarray(f["u"][:, ::2, ::2], np.float32)
            v = np.asarray(f["v"][:, ::2, ::2], np.float32)
        fr = np.stack([u, v, np.zeros_like(u)], -1)
        st = list(range(0, len(fr) - 40 + 1, 20))
        X.append(np.stack([fr[i:i+20] for i in st]))
        Y.append(np.stack([fr[i+20:i+40] for i in st]))
    X, Y = np.concatenate(X), np.concatenate(Y)
    P, L, U = [], [], []
    with contextlib.redirect_stderr(buf):
        for i in range(0, len(X), 5):
            o = S.predict(X[i:i+5])
            P.append(o["prediction"]); L.append(o["lower"]); U.append(o["upper"])
    p = np.concatenate(P).astype(np.float32)
    lo = np.concatenate(L); up = np.concatenate(U)
    w, cov = sc.aggregate_sps(p, Y, 2, lo, up)      # raises if bounds are invalid
    res.update(ok=True, coverage=float(cov),
               rel=sc.score_error(float(sc.rel_l2_per_sample(p, Y, 2).mean())),
               tke=sc.score_error(float(sc.tke_rel_l2_per_sample(p, Y, 2).mean())),
               mvpe=sc.score_error(float(sc.mvpe_rel_l2_per_sample(p, Y).mean())),
               sps=sc.score_sps(w))
except Exception as e:
    res["err"] = f"{type(e).__name__}: {e}"
print("PREFLIGHT_JSON " + json.dumps(res))
'''

hard_fail: list[str] = []
warn: list[str] = []


def run_child(workdir: str, code: str, args: list[str] | None = None, timeout: int = 1800):
    """Run code in a FRESH interpreter with workdir on sys.path."""
    path = os.path.join(workdir, "_preflight_child.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(code)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = workdir + os.pathsep + env.get("PYTHONPATH", "")
    try:
        cp = subprocess.run([sys.executable, path] + (args or []), cwd=HERE, env=env,
                            capture_output=True, text=True, timeout=timeout,
                            encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"ok": False, "err": f"TIMEOUT after {timeout}s"}, ""
    out = cp.stdout or ""
    for line in out.splitlines():
        if line.startswith("PREFLIGHT_JSON "):
            return json.loads(line[len("PREFLIGHT_JSON "):]), (cp.stderr or "")
    return ({"ok": False, "err": f"child died rc={cp.returncode}",
             "tb": ((cp.stderr or "")[-800:])}, cp.stderr or "")


def gate(name: str, ok: bool, detail: str = "", fatal: bool = True):
    mark = "PASS" if ok else ("FAIL" if fatal else "WARN")
    print(f"  [{mark}] {name}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        (hard_fail if fatal else warn).append(name)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    zip_path = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(HERE, sys.argv[1])
    print(f"\nPREFLIGHT  {os.path.basename(zip_path)}")
    print("=" * 74)

    # ---- TEST 1: the artifact itself -----------------------------------
    print("\nTEST 1  artifact")
    if not os.path.exists(zip_path):
        gate("zip exists", False, zip_path)
        return report()
    mb = os.path.getsize(zip_path) / 1e6
    gate("zip exists", True, f"{mb:.1f} MB")
    gate(f"under {LIMIT_MB} MB limit", mb <= LIMIT_MB, f"{mb:.1f} MB")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        gate("submission.py at archive root", "submission.py" in names)
        try:
            compile(z.read("submission.py").decode("utf-8"), "submission.py", "exec")
            gate("submission.py compiles", True)
        except Exception as e:
            gate("submission.py compiles", False, str(e))

    tmp = tempfile.mkdtemp(prefix="preflight_")
    work = os.path.join(tmp, "sub")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(work)

    try:
        # ---- TEST 2/3: fresh-subprocess import and the API contract ----
        print("\nTEST 2  fresh-subprocess import and API contract")
        print("        (this is the test that submission 884075 would have failed)")
        r, _ = run_child(work, CHILD)
        gate("module imports in a clean interpreter", r.get("stage") != "import",
             r.get("err") or "")
        gate("survives synthetic warmup + real call", r.get("ok", False),
             "" if r.get("ok") else (r.get("err") or f"died at stage '{r.get('stage')}'"))
        if r.get("tb") and not r.get("ok"):
            print("        traceback tail:")
            for line in r["tb"].strip().splitlines()[-6:]:
                print("          " + line[:100])
        if r.get("ok"):
            gate("dtypes", True, " / ".join(r.get("dtypes", [])), fatal=False)
            per = r.get("log_bytes", 0)
            tot = per * N_PROC
            gate(f"stderr fits the {LOG_CAP} B harness cap",
                 tot <= LOG_CAP, f"{per} B x {N_PROC} = {tot} B", fatal=False)

        # ---- TEST 4: the import guard actually guards -------------------
        print("\nTEST 4  import guard  (break the optimised path on purpose)")
        broken = os.path.join(tmp, "broken")
        shutil.copytree(work, broken)
        f = os.path.join(broken, "submission.py")
        src = open(f, encoding="utf-8").read()
        if "def _load_at_import():" in src:
            src = src.replace("def _load_at_import():\n",
                              "def _load_at_import():\n"
                              "    raise RuntimeError('preflight injected failure')\n", 1)
            open(f, "w", encoding="utf-8").write(src)
            rb, _ = run_child(broken, CHILD)
            gate("survives a failure in the optimised load path", rb.get("ok", False),
                 rb.get("err") or "falls back and still serves predictions")
        else:
            gate("import guard present", False, "no _load_at_import() found", fatal=False)

        # ---- TEST 5: real score, and the scorer's own validation --------
        print("\nTEST 5  score on the held-out FINAL pool")
        if os.path.exists(SCORING) and os.path.exists(SPLITS):
            rs_, _ = run_child(work, SCORE_CHILD, [SCORING, SPLITS])
            if rs_.get("ok"):
                gate("scorer accepts the bounds", True,
                     f"coverage {rs_['coverage']:.4f}")
                print(f"        rel {rs_['rel']:.3f}  tke {rs_['tke']:.3f}  "
                      f"mvpe {rs_['mvpe']:.3f}  sps {rs_['sps']:.3f}")
                for k, want in BASELINE.items():
                    got = rs_[k]
                    gate(f"{k} within 0.05 of the known field baseline",
                         abs(got - want) < 0.05, f"{got:.4f} vs {want:.4f}",
                         fatal=False)
            else:
                gate("scorer accepts the bounds", False, rs_.get("err", "?"))
        else:
            gate("scoring data available", False, "skipped", fatal=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return report()


def report() -> int:
    print("\n" + "=" * 74)
    if hard_fail:
        print(f"RESULT: DO NOT SUBMIT  --  {len(hard_fail)} hard gate(s) failed")
        for x in hard_fail:
            print(f"   FAIL  {x}")
        if warn:
            print(f"   ({len(warn)} warning(s) as well)")
        return 1
    if warn:
        print(f"RESULT: SAFE TO SUBMIT, with {len(warn)} warning(s)")
        for x in warn:
            print(f"   WARN  {x}")
        print("\n   Warnings do not risk a zero. Check that any score change is intended.")
        return 0
    print("RESULT: SAFE TO SUBMIT  --  all gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
