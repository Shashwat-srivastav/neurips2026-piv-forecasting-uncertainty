#!/usr/bin/env python3
"""Package a CNO submission and verify the payload is actually in it.

Same "verify the payload is IN the artefact" discipline as build_fno_zip.py,
parameterized for CNO's checkpoint (~32 MB, vs FNO's ~201 MB fp16 pack -- no
fp16 packing needed for CNO at all, it is already well under the 256 MB cap
at full fp32).

Usage: python build_cno_zip.py <stage_dir> <out.zip> [ckpt_filename]
"""
from __future__ import annotations

import os
import sys
import zipfile

STAGE = sys.argv[1]
OUT = sys.argv[2]
CKPT = sys.argv[3] if len(sys.argv) > 3 else "sim_cno.pth"
MIN_CKPT = 25_000_000   # CNO checkpoints are ~32 MB fp32; catch a truncated/empty file

if os.path.exists(OUT):
    os.remove(OUT)

n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for root, dirs, files in os.walk(STAGE):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".pyc"):
                continue
            p = os.path.join(root, f)
            arc = os.path.relpath(p, STAGE).replace(os.sep, "/")
            z.write(p, arc)
            n += 1

size = os.path.getsize(OUT)
print(f"entries {n}   zip {size:,} bytes ({size/1e6:.1f} MB)   limit 256 MB")

z = zipfile.ZipFile(OUT)
names = z.namelist()
info = {i.filename: i.file_size for i in z.infolist()}

ok = True
for need in ("submission.py", "load_baseline.py", "_vendor_requests_stub.py", CKPT):
    present = need in names
    print(f"  {need:26} present={present}  size={info.get(need, 0):,}")
    ok &= present
for pkg in ("rpde_baselines/", "_vendor/"):
    c = sum(1 for x in names if x.startswith(pkg))
    print(f"  {pkg:26} files={c}")
    ok &= c > 0

ck = info.get(CKPT, 0)
if ck < MIN_CKPT:
    print(f"  FAIL: checkpoint is {ck:,} bytes, expected >= {MIN_CKPT:,}")
    ok = False
if size > 256 * 1024 * 1024:
    print(f"  FAIL: zip exceeds the 256 MB limit")
    ok = False

print("PAYLOAD VERIFIED" if ok else "PAYLOAD BROKEN")
sys.exit(0 if ok else 1)
