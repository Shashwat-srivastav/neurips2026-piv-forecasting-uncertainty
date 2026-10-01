"""Paste this into a Kaggle notebook cell and run it BEFORE the training cell.
Confirms every file finetune_cno_kaggle.py needs is actually present under
/kaggle/input, with sizes, so a bad/incomplete upload fails loudly here
instead of silently inside the training run.

No arguments -- just run the cell.
"""
import glob
import os

REQUIRED = {
    "sim_cno.pth":              (25_000_000, 40_000_000),   # ~32 MB expected
    "finetune_fno_kaggle.py":   (10_000, 60_000),
    "finetune_cno_kaggle.py":   (2_000, 20_000),
    "train_pool_v12_3.json":    (500, 5_000),
    "load_baseline.py":         (2_000, 20_000),
    "kaggle_train_real.npz":    (100_000_000, None),         # the training data
}

print(f"{'file':28} {'found':6} {'count':6} {'size':>12}  path")
print("-" * 90)
all_ok = True
for fname, (lo, hi) in REQUIRED.items():
    hits = glob.glob(f"/kaggle/input/**/{fname}", recursive=True)
    if not hits:
        print(f"{fname:28} {'NO':6} {0:6} {'':>12}  ** MISSING **")
        all_ok = False
        continue
    ok_this = True
    for h in hits:
        size = os.path.getsize(h)
        size_ok = size >= lo and (hi is None or size <= hi)
        flag = "" if size_ok else "  ** SIZE LOOKS WRONG **"
        print(f"{fname:28} {'yes':6} {len(hits):6} {size:12,}  {h}{flag}")
        ok_this &= size_ok
    if len(hits) > 1:
        print(f"{'':28} {'':6} {'':6} {'':>12}  ** {len(hits)} copies found -- "
              f"make sure the training cell picks the right one **")
    all_ok &= ok_this

# also check rpde_baselines/ (a directory, not a single file) is somewhere
rb = glob.glob("/kaggle/input/**/rpde_baselines", recursive=True)
if rb:
    n = sum(len(files) for _, _, files in os.walk(rb[0]))
    print(f"{'rpde_baselines/':28} {'yes':6} {len(rb):6} {n:>9} files  {rb[0]}")
else:
    print(f"{'rpde_baselines/':28} {'NO':6} {0:6} {'':>12}  ** MISSING **")
    all_ok = False

print("-" * 90)
print("ALL FILES PRESENT AND SANE" if all_ok else
      "SOMETHING IS MISSING OR WRONG -- fix before running the training cell")
