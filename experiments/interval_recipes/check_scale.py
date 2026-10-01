"""Differential validation: live v6 (band x0.85) cost -0.990 sps vs v4 (T 0.4768 -> 0.4623, -3.0% T). What does the proxy say?"""
import numpy as np
from common import SIG
import os
D = "dump"
for tag, k in (("rel", 1.6645845770829055), ("T", 1.447430551051184)):
    row = []
    for s in (0.85, 1.0, 1.15, 1.3, 1.5):
        num = n = 0
        for c in (0, 1):
            z = np.load(os.path.join(D, f"final_{c}.npz")); a = z["ae"] * k; h = z["h8"] * s
            num += np.sum(np.exp(-2 * h / SIG) * (a <= h)); n += len(a)
        row.append(f"x{s}: T {num/n:.4f}")
    print(tag, " | ".join(row))
