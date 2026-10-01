"""Fill cache_all/ from forecasts we already computed (SELECT/FINAL caches, K9 random-split caches) before any new
inference. Window counts must match loo_lib.windows(name, 20); a trajectory that does not match is left for cache_all.py.
"""
import json, os
import numpy as np
from loo_lib import OUT, names, windows

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
usable = set(names()); done = 0


def put(nm, P):
    global done
    path = os.path.join(OUT, nm + ".npy")
    if nm not in usable or os.path.exists(path):
        return
    n = len(windows(nm, 20)[0])
    if len(P) != n:
        print("window count mismatch, skipped", nm, len(P), n); return
    np.save(path, np.ascontiguousarray(P, dtype=np.float32)); done += 1


for role in ("select", "final"):
    d = np.load(os.path.join(HERE, "..", "interval_recipes", f"cache_{role}_s20.npz"))
    traj, P = d["traj"], d["P"]
    for nm in dict.fromkeys(traj.tolist()):
        put(nm, P[traj == nm])
    del d, P
K9 = json.load(open(os.path.join(HERE, "k9_multi.json")))
for s, r in K9.items():
    f = os.path.join(HERE, "cache_rand_s20.npz" if s == "0" else f"cache_rand{s}_s20.npz")
    if not os.path.exists(f):
        continue
    P = np.load(f)["P"]; i = 0
    for nm in r["pick"]:
        n = len(windows(nm, 20)[0]); put(nm, P[i:i + n]); i += n
    assert i == len(P), (s, i, len(P))
print("reused", done, "trajectories; cache_all now holds", len(os.listdir(OUT)))
