"""Shared helpers for the leave-one-AoA-out / leave-one-Re-band-out analysis."""
import os
import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("RPDE_TRAIN_REAL", "data/train_real")
OUT = os.path.join(HERE, "cache_all")
RE_BANDS = {"Re<10k": (0, 10000), "Re 10-20k": (10000, 20000), "Re>20k": (20000, 10 ** 9)}


def names():
    return sorted(f[:-3] for f in os.listdir(ROOT) if f.endswith(".h5") and f[:-3] != "7575_0")


aoa = lambda n: int(n.split("_")[1])   # noqa: E731
re_ = lambda n: int(n.split("_")[0])   # noqa: E731


def windows(nm, stride, channels=2):
    """(X, Y) windows of one trajectory, same construction as cache_rand.py (stride-20 grid, then every stride/20-th)."""
    with h5py.File(os.path.join(ROOT, nm + ".h5")) as f:
        u = np.asarray(f["u"][:, ::2, ::2], np.float32); v = np.asarray(f["v"][:, ::2, ::2], np.float32)
    starts = list(range(0, len(u) - 40 + 1, 20))[::stride // 20]
    X = np.zeros((len(starts), 20, 32, 64, channels), np.float32); Y = np.zeros_like(X)
    for j, i in enumerate(starts):
        X[j, ..., 0], X[j, ..., 1] = u[i:i+20], v[i:i+20]
        Y[j, ..., 0], Y[j, ..., 1] = u[i+20:i+40], v[i+20:i+40]
    return X, Y


def load_set(traj, stride):
    """Stacked X, Y, P (2 channels) for a list of trajectories; P from cache_all (stride-20 grid)."""
    Xs, Ys, Ps, T = [], [], [], []
    for nm in traj:
        X, Y = windows(nm, stride)
        P = np.load(os.path.join(OUT, nm + ".npy"))[:: stride // 20]
        assert len(P) == len(X), (nm, len(P), len(X))
        Xs.append(X); Ys.append(Y); Ps.append(P); T += [nm] * len(X)
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ps), np.array(T)
