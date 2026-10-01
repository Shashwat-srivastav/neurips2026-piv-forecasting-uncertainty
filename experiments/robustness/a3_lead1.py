"""A3: lead-1 diagnosis on FINAL with real Y."""
from lib2 import *
X, Y, P, traj = load("final")
last = X[:, -1]
def rel(a, y):  # official per-sample rel L2 on a single frame, mean over windows
    return float(sc.rel_l2_per_sample(a[:, None], y[:, None], 2).mean())
for t in (0, 1, 2, 4, 9, 19):
    y = Y[:, t]; m = (y != 0) & (last != 0)
    di = (P[:, t] - last)[m]; dt = (y - last)[m]
    r = np.corrcoef(di, dt)[0, 1]; amp = di.std() / dt.std(); slope = (di * dt).sum() / (di * di).sum()
    s = f"lead {t+1:2}: rel P~Y[t] {rel(P[:, t], y):.4f}  last~Y[t] {rel(last, y):.4f}"
    if t + 1 < 20: s += f"  P[t]~Y[t+1] {rel(P[:, t], Y[:, t+1]):.4f}"
    if t >= 1: s += f"  P[t]~Y[t-1] {rel(P[:, t], Y[:, t-1]):.4f}"
    s += f" | incr corr {r:.3f} amp(std pred/true) {amp:.3f} LS slope true-on-pred {slope:.3f}"
    for c in (0, 1):
        mc = m[..., c]; a_ = (P[:, t, ..., c] - last[..., c])[mc]; b_ = (y[..., c] - last[..., c])[mc]
        s += f" | ch{c} r {np.corrcoef(a_, b_)[0,1]:.3f} amp {a_.std()/b_.std():.3f}"
    print(s, flush=True)
# where on the field: sextile (per-window sigma_local rank) breakdown at lead 1 and 2
E, valid = estimators(X); sb = rank_bins1(sigma_local(X).astype(np.float32), valid, 6)
for t in (0, 1):
    y = Y[:, t]
    for c in (0,):
        row = []
        for k in range(6):
            m = (y[..., c] != 0) & valid[..., c] & (sb[..., c] == k)
            eP = ((P[:, t, ..., c] - y[..., c])[m] ** 2).mean(); eL = ((last[..., c] - y[..., c])[m] ** 2).mean()
            eM = ((E["mean"][..., c] - y[..., c])[m] ** 2).mean()
            a_ = (P[:, t, ..., c] - last[..., c])[m]; b_ = (y[..., c] - last[..., c])[m]
            row.append(f"s{k+1}: mse P/last {eP/eL:.2f} P/mean {eP/eM:.2f} r {np.corrcoef(a_, b_)[0,1]:.2f}")
        print(f"lead {t+1} u by sextile:", " | ".join(row), flush=True)
