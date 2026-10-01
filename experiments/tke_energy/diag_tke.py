"""TKE error diagnosis for stock sim_real_fno on the cached holdout windows.
Uses experiments/interval_recipes caches (X,Y,P from the shipped v8 fp16 FNO, stride 20). Read-only on them.
Holdout was in sim_real_fno's training set -> only RELATIVE effects are meaningful.
"""
import os, sys, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
KIT = os.path.join(BASE, "realpde_t1_starting_kit_v9", "realpde_t1_starting_kit_v9")
spec = importlib.util.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)


def load(role):
    d = np.load(os.path.join(BASE, "lab", "bands", f"cache_{role}_s20.npz"))
    return d["X"], d["Y"], d["P"], d["traj"]


def ke(a):  # (N,T,H,W,2) -> (N,H,W)
    f = a - a.mean(1, keepdims=True)
    return 0.5 * (f ** 2).mean(1).sum(-1)


def scores(P, Y):
    r = sc.score_error(float(sc.rel_l2_per_sample(P, Y, 2).mean()))
    t = sc.score_error(float(sc.tke_rel_l2_per_sample(P, Y, 2).mean()))
    m = sc.score_error(float(sc.mvpe_rel_l2_per_sample(P, Y).mean()))
    return r, t, m


def report(role):
    X, Y, P, tr = load(role)
    N = len(Y)
    print(f"\n===== {role}: {N} windows, trajs {sorted(set(tr))}")
    print("official rel/tke/mvpe: %.3f %.3f %.3f" % scores(P, Y))
    Kp, Kt, Kx = ke(P), ke(Y), ke(X)
    # masking: pixels where Y==0 in some frames
    ym = (Y[..., 0] == 0)
    partial = ym.any(1) & ~ym.all(1)
    full = ym.all(1)
    print(f"target zero pixels: partial-in-window {partial.mean()*100:.2f}%  all-frames {full.mean()*100:.2f}%")
    e_all = np.linalg.norm((Kp - Kt).reshape(N, -1), axis=1) / np.linalg.norm(Kt.reshape(N, -1), axis=1)
    Ktc = np.where(partial, 0, Kt); Kpc = np.where(partial, 0, Kp)
    e_clean = np.linalg.norm((Kpc - Ktc).reshape(N, -1), axis=1) / np.linalg.norm(Ktc.reshape(N, -1), axis=1)
    share = (((Kp - Kt) ** 2) * partial).reshape(N, -1).sum(1) / ((Kp - Kt) ** 2).reshape(N, -1).sum(1)
    print(f"tke err mean {e_all.mean():.4f} (score {sc.score_error(e_all.mean()):.2f}); "
          f"excluding partially-masked pixels {e_clean.mean():.4f}; share of sq-err on partial-mask px {share.mean()*100:.1f}%")
    # amplitude vs shape decomposition, per window
    kp, kt = Kp.reshape(N, -1), Kt.reshape(N, -1)
    ratio = kp.sum(1) / kt.sum(1)
    s_opt = (kp * kt).sum(1) / (kp * kp).sum(1)
    e_opt = np.linalg.norm(s_opt[:, None] * kp - kt, axis=1) / np.linalg.norm(kt, axis=1)
    corr = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(kp, kt)])
    print(f"energy ratio sum(Kp)/sum(Kt): median {np.median(ratio):.3f} mean {ratio.mean():.3f} "
          f"[p10 {np.percentile(ratio,10):.3f}, p90 {np.percentile(ratio,90):.3f}]")
    print(f"per-window optimal scalar on Kp: median {np.median(s_opt):.3f}; tke err at oracle scalar {e_opt.mean():.4f} "
          f"(score {sc.score_error(e_opt.mean()):.2f}) vs {e_all.mean():.4f}")
    print(f"spatial corr(Kp,Kt) median {np.median(corr):.3f}; corr(Kx,Kt) median "
          f"{np.median([np.corrcoef(a,b)[0,1] for a,b in zip(Kx.reshape(N,-1),kt)]):.3f}")
    ex = np.linalg.norm(Kx.reshape(N, -1) - kt, axis=1) / np.linalg.norm(kt, axis=1)
    print(f"persistence-of-input-variance tke err {ex.mean():.4f} (score {sc.score_error(ex.mean()):.2f}); "
          f"sum(Kx)/sum(Kt) median {np.median(Kx.reshape(N,-1).sum(1)/kt.sum(1)):.3f}")
    # temporal: fluctuation energy by lead time, about each series' own window mean
    fp = P - P.mean(1, keepdims=True); fy = Y - Y.mean(1, keepdims=True)
    Ep = (fp ** 2).sum((2, 3, 4)).mean(0); Ey = (fy ** 2).sum((2, 3, 4)).mean(0)
    print("fluct energy ratio pred/target by lead (about own window mean):")
    print("  " + " ".join(f"{v:.2f}" for v in Ep / Ey))
    # frame-to-frame increment energy (independent of the window mean)
    dp = np.diff(P, axis=1); dy = np.diff(Y, axis=1)
    print("increment |dP|^2/|dY|^2 by lead:")
    print("  " + " ".join(f"{v:.2f}" for v in (dp ** 2).sum((2, 3, 4)).mean(0) / (dy ** 2).sum((2, 3, 4)).mean(0)))
    # least-squares gain of fluctuation per lead: <fp,fy>/<fp,fp>
    g = (fp * fy).sum((0, 2, 3, 4)) / (fp * fp).sum((0, 2, 3, 4))
    print("rel-optimal fluct gain per lead <fp,fy>/<fp,fp>:")
    print("  " + " ".join(f"{v:.2f}" for v in g))
    # region: streamwise thirds x cross-stream (centre band vs edges)
    H, W = Kt.shape[1:]
    print("region energy ratio sum(Kp)/sum(Kt) (rows=y bands top/mid/bot, cols=x thirds):")
    for yb in np.array_split(np.arange(H), 3):
        row = []
        for xb in np.array_split(np.arange(W), 3):
            row.append(Kp[:, yb][:, :, xb].sum() / Kt[:, yb][:, :, xb].sum())
        print("  " + " ".join(f"{v:.2f}" for v in row))
    # where is the target energy concentrated
    print("share of target KE per region:")
    for yb in np.array_split(np.arange(H), 3):
        print("  " + " ".join(f"{Kt[:, yb][:, :, xb].sum()/Kt.sum():.2f}" for xb in np.array_split(np.arange(W), 3)))
    # norms: how big is the fluctuation vs the full field (rel denominator)
    print(f"||fluct_Y||/||Y|| = {np.sqrt((fy**2).sum()/(Y**2).sum()):.3f}; ||fluct_P||/||Y|| = {np.sqrt((fp**2).sum()/(Y**2).sum()):.3f}; "
          f"||P-Y||/||Y|| = {np.sqrt(((P-Y)**2).sum()/(Y**2).sum()):.4f}")
    # decompose rel error into window-mean part and fluct part
    em = ((P.mean(1) - Y.mean(1)) ** 2).sum() * 20; ef = ((fp - fy) ** 2).sum()
    print(f"rel sq-error split: temporal-mean part {em/(em+ef)*100:.1f}%, fluctuation part {ef/(em+ef)*100:.1f}%")
    # dominant period of input window (per window FFT of spatially-averaged fluct v)
    return


if __name__ == "__main__":
    for r in sys.argv[1:] or ["select", "final"]:
        report(r)
