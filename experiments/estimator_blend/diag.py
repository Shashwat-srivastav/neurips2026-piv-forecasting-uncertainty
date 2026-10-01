from base import *
X, Y, P, _ = load("select")
E, valid = estimators(X)
pr, ks = proxies("select", Y, P); print("k A/B", ks)
for name, (Yt, Pt) in pr.items():
    print("proxy", name)
    for t in (0, 1, 2, 4, 8, 12, 19):
        yt, pt = Yt[:, t], Pt[:, t]; m = (yt != 0) & valid
        row = [f"t={t:2d}"]
        for c in (0, 1):
            mc = m[..., c]
            eP = np.abs(pt - yt)[..., c][mc]; em = np.abs(E["mean"] - yt)[..., c][mc]; el = np.abs(E["last"] - yt)[..., c][mc]
            a = (pt - E["mean"])[..., c][mc]; b = (yt - E["mean"])[..., c][mc]
            row.append(f"c{c} med|e| P {np.median(eP):.4f} mean {np.median(em):.4f} last {np.median(el):.4f} rmsP {np.sqrt((eP**2).mean()):.4f} rmsM {np.sqrt((em**2).mean()):.4f} w_mean {np.dot(a,b)/np.dot(a,a):.3f}")
        print("  ", " | ".join(row))
