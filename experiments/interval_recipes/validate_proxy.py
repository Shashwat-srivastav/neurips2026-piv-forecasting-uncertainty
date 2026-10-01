from common import *
for role in ("select", "final"):
    X, Y, P, _ = load(role)
    print(f"== {role}: n={len(X)} raw rel={rel_err(P,Y):.4f} tke={sc.tke_rel_l2_per_sample(P,Y,2).mean():.4f} mvpe={sc.mvpe_rel_l2_per_sample(P,Y).mean():.4f}")
    h8 = v8_half(X); h8n = v8_half(X, shape=False)
    s, T, cov, W = official(P, Y, h8); print(f"  k=1 (raw)      v8: sps {s:.2f} T {T:.4f} cov {cov:.3f} W {W:.4f}")
    for m in ("rel", "tke", "mvpe"):
        k = fit_k(Y, P, m)
        Yp = proxy_target(Y, P, k)
        errs = (rel_err(P,Yp), sc.tke_rel_l2_per_sample(P,Yp,2).mean(), sc.mvpe_rel_l2_per_sample(P,Yp).mean())
        s, T, cov, W = official(P, Yp, h8)
        s0, T0, cov0, _ = official(P, Yp, h8n)
        print(f"  k={k:.3f} ({m:4}) errs rel/tke/mvpe={errs[0]:.4f}/{errs[1]:.4f}/{errs[2]:.4f}  v8: sps {s:.2f} T {T:.4f} cov {cov:.3f} W {W:.4f} | noshape T {T0:.4f} sps {s0:.2f}")
print("LIVE targets: rel", LIVE)
