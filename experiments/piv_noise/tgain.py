"""Bin-optimal-h T (fit on SELECT sample, eval on FINAL sample) for candidate difficulty scores."""
import numpy as np
SIG=0.0563870259
A=np.load("resid_sample_select.npz"); B=np.load("resid_sample_final.npz")
use=["sigma_sm3","hf_d2","sigma_max5","absP_minus_last","rate_last","grad_mean"]
def design(D,m,use):
    return np.column_stack([np.log(np.abs(D[n][m].astype(float))+1e-4) for n in use]+[D["lead"][m]/19.0,np.ones(m.sum())])
H=np.geomspace(2e-4,0.2,300)
def best_h(e):
    t=np.exp(-2*H/SIG)[None,:]*(e[:,None]<=H[None,:]); return H[t.mean(0).argmax()]
def T_of(sA,sB,nb=20):
    tot=0; nB=0
    for c in (0,1):
        ma=A["ch"]==c; mb=B["ch"]==c
        edges=np.quantile(sA[ma],np.linspace(0,1,nb+1)[1:-1])
        ba=np.digitize(sA[ma],edges); bb=np.digitize(sB[mb],edges)
        eA=A["e"][ma]; eB=B["e"][mb]
        for b in range(nb):
            h=best_h(eA[ba==b][:20000]); x=eB[bb==b]
            tot+=(np.exp(-2*h/SIG)*(x<=h)).sum(); nB+=len(x)
    return tot/nB
def combo(use):
    sA=np.zeros(len(A["e"])); sB=np.zeros(len(B["e"]))
    for c in (0,1):
        ma=A["ch"]==c; mb=B["ch"]==c
        beta,*_=np.linalg.lstsq(design(A,ma,use),np.log(A["e"][ma]+1e-5),rcond=None)
        sA[ma]=design(A,ma,use)@beta; sB[mb]=design(B,mb,use)@beta
    return sA,sB
print("flat per-channel h:", round(T_of(np.zeros(len(A['e'])),np.zeros(len(B['e'])),1),4))
for n in ["sigma_local","sigma_sm3","hf_d2","sigma_max5","absP_minus_last"]:
    print(f"{n:18}", round(T_of(A[n],B[n]),4))
for u in (["sigma_sm3","hf_d2"],["hf_d2","absP_minus_last"],["sigma_sm3","hf_d2","absP_minus_last"],use):
    sA,sB=combo(u); print("combo",u, round(T_of(sA,sB),4))
e=B["e"]; print("oracle h=|e| T:", round(np.exp(-2*e/SIG).mean(),4))
