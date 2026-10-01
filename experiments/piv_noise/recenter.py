"""Band re-centering: bands need not contain the point prediction (scoring.aggregate_sps only checks lower<=upper).
Centre = P + w*(window_mean - P) per (input-window region, channel); lead 0 additionally blends last frame.
h = f * h_v8. w,f fitted on SELECT; evaluated on FINAL."""
import sys, numpy as np
from scipy import ndimage as ndi
sys.path.insert(0,"../bands"); import common as C
exec(open("_reg.py").read())
SIG=C.SIG; NAMES=["freestream","shear","wake","near-mask"]
def prep(role):
    X,Y,P,_=C.load(role); h=C.v8_half(X)          # (n,1,H,W,2)
    R=np.stack([reg_from_window(x) for x in X])     # (n,H,W)
    mu=X.mean(1,keepdims=True)
    return X,Y,P,np.broadcast_to(h,P.shape),np.broadcast_to(R[:,None,...,None],P.shape),np.broadcast_to(mu,P.shape)
A=prep("select"); B=prep("final")
def tsum(c,Y,h,m): e=np.abs(c-Y)[m]; hh=h[m]; return (np.exp(-2*hh/SIG)*(e<=hh)).sum()
WS=np.linspace(0,1,11); FS=np.geomspace(0.5,1.6,23)
num_old=num_new=den=0
for r in range(-1,4):
    for ch in (0,1):
        X,Y,P,h,R,mu=A
        m=(R[...,ch]==r)&(Y[...,ch]!=0)
        if r==-1: w,f=0.0,1.0
        else:
            # subsample for speed
            idx=np.flatnonzero(m.ravel()); idx=np.random.default_rng(0).choice(idx,min(len(idx),400000),replace=False)
            p=P[...,ch].ravel()[idx]; y=Y[...,ch].ravel()[idx]; mm=mu[...,ch].ravel()[idx]; hh=h[...,ch].ravel()[idx]
            best=(-1,0,1)
            for w in WS:
                e=np.abs(p+w*(mm-p)-y)
                for f in FS:
                    t=(np.exp(-2*f*hh/SIG)*(e<=f*hh)).mean()
                    if t>best[0]: best=(t,w,f)
            _,w,f=best
        X,Y,P,h,R,mu=B
        m=(R[...,ch]==r)&(Y[...,ch]!=0)
        old=tsum(P[...,ch],Y[...,ch],h[...,ch],m); new=tsum(P[...,ch]+w*(mu[...,ch]-P[...,ch]),Y[...,ch],f*h[...,ch],m)
        n=m.sum(); num_old+=old; num_new+=new; den+=n
        if r>=0: print(f"{NAMES[r]:10} {'uv'[ch]}: w_mean {w:.1f} h-scale {f:.2f}  FINAL T {old/n:.4f} -> {new/n:.4f}  (share {n/ (B[1]!=0).sum()*1:.3f})")
print(f"TOTAL FINAL T (both channels): v8 {num_old/den:.4f} -> recentred {num_new/den:.4f}  ({100*(num_new/num_old-1):+.2f}%)")
