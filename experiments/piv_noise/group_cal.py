"""Does a PIV-physics grouping (mask-edge distance, dropout count) add T on top of the v8 band? Fit on SELECT, eval FINAL.
Also: persistence vs FNO at the first leads."""
import sys, numpy as np
from scipy import ndimage as ndi
sys.path.insert(0,"../bands"); import common as C   # read-only import
SIG=C.SIG
def prep(role):
    X,Y,P,_=C.load(role); h=C.v8_half(X)             # (n,1,H,W,2)
    Z=(X==0); k=Z.sum(1); ever=Z.any(1).any(-1)
    dist=np.stack([ndi.distance_transform_edt(~e) for e in ever])[...,None]
    g=np.where(k==20,5,np.where(k>0,4,np.where(dist<=1,3,np.where(dist<=3,2,1))))  # 1 interior,2 near,3 edge,4 intermittent,5 dead
    return X,Y,P,h,np.broadcast_to(g,k.shape)
def T(P,Y,h):
    S=Y!=0; e=np.abs(P-Y); return float((np.exp(-2*h/SIG)*(e<=h))[S].sum()/S.sum())
XA,YA,PA,hA,gA=prep("select"); XB,YB,PB,hB,gB=prep("final")
print("v8 band T: select %.4f final %.4f"%(T(PA,YA,hA),T(PB,YB,hB)))
# per-group multiplicative factor on v8 h, per channel, fit on select
M=np.geomspace(0.3,3,41); fac=np.ones((6,2))
for c in (0,1):
    for g in range(1,6):
        m=np.broadcast_to((gA[...,c]==g)[:,None],PA[...,c].shape)&(YA[...,c]!=0)
        if m.sum()<1000: continue
        e=np.abs(PA[...,c]-YA[...,c])[m]; h=np.broadcast_to(hA[...,c],PA[...,c].shape)[m]
        t=[(np.exp(-2*f*h/SIG)*(e<=f*h)).mean() for f in M]; fac[g,c]=M[int(np.argmax(t))]
        mb=np.broadcast_to((gB[...,c]==g)[:,None],PB[...,c].shape)&(YB[...,c]!=0)
        eb=np.abs(PB[...,c]-YB[...,c])[mb]; hb=np.broadcast_to(hB[...,c],PB[...,c].shape)[mb]
        print(f" ch{c} group{g} share-of-scored {mb.sum()/(YB[...,c]!=0).sum():.4f} fac {fac[g,c]:.2f}  T_final v8 {(np.exp(-2*hb/SIG)*(eb<=hb)).mean():.4f} -> {(np.exp(-2*fac[g,c]*hb/SIG)*(eb<=fac[g,c]*hb)).mean():.4f}")
F=np.stack([fac[gB[...,c],c] for c in (0,1)],-1)[:,None]
print("grouped v8 T final: %.4f"%T(PB,YB,hB*F))
# persistence / blend at first leads (fit w on select)
for L in (0,1,2):
    best=None
    for w in np.linspace(0,1,21):
        PA2=PA.copy(); PA2[:,L]=w*XA[:,-1]+(1-w)*PA[:,L]
        r=np.abs(PA2[:,L]-YA[:,L])[YA[:,L]!=0].mean()
        if best is None or r<best[1]: best=(w,r)
    w=best[0]; eF=np.abs(PB[:,L]-YB[:,L])[YB[:,L]!=0].mean(); PB2=w*XB[:,-1]+(1-w)*PB[:,L]; eB=np.abs(PB2-YB[:,L])[YB[:,L]!=0].mean()
    print(f"lead {L}: persistence weight {w:.2f} (fit select); final mean|e| FNO {eF:.5f} -> blend {eB:.5f} ({100*(eB/eF-1):+.1f}%)")
