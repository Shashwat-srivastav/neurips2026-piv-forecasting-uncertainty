"""Which input-window features predict per-element |FNO error|? Uses experiments/interval_recipes cache (read-only):
v8 shipped sim_real_fno forecasts on the 14 holdout trajectories (select 4 + final 10, stride 20).
NOTE these trajectories were in sim_real_fno's training set -> magnitudes optimistic, structure informative."""
import numpy as np, sys
from scipy import ndimage as ndi
from scipy.stats import spearmanr, rankdata
def load(role):
    d=np.load(f"../bands/cache_{role}_s20.npz"); return d["X"],d["Y"],d["P"],d["traj"]
def feats(X,P):
    n=len(X); F={}
    Z=(X==0)                                   # (n,20,H,W,2)
    k=Z.sum(1).astype(np.float32)              # zero count per pixel
    Xm=np.where(Z,np.nan,X)
    mu=np.nanmean(Xm,1); sd=np.nanstd(Xm,1)
    mu=np.nan_to_num(mu); sd=np.nan_to_num(sd)
    F["sigma_local"]=sd
    F["sigma_sm3"]=ndi.uniform_filter(sd,size=(1,3,3,1))
    F["sigma_max5"]=ndi.maximum_filter(sd,size=(1,5,5,1))
    F["sigma_uvsum"]=np.repeat((sd[...,:1]+sd[...,1:]),2,-1)   # both channels' std
    F["sigma_last5"]=np.nan_to_num(np.nanstd(Xm[:,-5:],1))
    F["absdev_last"]=np.abs(X[:,-1]-mu)
    F["rate_last"]=np.abs(X[:,-1]-X[:,-2])
    d2=X[:,2:]-2*X[:,1:-1]+X[:,:-2]; F["hf_d2"]=np.sqrt((d2**2).mean(1)/6)
    g=np.zeros_like(mu)
    for c in (0,1):
        gy,gx=np.gradient(mu[...,c],axis=(1,2)); g[...,c]=np.hypot(gx,gy)
    F["grad_mean"]=np.repeat(g[...,:1],2,-1)
    ever=(Z.any(1)).any(-1)
    dist=np.stack([ndi.distance_transform_edt(~e) for e in ever]); F["dist_mask"]=-np.repeat(dist[...,None],2,-1)
    F["zero_count"]=k
    Umu=np.array([np.percentile(mu[i,...,0][~ever[i]],95) for i in range(n)])
    F["deficit"]=np.repeat((1-mu[...,:1]/Umu[:,None,None,None]),2,-1)
    F["win_sigma"]=np.broadcast_to(sd.mean((1,2),keepdims=True),sd.shape)
    # prediction-based (lead dependent)
    Fl={}
    Fl["absP_minus_last"]=np.abs(P-X[:,-1:]); Fl["absP_minus_mean"]=np.abs(P-mu[:,None])
    Fl["absP"]=np.abs(P)
    return F,Fl,k
role=sys.argv[1] if len(sys.argv)>1 else "final"
X,Y,P,T=load(role); E=np.abs(P-Y); S=(Y!=0)
F,Fl,k=feats(X,P)
rng=np.random.default_rng(0)
idx=np.flatnonzero(S.ravel()); idx=rng.choice(idx,2_000_000,replace=False)
e=E.ravel()[idx]
def bc(f): return np.broadcast_to(f[:,None],E.shape).ravel()[idx] if f.ndim==4 else f.ravel()[idx]
lead=np.broadcast_to(np.arange(20)[None,:,None,None,None],E.shape).ravel()[idx]
ch=np.broadcast_to(np.arange(2),E.shape).ravel()[idx]
win=np.broadcast_to(np.arange(len(X))[:,None,None,None,None],E.shape).ravel()[idx]
print(f"{role}: {len(X)} windows, scored frac {S.mean():.4f}, |e| median {np.median(e):.4f} mean {e.mean():.4f}")
def within(f):
    # mean over windows of within-window Spearman (what a per-call rank rule sees)
    rs=[]
    for w in range(0,len(X),7):
        m=win==w
        if m.sum()>500: rs.append(spearmanr(f[m],e[m])[0])
    return np.nanmean(rs)
res=[]
for nm,f in list(F.items())+list(Fl.items()):
    fv=bc(f).astype(np.float64)
    r=spearmanr(fv,e)[0]; p=np.corrcoef(fv,e)[0,1]; rw=within(fv)
    ru=spearmanr(fv[ch==0],e[ch==0])[0]; rv=spearmanr(fv[ch==1],e[ch==1])[0]
    res.append((nm,r,p,rw,ru,rv))
res.sort(key=lambda t:-abs(t[3]))
print(f"{'feature':18} {'Spear':>7} {'Pears':>7} {'withinWin':>9} {'u':>7} {'v':>7}")
for t in res: print(f"{t[0]:18} {t[1]:7.3f} {t[2]:7.3f} {t[3]:9.3f} {t[4]:7.3f} {t[5]:7.3f}")
# ratio |e|/sigma by lead
sl=bc(F["sigma_local"])
print("median |e| / sigma_local by lead:", " ".join(f"{np.median(e[lead==L]/(sl[lead==L]+1e-4)):.2f}" for L in range(20)))
print("Spearman(sigma_local,|e|) by lead:", " ".join(f"{spearmanr(sl[lead==L],e[lead==L])[0]:.2f}" for L in range(0,20,3)))
np.savez("resid_sample_"+role+".npz", e=e, lead=lead, ch=ch, win=win, **{nm:bc(f).astype(np.float32) for nm,f in list(F.items())+list(Fl.items())})
