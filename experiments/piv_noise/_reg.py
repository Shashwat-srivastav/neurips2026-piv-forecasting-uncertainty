def reg_from_window(Xw):
    u=Xw[...,0]; ever=(Xw==0).any(0).any(-1); mu=u.mean(0)
    Uinf=np.percentile(mu[~ever],95); near=ndi.binary_dilation(ever,iterations=2)&~ever
    gy,gx=np.gradient(ndi.uniform_filter(np.where(ever,Uinf,mu),3)); g=np.hypot(gx,gy); gthr=np.percentile(g[~ever],85)
    R=np.full(mu.shape,2); R[mu>=0.9*Uinf]=0; R[(g>=gthr)&(mu<0.97*Uinf)]=1; R[near]=3; R[ever]=-1
    return R
