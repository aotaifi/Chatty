import numpy as np, math
P="/Users/aliotaifi/j1j2_vit_bench/global_field_cache.npz"
g=np.load(P); L=6; J2=.5; N=36
A=np.array([x+L*y for y in range(L) for x in range(L) if (x+y)%2==0])

NN=set(); NNN=set()
q=lambda x,y:(x%L)+L*(y%L)
for y in range(L):
    for x in range(L):
        i=q(x,y)
        NN.add(tuple(sorted((i,q(x+1,y))))); NN.add(tuple(sorted((i,q(x,y+1)))))
        NNN.add(tuple(sorted((i,q(x+1,y+1))))); NNN.add(tuple(sorted((i,q(x+1,y-1)))))

def oracle(score,m,h,w):
    score=np.asarray(score,float); w=np.asarray(w,float); w=w/w.sum()
    z=w*m*h; order=np.argsort(score); base=-float(z.sum()); pref=0.0
    best=(base,float(score[order[0]]-1),1.0)
    for k,ix in enumerate(order):
        pref += 2*float(z[ix]); c=base+pref
        t=float(score[ix]) if k==len(order)-1 else float(.5*(score[ix]+score[order[k+1]]))
        fm=float(w[score>t].sum())
        if c>best[0]: best=(c,t,fm)
    return best
def analyze(pre):
    X=g["X"+pre]; Xn=g["Xn"+pre]; C=g["C"+pre]; M=g["M"+pre]; MP=g["MP"+pre]
    w=g["iw"+pre].astype(float); w/=w.sum(); zx=g["zx"+pre]
    m=np.where((X[:,A]<0).sum(1)%2==0,1.,-1.)
    phi=.5*np.angle(np.sum(w*np.exp(2j*zx.imag)))
    h=np.where(np.cos(zx.imag-phi)>=0,1.,-1.)
    D1=np.zeros(len(X)); D2=np.zeros(len(X))
    for i,j in NN: D1 += .25*X[:,i]*X[:,j]
    for i,j in NNN: D2 += .25*J2*X[:,i]*X[:,j]
    mask1=np.zeros(M.shape,bool); mask2=np.zeros(M.shape,bool)
    bad=0
    for i in range(len(X)):
        for k in np.flatnonzero(M[i]>0):
            d=np.flatnonzero(X[i]!=Xn[i,k])
            if len(d)!=2: bad+=1; continue
            p=tuple(sorted((int(d[0]),int(d[1]))))
            if p in NN: mask1[i,k]=1
            elif p in NNN: mask2[i,k]=1
            else: bad+=1
    O1=np.sum(C*MP*mask1,axis=1); O2=np.sum(C*MP*mask2,axis=1)
    eps=1e-12
    ratio=O2/(-O1+eps)
    normcontrast=(O1+O2)/(np.abs(O1)+np.abs(O2)+eps)
    scores={"D1":D1,"D2":D2,"O1":O1,"O2":O2,
            "J1block":D1+O1,"J2block":D2+O2,
            "ratio_O2_over_minusO1":ratio,"normcontrast":normcontrast,
            "diag":D1+D2,"offdiag":O1+O2,"full":D1+D2+O1+O2}
    print(pre,"baseline",float(np.sum(w*m*h)),"bad_edges",bad,
          "counts",int(mask1.sum()),int(mask2.sum()),flush=True)
    for name,s in scores.items():
        a=oracle(s,m,h,w); b=oracle(-s,m,h,w)
        best=a if a[0]>=b[0] else (b[0],-b[1],b[2])
        print(pre,name,"oracleC",best[0],"t",best[1],"flipmass",best[2],
              "gain",best[0]-float(np.sum(w*m*h)),flush=True)
    sweep=[]
    for lam in [0,.25,.5,.75,1.,1.25,1.5,2.,3.,4.]:
        s=O1+lam*O2
        a=oracle(s,m,h,w); b=oracle(-s,m,h,w)
        best=a if a[0]>=b[0] else (b[0],-b[1],b[2])
        sweep.append((lam,best[0],best[2]))
    print(pre,"lambda_sweep",sweep,flush=True)
    return scores,m,h,w

analyze("tr"); analyze("va")
