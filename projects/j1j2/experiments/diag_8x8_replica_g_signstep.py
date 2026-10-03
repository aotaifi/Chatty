import math, numpy as np
import gfmc_8x8_gr_population_size as b

ETA=0.5
ALPHA=1.2
T0=b.T
PHI=0.056022964674162276

zpop=np.load('gr_8x8_population_scaling_M128.npz')
EDGES=zpop['edges'].astype(float)
GREPS=zpop['greps'].astype(float)
GCOMB=zpop['gcombined'].astype(float)

base=np.load('krylov_scaling_8x8_a1.20_tr4096_va2048.npz')
tr=base['train_states'][:1024].astype(np.uint64)
iw0=base['iwtrain'][:1024].astype(float)
iw0/=iw0.sum()

def bin0(r):
    return int(np.clip(np.searchsorted(EDGES,r,side='right')-1,0,3))

def weighted_quantile(x,w,p):
    o=np.argsort(x); xx=np.asarray(x)[o]; ww=np.asarray(w)[o]
    c=np.cumsum(ww); c/=c[-1]
    return float(np.interp(p,c,xx))

def fit_threshold(r1,w):
    lo=weighted_quantile(r1,w,.05); hi=weighted_quantile(r1,w,.95)
    a=np.clip(r1,lo,hi)
    m1=weighted_quantile(a,w,.30); m2=weighted_quantile(a,w,.80)
    for _ in range(100):
        cut=.5*(m1+m2); lab=a>cut
        if lab.all() or (~lab).all(): break
        n1=np.sum(w[~lab]*a[~lab])/np.sum(w[~lab])
        n2=np.sum(w[lab]*a[lab])/np.sum(w[lab])
        if abs(n1-m1)+abs(n2-m2)<1e-10: break
        m1,m2=n1,n2
    return .5*(m1+m2),(m1,m2,lo,hi)

A_MASK=b.A_MASK
def marshall(x):
    return 1 if ((int(x)&A_MASK).bit_count()%2)==0 else -1

# Build one shared one-hop structure and populate the expensive base r0/log-amplitude cache once.
metas=[]; alln=[]
for x in tr:
    ls=b.neigh(int(x)); metas.append(ls); alln.extend(y for y,J,mr in ls)
b.ensure_r(list(map(int,tr))+alln)
r0=np.array([b.rc[int(x)] for x in tr],float)
bins0=np.array([bin0(r) for r in r0],int)

# Hidden target signs are diagnostics only.
ztr=b.evalz(b.bits2x(tr))
rel=np.angle(np.exp(1j*(ztr.imag-PHI)))
h=np.where(np.cos(rel)>=0,1,-1)
m=np.array([marshall(x) for x in tr],int)
y=h*m
p0=np.where(r0<=T0,1,-1)

def r1_for_g(G):
    out=np.empty(len(tr),float)
    for k,(x,ls) in enumerate(zip(tr,metas)):
        xx=int(x)
        lx=b.logc[xx]+ETA*G[bin0(b.rc[xx])]
        rr=b.diag_energy(xx)
        for yy,J,mr in ls:
            ly=b.logc[int(yy)]+ETA*G[bin0(b.rc[int(yy)])]
            rr += .5*J*mr*math.exp(ly-lx)
        out[k]=rr
    return out

def mass(mask,w): return float(np.sum(w*mask))
def overlap(p,w): return float(np.sum(w*p*y))

rows={}
changed_sets={}
for name,G in [('rep1',GREPS[0]),('rep2',GREPS[1]),('combined',GCOMB)]:
    r1=r1_for_g(G)
    wt=np.exp(ALPHA*ETA*G[bins0]); wt/=wt.sum()
    T1,aux=fit_threshold(r1,wt)
    p1=np.where(r1<=T1,1,-1)

    # Physical |a1|^2 importance weights from the stored alpha=1.2 sample weights.
    wp=iw0*np.exp(2*ETA*G[bins0]); wp/=wp.sum()
    changed=(p0!=p1)
    hit=changed & (p0!=y) & (p1==y)
    harm=changed & (p0==y) & (p1!=y)

    row=dict(
        G=G.tolist(),T1=float(T1),
        changed_phys=mass(changed,wp),
        hit_phys=mass(hit,wp),
        harm_phys=mass(harm,wp),
        precision_phys=mass(hit,wp)/max(mass(changed,wp),1e-300),
        O0_phys=overlap(p0,wp),
        O1_phys=overlap(p1,wp),
        wrong0_phys=mass(p0!=y,wp),
        wrong1_phys=mass(p1!=y,wp),
        n_changed=int(changed.sum()),n_hit=int(hit.sum()),n_harm=int(harm.sum())
    )
    rows[name]=row
    changed_sets[name]=set(np.flatnonzero(changed).tolist())
    print('RESULT',name,row,flush=True)
    for i in np.flatnonzero(changed):
        print('CHANGED',name,int(i),'state',int(tr[i]),'r0',float(r0[i]),'r1',float(r1[i]),
              'p0',int(p0[i]),'p1',int(p1[i]),'target',int(y[i]),'wphys',float(wp[i]),flush=True)

# Compare whether the replicas are changing the same states.
for a,c in [('rep1','rep2'),('rep1','combined'),('rep2','combined')]:
    A=changed_sets[a]; C=changed_sets[c]
    inter=len(A&C); union=len(A|C)
    print('SET_COMPARE',a,c,'intersection',inter,'union',union,
          'jaccard',float(inter/union) if union else 1.0,
          'only_'+a,sorted(A-C),'only_'+c,sorted(C-A),flush=True)

np.savez_compressed('diag_8x8_replica_g_signstep.npz',
    states=tr,r0=r0,p0=p0,target=y,iw0=iw0,edges=EDGES,
    greps=GREPS,gcombined=GCOMB)
