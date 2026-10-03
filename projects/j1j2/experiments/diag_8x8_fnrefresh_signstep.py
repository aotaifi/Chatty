import math,time,numpy as np
import gfmc_8x8_gr_population_size as b

ETA=.5
ALPHA=1.2
T0=b.T
zpop=np.load('gr_8x8_population_scaling_M128.npz')
EDGES=zpop['edges'].astype(float)
Q0=zpop['qmass'].astype(float); Q0/=Q0.sum()
G0=zpop['gcombined'].astype(float)
Q1=Q0*np.exp(2*ETA*G0); Q1/=Q1.sum()
print("CORRECTION eta",ETA,"edges",EDGES.tolist(),"q0",Q0.tolist(),
      "g0",G0.tolist(),"q1",Q1.tolist(),flush=True)

def bin0_from_r(r):
    return int(np.clip(np.searchsorted(EDGES,r,side='right')-1,0,3))
def ensure_loga1(states):
    b.ensure_r(states)
def loga1(x):
    x=int(x); b.ensure_r([x])
    return b.logc[x]+ETA*G0[bin0_from_r(b.rc[x])]

r1c={}; s1c={}; local1c={}
def ensure_r1(states):
    miss=[int(x) for x in np.unique(np.asarray(states,np.uint64)) if int(x) not in r1c]
    if not miss:return
    metas=[]; alln=[]
    for x in miss:
        ls=b.neigh(x); metas.append(ls); alln.extend(y for y,J,mr in ls)
    b.ensure_r(miss+alln)
    for x,ls in zip(miss,metas):
        lx=loga1(x); r=b.diag_energy(x)
        for y,J,mr in ls:
            r += .5*J*mr*math.exp(loga1(y)-lx)
        r1c[x]=float(r)

def weighted_quantile(x,w,p):
    o=np.argsort(x); xx=np.asarray(x)[o]; ww=np.asarray(w)[o]
    c=np.cumsum(ww); c/=c[-1]
    return float(np.interp(p,c,xx))

# Determine next threshold label-free on the existing alpha=1.2 sample,
# reweighted exactly for the small static-bin amplitude correction.
base=np.load('krylov_scaling_8x8_a1.20_tr4096_va2048.npz')
tr=base['train_states'][:1024].astype(np.uint64)
ensure_r1(tr)
rtr=np.array([r1c[int(x)] for x in tr])
r0tr=np.array([b.rc[int(x)] for x in tr])
bins=np.array([bin0_from_r(x) for x in r0tr])
w=np.exp(ALPHA*ETA*G0[bins]); w/=w.sum()
lo=weighted_quantile(rtr,w,.05); hi=weighted_quantile(rtr,w,.95)
a=np.clip(rtr,lo,hi)
m1=weighted_quantile(a,w,.30);m2=weighted_quantile(a,w,.80)
for _ in range(100):
    cut=.5*(m1+m2); lab=a>cut
    if lab.all() or (~lab).all():break
    n1=np.sum(w[~lab]*a[~lab])/np.sum(w[~lab])
    n2=np.sum(w[lab]*a[lab])/np.sum(w[lab])
    if abs(n1-m1)+abs(n2-m2)<1e-10:break
    m1,m2=n1,n2
T1=.5*(m1+m2)
print("THRESHOLD T0",T0,"T1",T1,"centers",m1,m2,"trim",lo,hi,flush=True)

A_MASK=b.A_MASK if hasattr(b,'A_MASK') else sum(1<<(x+b.L*y) for y in range(b.L) for x in range(b.L) if (x+y)%2==0)
def marshall(x):return 1 if ((int(x)&A_MASK).bit_count()%2)==0 else -1
def sign0(x):
    b.ensure_r([x]); return marshall(x)*(1 if b.rc[int(x)]<=T0 else -1)
def ensure_sign1(states):
    b.ensure_r(states); ensure_r1(states)
    for x in np.unique(np.asarray(states,np.uint64)):
        xx=int(x)
        if xx not in s1c:s1c[xx]=marshall(xx)*(1 if r1c[xx]<=T1 else -1)

def ensure_local1(states):
    miss=[int(x) for x in np.unique(np.asarray(states,np.uint64)) if int(x) not in local1c]
    if not miss:return
    metas=[];alln=[]
    for x in miss:
        ls=b.neigh(x);metas.append(ls);alln.extend(y for y,J,mr in ls)
    ensure_sign1(miss+alln)
    for x,ls in zip(miss,metas):
        lx=loga1(x); sx=s1c[x]; d=b.diag_energy(x);ys=[];rates=[]
        for y,J,mr in ls:
            rat=math.exp(loga1(y)-lx)
            if sx*s1c[y]<0:
                ys.append(y);rates.append(.5*J*rat)
            else:d += .5*J*rat
        rates=np.asarray(rates,float);ys=np.asarray(ys,np.uint64)
        local1c[x]=(float(d),float(d-rates.sum()),ys,rates)

# Memory-safe diagnostic: use the 1024 alpha=1.2 states for which r1 is already computed.
# Do not expand the full 4096-state physical pool into its second neighbor shell.
sp0tr=np.array([marshall(x)*(1 if r0<=T0 else -1) for x,r0 in zip(tr,r0tr)])
sp1tr=np.array([marshall(x)*(1 if r1<=T1 else -1) for x,r1 in zip(tr,rtr)])
flip=float(np.sum(w*(sp0tr!=sp1tr)))
print("DIAG train_reweighted_sign_change",flip,
      "n_changed",int(np.sum(sp0tr!=sp1tr)),
      "r1q",np.quantile(rtr,[0,.1,.5,.9,1]).tolist(),flush=True)


# Hidden ViT signs are diagnostics only; threshold T1 above is fully label-free.
PHI=0.056022964674162276  # fixed from original alpha=1.2 8x8 scaling run
ztr=b.evalz(b.bits2x(tr))
rel=np.angle(np.exp(1j*(ztr.imag-PHI)))
h=np.where(np.cos(rel)>=0,1,-1)
m=np.array([marshall(x) for x in tr],dtype=int)
y=h*m
p0=np.where(r0tr<=T0,1,-1)
p1=np.where(rtr<=T1,1,-1)

# Two weightings: threshold-sample reweighting and physical |a1|^2 importance weighting.
iw0=base['iwtrain'][:len(tr)].astype(float)
wp=iw0*np.exp(2*ETA*G0[bins]); wp/=wp.sum()
changed=(p0!=p1)
hit=changed & (p0!=y) & (p1==y)
harm=changed & (p0==y) & (p1!=y)
amb=changed & ~(hit|harm)

def ov(p,ww): return float(np.sum(ww*p*y))
def mass(mask,ww): return float(np.sum(ww*mask))
print('NODE_DIAG',
      'O0_phys',ov(p0,wp),'O1_phys',ov(p1,wp),
      'wrong0_phys',mass(p0!=y,wp),'wrong1_phys',mass(p1!=y,wp),
      'changed_phys',mass(changed,wp),
      'hit_phys',mass(hit,wp),'harm_phys',mass(harm,wp),
      'precision_phys',mass(hit,wp)/max(mass(changed,wp),1e-300),
      'n_changed',int(changed.sum()),'n_hit',int(hit.sum()),'n_harm',int(harm.sum()),
      flush=True)
idx=np.flatnonzero(changed)
for i in idx:
    print('CHANGED',int(i),'state',int(tr[i]),'r0',float(r0tr[i]),'r1',float(rtr[i]),
          'p0',int(p0[i]),'p1',int(p1[i]),'target',int(y[i]),'wphys',float(wp[i]),flush=True)
np.savez_compressed('diag_8x8_fnrefresh_signstep.npz',
    states=tr,r0=r0tr,r1=rtr,T0=T0,T1=T1,bins=bins,
    p0=p0,p1=p1,target=y,wphys=wp,changed=changed,hit=hit,harm=harm,
    g0=G0,edges=EDGES)
