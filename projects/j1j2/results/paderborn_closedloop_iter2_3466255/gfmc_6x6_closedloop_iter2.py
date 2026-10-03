import math,time,numpy as np
import gfmc_6x6_gr_population_size as b

ETA=.5
ALPHA=1.2
T0=-14.985799779143964
zpop=np.load('gr_6x6_population_scaling_M128.npz')
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
base=np.load('krylov_scaling_6x6_a1.20_tr4096_va2048.npz')
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

# Cheap diagnostics on same physical sample pool.
pool=np.load('energy_krylov_vs_vit_6x6_indep.npz')['states'].astype(np.uint64)
ensure_sign1(pool);ensure_local1(pool)
sp0=np.array([sign0(x) for x in pool]);sp1=np.array([s1c[int(x)] for x in pool])
bp=np.array([bin0_from_r(b.rc[int(x)]) for x in pool])
wp=np.exp(2*ETA*G0[bp]);wp/=wp.sum()
flip=float(np.sum(wp*(sp0!=sp1)))
eguide=float(np.sum(wp*np.array([local1c[int(x)][1] for x in pool])))
print("DIAG physical_weighted_sign_change",flip,"guideE_reweighted",eguide,
      "r1q",np.quantile([r1c[int(x)] for x in pool],[0,.1,.5,.9,1]).tolist(),flush=True)

def systematic(w,rng,M):
    c=np.cumsum(w);c[-1]=1.
    return np.searchsorted(c,rng.random()/M+np.arange(M)/M,'right')

def run(seed,M=128,beta_target=1.2,burn_beta=.4,tau_max=.025):
    rg=np.random.default_rng(seed)
    # Initial states from physical pool, mildly reweighted toward a1^2.
    b.ensure_r(pool)
    bp=np.array([bin0_from_r(b.rc[int(x)]) for x in pool])
    pr=np.exp(2*ETA*G0[bp]);pr/=pr.sum()
    walkers=pool[rg.choice(len(pool),M,replace=True,p=pr)].copy()
    beta=0.;it=0;Es=[];snaps=[];t0=time.time()
    while beta<beta_target:
        ensure_local1(walkers);dat=[local1c[int(x)] for x in walkers]
        diag=np.array([z[0] for z in dat]);el=np.array([z[1] for z in dat]);Eref=float(el.mean())
        mx=max(0.,float(np.max(diag-Eref)));tau=min(tau_max,0.8/mx if mx>0 else tau_max)
        nxt=np.empty(M,np.uint64);bw=np.empty(M)
        for k,(x,(d,e,ys,rate)) in enumerate(zip(walkers,dat)):
            stay=1-tau*(d-Eref);ws=tau*rate;tot=stay+ws.sum()
            if stay<0 or tot<=0:raise RuntimeError(("bad",stay,tot,tau,d,Eref))
            u=rg.random()*tot
            if u<stay:y=x
            else:
                j=np.searchsorted(np.cumsum(ws),u-stay,'right')
                y=int(ys[min(j,len(ys)-1)])
            nxt[k]=y;bw[k]=tot
        bw/=bw.sum();walkers=nxt[systematic(bw,rg,M)]
        beta+=tau;it+=1
        if beta>=burn_beta:
            ensure_local1(walkers)
            Es.append(float(np.mean([local1c[int(x)][1] for x in walkers])))
            snaps.append(walkers.copy())
        if it==1 or it%5==0:
            print("ITER2_PROG",seed,it,"beta",beta,"tau",tau,"E",Eref,
                  "uniq",len(np.unique(walkers)),"r0cache",len(b.rc),
                  "r1cache",len(r1c),"local",len(local1c),"neval",b.neval,
                  "sec",time.time()-t0,flush=True)
    mixed=np.concatenate(snaps);E=np.asarray(Es)
    print("ITER2_RESULT",seed,"Emean",float(E.mean()),"tail8",float(E[-8:].mean()),
          "nE",len(E),"Nmixed",len(mixed),"unique",len(np.unique(mixed)),
          "sec",time.time()-t0,flush=True)
    np.savez_compressed(f'gfmc_6x6_closedloop_iter2_seed{seed}.npz',
                        mixed=mixed,Es=E,T1=T1,eta=ETA,g0=G0,edges=EDGES)
    return mixed,E

# Two independent FN replicas for stability of the second iteration.
mixes=[];erows=[]
for seed in (9701,9702):
    mix,E=run(seed)
    mixes.append(mix);erows.append([seed,E.mean(),E[-8:].mean(),len(E),len(mix),len(np.unique(mix))])

# Second density-ratio correction in the same static r0 coordinate.
# q1 bin masses are known analytically because the first correction is piecewise constant in these bins.
g2=[];cts=[]
for seed,mix in zip((9701,9702),mixes):
    b.ensure_r(mix)
    bb=np.array([bin0_from_r(b.rc[int(x)]) for x in mix])
    cc=np.bincount(bb,minlength=4).astype(float);fm=(cc+.5)/(cc.sum()+2.0)
    gg=np.log(fm/Q1);gg-=np.sum(Q1*gg)
    g2.append(gg);cts.append(cc)
    print("G2_REP",seed,"counts",cc.tolist(),"g2",gg.tolist(),flush=True)
g2=np.asarray(g2);dg=g2[0]-g2[1]
allmix=np.concatenate(mixes);b.ensure_r(allmix)
bb=np.array([bin0_from_r(b.rc[int(x)]) for x in allmix])
cc=np.bincount(bb,minlength=4).astype(float);fm=(cc+.5)/(cc.sum()+2.0)
gc=np.log(fm/Q1);gc-=np.sum(Q1*gc)
print("CLOSED_LOOP_SUMMARY","T1",T1,"sign_change",flip,
      "Erows",erows,"g2combined",gc.tolist(),"g2diff",dg.tolist(),flush=True)
np.savez_compressed('gfmc_6x6_closedloop_iter2_summary.npz',
                    T1=T1,eta=ETA,g0=G0,q1=Q1,edges=EDGES,
                    erows=np.asarray(erows,float),g2reps=g2,g2combined=gc,
                    g2diff=dg,sign_change=flip,guideE=eguide)
