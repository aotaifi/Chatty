import os, sys, json, math, time
import numpy as np
import jax, jax.numpy as jnp
import netket as nk, flax
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

J2=float(sys.argv[1])
L=8; N=64; ALPHA=1.2
NTR=int(os.environ.get("NTR","1024"))
NVA=int(os.environ.get("NVA","512"))
NCHAINS=16
seedbase=int(round(J2*10000))+20260930
hi=nk.hilbert.Spin(s=.5,N=N)
graph=nk.graph.Hypercube(length=L,n_dim=2,pbc=True,max_neighbor_order=2)
model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
sam=nk.sampler.MetropolisExchange(hi,graph=graph,d_max=2,n_chains=16,sweep_size=N)
v=nk.vqs.MCState(sampler=sam,apply_fun=apply,n_samples=16,
    variables=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N))),n_discard_per_chain=10)
ck=f"vit_J2={J2:.2f}_N=8x8_k=0.mpack"
with open(ck,"rb") as f: obj=flax.serialization.msgpack_restore(f.read())
v.variables=flax.serialization.from_state_dict(v.variables,obj)
print("LOADED",J2,ck,"nparams",v.n_parameters,flush=True)

def evalz(X,batch=4096):
    X=np.asarray(X,float); out=[]
    for i in range(0,len(X),batch):
        out.append(np.asarray(v.log_value(jnp.asarray(X[i:i+batch]))))
    return np.concatenate(out)

def bits2x(ss):
    a=np.asarray(ss,np.uint64).reshape(-1,1)
    return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(float)-1

def bonds():
    nn=[]; nnn=[]; q=lambda x,y:(x%L)+L*(y%L)
    for y in range(L):
        for x in range(L):
            i=q(x,y)
            nn += [(i,q(x+1,y)),(i,q(x,y+1))]
            nnn += [(i,q(x+1,y+1)),(i,q(x+1,y-1))]
    return nn,nnn
NN,NNN=bonds()
ALL=[(i,j,0) for i,j in NN]+[(i,j,1) for i,j in NNN]

def sample(seed,n):
    rg=np.random.default_rng(seed); chains=[]
    for _ in range(NCHAINS):
        pos=rg.choice(N,N//2,replace=False); s=0
        for i in pos:s|=1<<int(i)
        chains.append(s)
    la=evalz(bits2x(chains)).real
    out=[]; burn=160; between=8; nround=n//NCHAINS
    acc=props=0
    for it in range(burn+between*nround):
        cand=[]; valid=[]
        for s in chains:
            i,j,_=ALL[int(rg.integers(len(ALL)))]
            ok=((s>>i)^(s>>j))&1
            cand.append(s^(1<<i)^(1<<j) if ok else s); valid.append(bool(ok))
        lb=evalz(bits2x(cand)).real
        ac=(np.log(rg.random(NCHAINS))<np.minimum(0,ALPHA*(lb-la))) & np.asarray(valid)
        acc+=int(ac.sum()); props+=NCHAINS
        for k in np.where(ac)[0]: chains[k]=cand[k]
        la=np.where(ac,lb,la)
        if it>=burn and (it-burn+1)%between==0: out.extend(chains)
    print("SAMPLE",seed,len(out),"acc",acc/props,flush=True)
    return np.asarray(out[:n],np.uint64)

def fields(states,label):
    states=np.asarray(states,np.uint64)
    z=evalz(bits2x(states)); la=z.real
    D1=np.zeros(len(states)); D2=np.zeros(len(states))
    F1=np.zeros(len(states)); F2=np.zeros(len(states))
    flat=[]; meta=[]
    for bi,s0 in enumerate(states):
        s=int(s0)
        for i,j,typ in ALL:
            same=((s>>i)&1)==((s>>j)&1)
            if typ==0: D1[bi] += .25 if same else -.25
            else: D2[bi] += .25 if same else -.25
            if not same:
                flat.append(s^(1<<i)^(1<<j)); meta.append((bi,typ))
    unq,inv=np.unique(np.asarray(flat,np.uint64),return_inverse=True)
    lu=evalz(bits2x(unq)).real
    for jj,(bi,typ) in zip(inv,meta):
        rat=math.exp(lu[jj]-la[bi])
        if typ==0: F1[bi] += -.5*rat
        else: F2[bi] += +.5*rat
    lw=(2-ALPHA)*la; lw-=lw.max(); w=np.exp(lw); w/=w.sum()
    print("FIELDS",label,"n",len(states),"unique_neigh",len(unq),"ESS",1/np.sum(w*w),flush=True)
    return dict(states=states,z=z,D1=D1,D2=D2,F1=F1,F2=F2,w=w)

TR=fields(sample(seedbase,NTR),"tr")
VA=fields(sample(seedbase+1,NVA),"va")

# One global binary phase from both samples, then orient relative to Marshall.
def binary_phi(dat):
    return .5*np.angle(np.sum(dat["w"]*np.exp(2j*dat["z"].imag)))
pt=binary_phi(TR); pv=binary_phi(VA)
phi=.5*np.angle(np.exp(2j*pt)+np.exp(2j*pv))
A_MASK=sum(1<<(x+L*y) for y in range(L) for x in range(L) if (x+y)%2==0)
def marshall(states):
    return np.asarray([1 if ((int(s)&A_MASK).bit_count()%2)==0 else -1 for s in states],np.int8)
def target(dat):
    rel=np.angle(np.exp(1j*(dat["z"].imag-phi)))
    h=np.where(np.cos(rel)>=0,1,-1)
    y=h*marshall(dat["states"])
    return y.astype(int)
Yt=target(TR); Yv=target(VA)
if np.sum(TR["w"]*Yt)<0:
    Yt=-Yt; Yv=-Yv

def overlap(pred,y,w): return float(np.sum(w*pred*y))
def wrong(pred,y,w): return float(np.sum(w*(pred!=y)))

baseOtr=overlap(np.ones(len(Yt),int),Yt,TR["w"])
baseOva=overlap(np.ones(len(Yv),int),Yv,VA["w"])

def weighted_quantile(x,w,p):
    o=np.argsort(x); xx=np.asarray(x)[o]; ww=np.asarray(w)[o]
    c=np.cumsum(ww); c/=c[-1]
    return float(np.interp(p,c,xx))

def kmeans_threshold(score,w):
    lo=weighted_quantile(score,w,.1); hi=weighted_quantile(score,w,.9)
    a=np.clip(score,lo,hi)
    m1=weighted_quantile(a,w,.3); m2=weighted_quantile(a,w,.8)
    for _ in range(100):
        cut=.5*(m1+m2); lab=a>cut
        if lab.all() or (~lab).all():break
        n1=np.sum(w[~lab]*a[~lab])/np.sum(w[~lab])
        n2=np.sum(w[lab]*a[lab])/np.sum(w[lab])
        if abs(n1-m1)+abs(n2-m2)<1e-10:break
        m1,m2=n1,n2
    return .5*(m1+m2)

def oracle_fit(score,y,w):
    score=np.asarray(score,float); y=np.asarray(y,int); w=np.asarray(w,float); w=w/w.sum()
    best=(-2.0,None,None)
    for orient in (1,-1):
        s=orient*score
        order=np.argsort(s)
        # pred=+1 for s<=t and -1 for s>t.
        # Start below min: all -1, then sweep t upward and flip states to +1.
        pred=-np.ones(len(s),int)
        t0=float(s[order[0]]-max(1.0,abs(s[order[0]])*1e-9))
        cur=overlap(pred,y,w)
        if cur>best[0]: best=(cur,t0,orient)
        for k,ix in enumerate(order):
            pred[ix]=1
            cur=overlap(pred,y,w)
            if k==len(order)-1:
                t=float(s[ix]+max(1.0,abs(s[ix])*1e-9))
            else:
                t=float(.5*(s[ix]+s[order[k+1]]))
            if cur>best[0]: best=(cur,t,orient)
    return best

def apply_oracle(score,t,orient):
    s=orient*np.asarray(score,float)
    return np.where(s<=t,1,-1)

rtr=TR["D1"]+J2*TR["D2"]+TR["F1"]+J2*TR["F2"]
rva=VA["D1"]+J2*VA["D2"]+VA["F1"]+J2*VA["F2"]
tk=kmeans_threshold(rtr,TR["w"])
pktr=np.where(rtr<=tk,1,-1); pkva=np.where(rva<=tk,1,-1)

lams=np.round(np.arange(-.25,2.0001,.05),10)
lsweep=[]
for lam in lams:
    st=TR["F1"]+lam*TR["F2"]
    sv=VA["F1"]+lam*VA["F2"]
    otr,t,orient=oracle_fit(st,Yt,TR["w"])
    pvpred=apply_oracle(sv,t,orient)
    lsweep.append((float(lam),float(otr),overlap(pvpred,Yv,VA["w"]),float(t),int(orient)))
best=max(lsweep,key=lambda q:q[1])
phys=min(lsweep,key=lambda q:abs(q[0]-J2))

wmtr=wrong(np.ones(len(Yt),int),Yt,TR["w"])
wmva=wrong(np.ones(len(Yv),int),Yv,VA["w"])
wktr=wrong(pktr,Yt,TR["w"]); wkva=wrong(pkva,Yv,VA["w"])
removed_tr=(wmtr-wktr)/wmtr if wmtr>0 else 1.0
removed_va=(wmva-wkva)/wmva if wmva>0 else 1.0

out=dict(
    J2=J2,phi=float(phi),
    MarshallO_train=baseOtr,MarshallO_val=baseOva,
    MarshallWrong_train=wmtr,MarshallWrong_val=wmva,
    K1_threshold=float(tk),
    K1O_train=overlap(pktr,Yt,TR["w"]),K1O_val=overlap(pkva,Yv,VA["w"]),
    K1Wrong_train=wktr,K1Wrong_val=wkva,
    K1RemovedFraction_train=float(removed_tr),K1RemovedFraction_val=float(removed_va),
    lambda_best_train=best[0],lambda_best_trainO=best[1],lambda_best_valO=best[2],
    lambda_physical=phys[0],lambda_physical_trainO=phys[1],lambda_physical_valO=phys[2],
    lambda_gap=float(best[0]-J2),
    lambda_sweep=lsweep,
    NTR=NTR,NVA=NVA
)
print("SUMMARY",json.dumps(out,sort_keys=True),flush=True)
with open(f"frustration_scan_8x8_J2_{J2:.2f}.json","w") as f: json.dump(out,f,indent=2)
np.savez_compressed(f"frustration_scan_8x8_J2_{J2:.2f}.npz",
    tr_states=TR["states"],va_states=VA["states"],rtr=rtr,rva=rva,
    F1tr=TR["F1"],F2tr=TR["F2"],F1va=VA["F1"],F2va=VA["F2"],
    wt=TR["w"],wv=VA["w"],Yt=Yt,Yv=Yv)
