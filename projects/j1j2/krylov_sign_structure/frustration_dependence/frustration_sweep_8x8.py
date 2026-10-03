import os, math, json, time
import numpy as np
import jax, jax.numpy as jnp
import netket as nk, flax
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

L=8; N=64; ALPHA=1.2; NTR=2048; NVA=1024; NCHAINS=16
J2S=[0.00,0.20,0.40,0.50,0.60,0.80,1.00]
LAMS=[-2.,-1.,-.5,0.,.25,.5,.75,1.,1.25,1.5,2.,3.,4.]
SEED_T=202609282; SEED_V=202609283

hi=nk.hilbert.Spin(s=.5,N=N)
graph=nk.graph.Hypercube(length=L,n_dim=2,pbc=True,max_neighbor_order=2)
model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
sampler=nk.sampler.MetropolisExchange(hi,graph=graph,d_max=2,n_chains=16,sweep_size=N)
v=nk.vqs.MCState(sampler=sampler,apply_fun=apply,n_samples=16,
    variables=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N))),n_discard_per_chain=10)

q=lambda x,y:(x%L)+L*(y%L)
NN=[]; NNN=[]
for y in range(L):
    for x in range(L):
        i=q(x,y)
        NN += [(i,q(x+1,y)),(i,q(x,y+1))]
        NNN += [(i,q(x+1,y+1)),(i,q(x+1,y-1))]
A_MASK=sum(1<<(x+L*y) for y in range(L) for x in range(L) if (x+y)%2==0)

def bits2x(ss):
    a=np.asarray(ss,np.uint64).reshape(-1,1)
    return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(float)-1

def evalz(states,batch=4096):
    X=bits2x(states); out=[]
    for i in range(0,len(X),batch):
        out.append(np.asarray(v.log_value(jnp.asarray(X[i:i+batch]))))
    return np.concatenate(out)

def load_ckpt(J2):
    fn=f"vit_J2={J2:.2f}_N=8x8_k=0.mpack"
    with open(fn,'rb') as f: obj=flax.serialization.msgpack_restore(f.read())
    v.variables=flax.serialization.from_state_dict(v.variables,obj)
    print("LOADED",J2,fn,flush=True)

def all_bonds(J2):
    return [(i,j,1.0,0) for i,j in NN]+[(i,j,J2,1) for i,j in NNN]

def sample(J2,seed,nout,burn=160,between=8):
    ALL=all_bonds(J2)
    rg=np.random.default_rng(seed)
    ss=[]
    for _ in range(NCHAINS):
        pos=rg.choice(N,N//2,replace=False); z=0
        for i in pos: z|=1<<int(i)
        ss.append(z)
    la=evalz(ss).real
    out=[]; nround=nout//NCHAINS; acc=props=0
    for it in range(burn+between*nround):
        cand=[]; valid=[]
        for s in ss:
            i,j,J,kind=ALL[int(rg.integers(len(ALL)))]
            ok=((s>>i)^(s>>j))&1
            cand.append(s^(1<<i)^(1<<j) if ok else s); valid.append(bool(ok))
        lb=evalz(cand).real
        ac=(np.log(rg.random(NCHAINS))<np.minimum(0,ALPHA*(lb-la))) & np.asarray(valid)
        acc+=int(ac.sum()); props+=NCHAINS
        for k in np.where(ac)[0]: ss[k]=cand[k]
        la=np.where(ac,lb,la)
        if it>=burn and (it-burn+1)%between==0: out.extend(ss)
    print("SAMPLE",J2,seed,len(out),"acc",acc/props,flush=True)
    return np.asarray(out,np.uint64)

def marshall(states):
    return np.asarray([1 if ((int(s)&A_MASK).bit_count()%2)==0 else -1 for s in states],np.int8)

def components(states,J2,label):
    states=np.asarray(states,np.uint64)
    z=evalz(states); la=z.real
    n=len(states)
    D1=np.zeros(n); D2=np.zeros(n); O1=np.zeros(n); O2=np.zeros(n)

    # diagonal pieces
    X=bits2x(states)
    for i,j in NN: D1 += .25*X[:,i]*X[:,j]
    for i,j in NNN: D2 += .25*J2*X[:,i]*X[:,j]

    # one-hop off-diagonal pieces; evaluate every unique neighbor once
    flat=[]; meta=[]
    for bi,s0 in enumerate(states):
        s=int(s0)
        for i,j in NN:
            if ((s>>i)^(s>>j))&1:
                flat.append(s^(1<<i)^(1<<j)); meta.append((bi,0))
        if J2!=0:
            for i,j in NNN:
                if ((s>>i)^(s>>j))&1:
                    flat.append(s^(1<<i)^(1<<j)); meta.append((bi,1))
    unq,inv=np.unique(np.asarray(flat,np.uint64),return_inverse=True)
    lu=evalz(unq).real
    for jj,(bi,kind) in zip(inv,meta):
        rat=math.exp(float(lu[jj]-la[bi]))
        if kind==0: O1[bi] += -.5*rat
        else: O2[bi] += .5*J2*rat

    lw=(2-ALPHA)*la; lw-=lw.max(); w=np.exp(lw); w/=w.sum()
    r=D1+D2+O1+O2
    print("COMP",J2,label,"n",n,"uniqN",len(unq),
          "rQ",np.quantile(r,[0,.1,.5,.9,1]).tolist(),flush=True)
    return dict(states=states,z=z,la=la,w=w,D1=D1,D2=D2,O1=O1,O2=O2,r=r)

def robust_kmeans_t(score):
    a=np.asarray(score,float)
    lo,hi=np.quantile(a,[.1,.9]); aa=np.clip(a,lo,hi)
    m1,m2=np.quantile(aa,[.3,.8])
    for _ in range(100):
        cut=.5*(m1+m2); lab=aa>cut
        if lab.all() or (~lab).all(): break
        n1=aa[~lab].mean(); n2=aa[lab].mean()
        if abs(n1-m1)+abs(n2-m2)<1e-10: break
        m1,m2=n1,n2
    return float(.5*(m1+m2))

def oracle_threshold(score,y,w):
    # prediction p=+1 for oriented_score <= t, -1 above t
    best=None
    for orient in (1.,-1.):
        s=orient*np.asarray(score,float)
        order=np.argsort(s)
        yy=np.asarray(y,float)[order]; ww=np.asarray(w,float)[order]; ss=s[order]
        cur=-float(np.sum(ww*yy)) # all p=-1
        # threshold below minimum
        cand=(cur,float(ss[0]-1.),orient)
        if best is None or cand[0]>best[0]: best=cand
        k=0
        while k<len(ss):
            val=ss[k]
            while k<len(ss) and ss[k]==val:
                cur += 2*float(ww[k]*yy[k]); k+=1
            t=float(val if k==len(ss) else .5*(val+ss[k]))
            cand=(cur,t,orient)
            if cand[0]>best[0]: best=cand
    return best

def pred_thresh(score,t,orient=1.):
    return np.where(orient*np.asarray(score)<=t,1,-1)

def summarize_split(dat,y,p):
    w=dat['w']; wrong=float(np.sum(w*(p!=y)))
    return dict(overlap=float(np.sum(w*p*y)),wrong=wrong,flipmass=float(np.sum(w*(p<0))))

def analyze_j2(J2):
    load_ckpt(J2)
    tr=components(sample(J2,SEED_T,NTR),J2,"tr")
    va=components(sample(J2,SEED_V,NVA),J2,"va")

    # Binary global phase from train/val, then orient global sign so Marshall is majority-positive.
    def phase(dat):
        return .5*np.angle(np.sum(dat['w']*np.exp(2j*dat['z'].imag)))
    phit,phiv=phase(tr),phase(va)
    phi=.5*np.angle(np.exp(2j*phit)+np.exp(2j*phiv))
    def corr(dat):
        rel=np.angle(np.exp(1j*(dat['z'].imag-phi)))
        h=np.where(np.cos(rel)>=0,1,-1)
        y=h*marshall(dat['states'])
        leak=float(np.sum(dat['w']*np.abs(np.sin(rel))))
        return y,leak
    yt,leakt=corr(tr); yv,leakv=corr(va)
    if float(np.sum(tr['w']*yt)+np.sum(va['w']*yv))<0:
        yt=-yt; yv=-yv

    pm_t=np.ones(len(yt),int); pm_v=np.ones(len(yv),int)
    Mtr=summarize_split(tr,yt,pm_t); Mva=summarize_split(va,yv,pm_v)

    # Actual label-free K1.
    tk=robust_kmeans_t(tr['r'])
    pk_t=np.where(tr['r']<=tk,1,-1); pk_v=np.where(va['r']<=tk,1,-1)
    Ktr=summarize_split(tr,yt,pk_t); Kva=summarize_split(va,yv,pk_v)
    removed_tr=(Mtr['wrong']-Ktr['wrong'])/Mtr['wrong'] if Mtr['wrong']>1e-14 else 0.
    removed_va=(Mva['wrong']-Kva['wrong'])/Mva['wrong'] if Mva['wrong']>1e-14 else 0.

    # Oracle diagnostic for full r and off-diagonal frustration field.
    ofull=oracle_threshold(tr['r'],yt,tr['w'])
    pfv=pred_thresh(va['r'],ofull[1],ofull[2])
    full_oracle_val=summarize_split(va,yv,pfv)

    sweep=[]
    for lam in LAMS:
        st=tr['O1']+lam*tr['O2']; sv=va['O1']+lam*va['O2']
        o=oracle_threshold(st,yt,tr['w'])
        pt=pred_thresh(st,o[1],o[2]); pv=pred_thresh(sv,o[1],o[2])
        rt=summarize_split(tr,yt,pt); rv=summarize_split(va,yv,pv)
        sweep.append(dict(lam=lam,t=float(o[1]),orient=float(o[2]),
                          trainO=rt['overlap'],valO=rv['overlap'],
                          trainWrong=rt['wrong'],valWrong=rv['wrong']))
    best_train=max(sweep,key=lambda x:x['trainO'])
    best_val=max(sweep,key=lambda x:x['valO'])
    phys=next(x for x in sweep if x['lam']==1.)

    out=dict(J2=J2,phi=float(phi),leakTrain=leakt,leakVal=leakv,
        marshallTrain=Mtr,marshallVal=Mva,
        k1Threshold=tk,k1Train=Ktr,k1Val=Kva,
        removedFracTrain=float(removed_tr),removedFracVal=float(removed_va),
        fullOracleTrain=float(ofull[0]),fullOracleVal=full_oracle_val,
        lambdaPhysical=phys,lambdaBestTrain=best_train,lambdaBestValOracle=best_val,
        lambdaSweep=sweep)
    print("SUMMARY",json.dumps(out),flush=True)
    return out

allres=[]
for J2 in J2S:
    t0=time.time(); out=analyze_j2(J2); out['seconds']=time.time()-t0; allres.append(out)
    with open('frustration_sweep_8x8_partial.json','w') as f: json.dump(allres,f,indent=2)
with open('frustration_sweep_8x8.json','w') as f: json.dump(allres,f,indent=2)
print("FINAL_DONE",flush=True)
