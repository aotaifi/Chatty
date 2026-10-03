import math,time,json
import numpy as np
import jax, jax.numpy as jnp
from flax import linen as nn, serialization
import optax
import gfmc_6x6_gr_population_size as b

SEED=202609294
ETA=0.5
NTR=2048
NVA=1024
summary=np.load('gr_6x6_population_scaling_M128.npz')
G0=summary['gcombined'].astype(float)
EDGES=summary['edges'].astype(float)
base=np.load('krylov_scaling_6x6_a1.20_tr4096_va2048.npz')
rg=np.random.default_rng(SEED)

def bin0(r):
    return int(np.clip(np.searchsorted(EDGES,r,side='right')-1,0,3))
def loga1(x):
    x=int(x); b.ensure_r([x])
    return b.logc[x]+ETA*G0[bin0(b.rc[x])]
def k1sign(x):
    x=int(x); b.ensure_r([x]); return b.signc[x]

def choose(states,w,n,seed):
    rr=np.random.default_rng(seed)
    ids=np.sort(rr.choice(len(states),size=n,replace=False))
    ww=np.asarray(w[ids],float); ww/=ww.sum()
    return states[ids].astype(np.uint64),ww,ids

tr,trw0,trids=choose(base['train_states'],base['iwtrain'],NTR,SEED+1)
va,vaw0,vaids=choose(base['val_states'],base['iwval'],NVA,SEED+2)
print('SUBSETS',len(tr),len(va),flush=True)

def diagonal(s):
    return b.diag_energy(int(s))

def build(states,w0,label):
    t0=time.time(); states=np.asarray(states,np.uint64)
    lists=[]; flat=[]; maxm=0
    for s in states:
        li=b.neigh(int(s)); lists.append(li); maxm=max(maxm,len(li))
        flat.extend(y for y,J,mr in li)
    flat=np.asarray(flat,np.uint64)
    b.ensure_r(np.concatenate([states,flat]))
    X=b.bits2x(states).astype(np.float32)
    Xn=np.zeros((len(states),maxm,b.N),np.float32)
    C0=np.zeros((len(states),maxm),np.float32)
    C1=np.zeros_like(C0); M=np.zeros_like(C0); SP=np.zeros_like(C0)
    D=np.asarray([diagonal(s) for s in states],np.float32)
    bins=np.array([bin0(b.rc[int(x)]) for x in states],int)
    w1=np.asarray(w0,float)*np.exp(2*ETA*G0[bins]); w1/=w1.sum()
    for i,(x,li) in enumerate(zip(states,lists)):
        x=int(x); lx0=b.logc[x]; lx1=loga1(x); sx=k1sign(x)
        for k,(y,J,mr) in enumerate(li):
            y=int(y)
            Xn[i,k]=b.bits2x(np.array([y],np.uint64))[0]
            C0[i,k]=.5*J*math.exp(b.logc[y]-lx0)
            C1[i,k]=.5*J*math.exp(loga1(y)-lx1)
            SP[i,k]=sx*k1sign(y); M[i,k]=1.
    print('BUILT',label,'maxm',maxm,'r_cache',len(b.rc),'neval',b.neval,'sec',time.time()-t0,flush=True)
    return dict(states=states,X=X,Xn=Xn,C0=C0,C1=C1,M=M,SP=SP,D=D,
                w0=np.asarray(w0,np.float32),w1=np.asarray(w1,np.float32))

TR=build(tr,trw0,'TRAIN')
VA=build(va,vaw0,'VAL')

def weighted_stats(el,w):
    w=np.asarray(w,float); w/=w.sum(); el=np.asarray(el,float)
    mu=float(np.sum(w*el)); var=float(np.sum(w*(el-mu)**2))
    ess=float(1/np.sum(w*w))
    return mu,math.sqrt(var),math.sqrt(var/ess),ess

def baseline_local(dat,Ckey,wkey):
    el=dat['D']+np.sum(dat['M']*dat[Ckey]*dat['SP'],axis=1)
    return el,weighted_stats(el,dat[wkey])

for label,dat in [('TRAIN',TR),('VAL',VA)]:
    e0,s0=baseline_local(dat,'C0','w0')
    e1,s1=baseline_local(dat,'C1','w1')
    print('AMP_RESID',label,
          'a0_mean_sd_se_ess',s0,
          'a1_mean_sd_se_ess',s1,
          'sd_ratio',s1[1]/s0[1],flush=True)

class TICorr(nn.Module):
    ch:int=24
    @nn.compact
    def __call__(self,x):
        y=x.reshape((-1,b.L,b.L,1))
        for _ in range(3):
            y=jnp.pad(y,((0,0),(1,1),(1,1),(0,0)),mode='wrap')
            y=nn.Conv(self.ch,(3,3),padding='VALID')(y); y=nn.tanh(y)
        y=jnp.mean(y,axis=(1,2))
        y=nn.Dense(24)(y); y=nn.tanh(y)
        y=nn.Dense(1,kernel_init=nn.initializers.zeros,
                   bias_init=lambda key,shape,dtype:jnp.ones(shape,dtype))(y)
        return y[:,0]

def subset_dat(dat,ix):
    out={k:(v[ix] if isinstance(v,np.ndarray) and v.shape[:1]==(len(dat['X']),) else v)
         for k,v in dat.items()}
    for k in ('w0','w1'):
        out[k]=np.asarray(out[k],np.float32); out[k]/=out[k].sum()
    return out

pv=np.random.default_rng(SEED+3).permutation(len(VA['X']))
VSEL=subset_dat(VA,pv[:len(pv)//2])
VTEST=subset_dat(VA,pv[len(pv)//2:])
test_orig_ids=vaids[pv[len(pv)//2:]]

def ste_sign(q,T):
    soft=jnp.tanh(q/T); hard=jnp.where(q>=0,1.,-1.)
    return soft+jax.lax.stop_gradient(hard-soft)

def make_train(seed):
    model=TICorr()
    params=model.init(jax.random.PRNGKey(seed),jnp.asarray(TR['X'][:4]))['params']
    opt=optax.chain(optax.clip_by_global_norm(2.0),optax.adamw(7e-4,weight_decay=2e-5))
    ost=opt.init(params); rr=np.random.default_rng(seed+100)
    TRN=len(TR['X'])

    @jax.jit
    def loss_fn(params,X,Xn,C,M,SP,D,W,T):
        q0=model.apply({'params':params},X)
        qn=model.apply({'params':params},Xn.reshape((-1,b.N))).reshape(Xn.shape[:2])
        c0=ste_sign(q0,T); cn=ste_sign(qn,T)
        el=D+jnp.sum(M*C*SP*(c0[:,None]*cn),axis=1)
        loss=TRN*jnp.mean(W*el)
        return loss,(loss,jnp.mean(q0<0))

    @jax.jit
    def step(params,ost,X,Xn,C,M,SP,D,W,T):
        (loss,aux),gr=jax.value_and_grad(loss_fn,has_aux=True)(params,X,Xn,C,M,SP,D,W,T)
        upd,ost=opt.update(gr,ost,params)
        return optax.apply_updates(params,upd),ost,loss,aux

    def corr(params,X,bs=4096):
        out=[]
        for i in range(0,len(X),bs):
            q=np.asarray(model.apply({'params':params},jnp.asarray(X[i:i+bs])))
            out.append(np.where(q>=0,1.,-1.))
        return np.concatenate(out)

    def eval_dat(params,dat):
        c0=corr(params,dat['X'])
        cn=corr(params,dat['Xn'].reshape(-1,b.N)).reshape(dat['Xn'].shape[:2])
        el=dat['D']+np.sum(dat['M']*dat['C1']*dat['SP']*(c0[:,None]*cn),axis=1)
        st=weighted_stats(el,dat['w1'])
        flip=float(np.sum(dat['w1']*(c0<0)))
        return st,flip,el,c0

    base_sel=baseline_local(VSEL,'C1','w1')[1]
    best=(base_sel[0],serialization.to_bytes(params),0)
    B=64
    for it in range(1201):
        ids=rr.choice(TRN,B,replace=False)
        T=max(.35,1.5*(.998**it))
        params,ost,loss,aux=step(params,ost,
            jnp.asarray(TR['X'][ids]),jnp.asarray(TR['Xn'][ids]),
            jnp.asarray(TR['C1'][ids]),jnp.asarray(TR['M'][ids]),
            jnp.asarray(TR['SP'][ids]),jnp.asarray(TR['D'][ids]),
            jnp.asarray(TR['w1'][ids]),jnp.asarray(T,np.float32))
        if it%100==0:
            st,ff,_,_=eval_dat(params,VSEL)
            print('NODE_TRAIN',seed,it,'loss',float(loss),'VSEL_E',st[0],
                  'SE',st[2],'flipmass',ff,flush=True)
            if st[0]<best[0]:
                best=(st[0],serialization.to_bytes(params),it)
    params=serialization.from_bytes(params,best[1])
    ssel,fsel,_,_=eval_dat(params,VSEL)
    stest,ftest,eltest,ctest=eval_dat(params,VTEST)
    return dict(seed=seed,beststep=best[2],params=params,
                sel=ssel,test=stest,flip_sel=fsel,flip_test=ftest,
                eltest=eltest,ctest=ctest)

base_test_el,base_test=baseline_local(VTEST,'C1','w1')
runs=[]
for seed in (SEED+10,SEED+20,SEED+30):
    rr=make_train(seed); runs.append(rr)
    delta=rr['eltest']-base_test_el
    w=VTEST['w1'].astype(float); w/=w.sum()
    dmu=float(np.sum(w*delta))
    dess=1/np.sum(w*w)
    dvar=float(np.sum(w*(delta-dmu)**2))
    dnaive=math.sqrt(dvar/dess)
    chainmeans=[]
    for c in range(16):
        m=(test_orig_ids%16)==c
        if np.any(m):
            wc=w[m]; wc/=wc.sum(); chainmeans.append(float(np.sum(wc*delta[m])))
    chainmeans=np.asarray(chainmeans)
    dchain=float(chainmeans.std(ddof=1)/math.sqrt(len(chainmeans)))
    print('NODE_TEST',seed,'baseE',base_test[0],'newE',rr['test'][0],
          'delta',dmu,'naiveSE',dnaive,'chainSE',dchain,
          'flipmass',rr['flip_test'],'beststep',rr['beststep'],flush=True)

out=[]
for rr in runs:
    delta=rr['eltest']-base_test_el
    w=VTEST['w1'].astype(float); w/=w.sum()
    dmu=float(np.sum(w*delta))
    chainmeans=[]
    for c in range(16):
        m=(test_orig_ids%16)==c
        if np.any(m):
            wc=w[m]; wc/=wc.sum(); chainmeans.append(float(np.sum(wc*delta[m])))
    chainmeans=np.asarray(chainmeans)
    out.append([rr['seed'],rr['beststep'],rr['test'][0],rr['test'][2],
                rr['flip_test'],dmu,
                float(chainmeans.std(ddof=1)/math.sqrt(len(chainmeans)))])
np.savez_compressed('a1_node_audit_6x6.npz',
    rows=np.asarray(out,float),g0=G0,edges=EDGES,
    amp_train_a0=np.asarray(baseline_local(TR,'C0','w0')[1]),
    amp_train_a1=np.asarray(baseline_local(TR,'C1','w1')[1]),
    amp_val_a0=np.asarray(baseline_local(VA,'C0','w0')[1]),
    amp_val_a1=np.asarray(baseline_local(VA,'C1','w1')[1]),
    base_test=np.asarray(base_test))
with open('a1_node_audit_6x6.json','w') as f:
    json.dump(dict(rows=out,columns=['seed','beststep','Etest','Etest_naiveSE',
        'flipmass','deltaE','deltaE_chainSE']),f,indent=2)
for i,rr in enumerate(runs):
    with open(f'a1_node_corr_seed{rr["seed"]}.msgpack','wb') as f:
        f.write(serialization.to_bytes(rr['params']))
print('DONE_AUDIT rows',out,flush=True)
