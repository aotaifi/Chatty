import math,time,json
import numpy as np
import jax, jax.numpy as jnp
from flax import linen as nn
import optax
exec(open('bench_native_fast.py').read().split("# flattened seeds:")[0])

SEED=202609271
rg=np.random.default_rng(SEED)

# Marshall sign on checkerboard A sublattice. Global sign convention irrelevant.
Aidx=np.array([x+L*y for y in range(L) for x in range(L) if (x+y)%2==0],dtype=int)
def marshall_x(X):
    # X is +/-1; down=-1
    ndown=np.sum(X[:,Aidx]<0,axis=1)
    return np.where(ndown%2==0,1.,-1.).astype(np.float32)

def sample_a2(nchains=16,burn=160,between=6,nround=48):
    ss=[]
    for _ in range(nchains):
        pos=rg.choice(N,N//2,replace=False); q=0
        for i in pos:q|=1<<int(i)
        ss.append(q)
    la=evalz(bits2x(ss)).real
    out=[]
    total=burn+between*nround
    accepted=0
    for it in range(total):
        cand=[]
        for q in ss:
            up=[i for i in range(N) if (q>>i)&1]; dn=[i for i in range(N) if not((q>>i)&1)]
            i=int(rg.choice(up));j=int(rg.choice(dn));cand.append(q^(1<<i)^(1<<j))
        lb=evalz(bits2x(cand)).real
        ac=np.log(rg.random(nchains))<np.minimum(0,2.0*(lb-la))
        accepted+=int(np.sum(ac))
        for k in np.where(ac)[0]: ss[k]=cand[k]
        la=np.where(ac,lb,la)
        if it>=burn and (it-burn+1)%between==0: out.extend(ss)
    print('SAMPLE',len(out),'accept',accepted/(total*nchains),flush=True)
    return np.asarray(out,dtype=np.uint64)

def diagonal(s):
    e=0.
    for bb,J in ((NN,1.0),(NNN,J2)):
        for i,j in bb:
            e += J*(0.25 if (((s>>i)&1)==((s>>j)&1)) else -0.25)
    return e

def build_data(states):
    states=np.asarray(states,dtype=np.uint64)
    X=bits2x(states).astype(np.float32)
    zx=evalz(X); lax=zx.real
    neigh_lists=[]; maxm=0
    for s in states:
        a=list(neigh(int(s))); neigh_lists.append(a); maxm=max(maxm,len(a))
    flat=[]; offsets=[]
    for li in neigh_lists:
        offsets.append(len(flat)); flat.extend([q for q,J in li])
    unq,inv=np.unique(np.asarray(flat,dtype=np.uint64),return_inverse=True)
    zu=evalz(bits2x(unq)); lau=zu.real
    Xn=np.zeros((len(states),maxm,N),np.float32)
    C=np.zeros((len(states),maxm),np.float32)
    M=np.zeros((len(states),maxm),np.float32)
    m0=marshall_x(X)
    pos=0
    for i,li in enumerate(neigh_lists):
        for k,(q,J) in enumerate(li):
            jj=inv[pos]; pos+=1
            Xn[i,k]=bits2x(np.array([q],np.uint64))[0]
            C[i,k]=0.5*J*np.exp(lau[jj]-lax[i])
            M[i,k]=1.
    mn=marshall_x(Xn.reshape(-1,N)).reshape(len(states),maxm)
    MP=(m0[:,None]*mn).astype(np.float32)
    D=np.asarray([diagonal(int(s)) for s in states],np.float32)
    return dict(states=states,X=X,Xn=Xn,C=C,M=M,MP=MP,D=D,zx=zx,
                unq=unq,zu=zu,maxm=maxm)

allstates=sample_a2()
# deterministic chain-round split: every 4th held out
idx=np.arange(len(allstates))
va=(idx%4==0); tr=~va
train_states=allstates[tr][:512]
val_states=allstates[va][:256]
print('DATA train',len(train_states),'val',len(val_states),flush=True)
TR=build_data(train_states); VA=build_data(val_states)
np.savez_compressed('global_sign_dataset.npz',train_states=train_states,val_states=val_states)
print('DATASET_SHA_READY',flush=True)

class SignMLP(nn.Module):
    width:int=96
    @nn.compact
    def __call__(self,x):
        y=nn.Dense(self.width)(x); y=nn.tanh(y)
        y=nn.Dense(self.width)(y); y=nn.tanh(y)
        # Positive bias => exact Marshall correction +1 at initialization.
        y=nn.Dense(1,kernel_init=nn.initializers.zeros,
                   bias_init=lambda key,shape,dtype:jnp.ones(shape,dtype)*1.0)(y)
        return y[...,0]

model=SignMLP()
key=jax.random.PRNGKey(SEED)
params=model.init(key,jnp.asarray(TR['X']))['params']
opt=optax.chain(optax.clip_by_global_norm(5.0),optax.adam(2e-3))
ost=opt.init(params)

X=jnp.asarray(TR['X']); Xn=jnp.asarray(TR['Xn']); C=jnp.asarray(TR['C'])
Mask=jnp.asarray(TR['M']); MP=jnp.asarray(TR['MP']); D=jnp.asarray(TR['D'])

def ste_sign(q,T):
    soft=jnp.tanh(q/T)
    hard=jnp.where(q>=0,1.,-1.)
    return soft+jax.lax.stop_gradient(hard-soft)

def loss_fn(params,T):
    q0=model.apply({'params':params},X)
    qn=model.apply({'params':params},Xn.reshape(-1,N)).reshape(Xn.shape[:2])
    s0=ste_sign(q0,T); sn=ste_sign(qn,T)
    # Total sign ratio = Marshall ratio * correction ratio.
    el=D+jnp.sum(Mask*C*MP*(s0[:,None]*sn),axis=1)
    # Forward is the exact hard-sign local-energy estimator.
    return jnp.mean(el), (jnp.std(el),jnp.mean(jnp.abs(q0)),jnp.mean(q0<0))

@jax.jit
def step(params,ost,T):
    (loss,aux),gr=jax.value_and_grad(loss_fn,has_aux=True)(params,T)
    upd,ost=opt.update(gr,ost,params); params=optax.apply_updates(params,upd)
    return params,ost,loss,aux

def pred_corr(params,X,T=.5):
    q=np.asarray(model.apply({'params':params},jnp.asarray(X)))
    return np.where(q>=0,1.,-1.),q

def eval_energy(params,dat):
    s0,_=pred_corr(params,dat['X'])
    sn,_=pred_corr(params,dat['Xn'].reshape(-1,N))
    sn=sn.reshape(dat['Xn'].shape[:2])
    el=dat['D']+np.sum(dat['M']*dat['C']*dat['MP']*(s0[:,None]*sn),axis=1)
    return float(np.mean(el)),float(np.std(el)/np.sqrt(len(el)))

def hidden_scores(params,dat):
    # derive one global binary phase from held-out central states only
    z=dat['zx']; la=z.real; ww=np.exp(2*(la-la.max()))
    phi=.5*np.angle(np.sum(ww*np.exp(2j*z.imag)))
    h=np.where(np.cos(np.angle(np.exp(1j*(z.imag-phi))))>=0,1.,-1.)
    m=marshall_x(dat['X'])
    corr,_=pred_corr(params,dat['X'])
    pred=m*corr
    O=abs(np.sum(ww*pred*h)/np.sum(ww))
    Om=abs(np.sum(ww*m*h)/np.sum(ww))
    return float(O),float(Om),float(phi),float(np.mean(corr<0))

E0,se0=eval_energy(params,VA)
O0,OM,phi,f0=hidden_scores(params,VA)
print('INIT Eval',E0,se0,'O',O0,'MarshallO',OM,'flipfrac',f0,flush=True)

best=(1e99,None,None)
for it in range(1201):
    T=max(.35,1.5*(.997**it))
    params,ost,loss,aux=step(params,ost,jnp.asarray(T,np.float32))
    if it%50==0:
        Ev,sev=eval_energy(params,VA)
        O,_,_,ff=hidden_scores(params,VA)
        print('TRAIN',it,'T',T,'Etrain',float(loss),'Eval',Ev,'se',sev,
              'O',O,'flipfrac',ff,'qabs',float(aux[1]),flush=True)
        if Ev<best[0]: best=(Ev,jax.tree_util.tree_map(lambda x:np.array(x),params),it)

# restore best validation-energy checkpoint
if best[1] is not None:
    params=jax.tree_util.tree_map(jnp.asarray,best[1])
Ef,sef=eval_energy(params,VA); Of,OM,phi,ff=hidden_scores(params,VA)
print('FINAL beststep',best[2],'Eval',Ef,sef,'O',Of,'MarshallO',OM,'flipfrac',ff,'phi',phi,flush=True)
# Persist params and metadata.
from flax import serialization
with open('global_sign_params.msgpack','wb') as f: f.write(serialization.to_bytes(params))
with open('global_sign_result.json','w') as f:
    json.dump(dict(seed=SEED,E_init=E0,E_final=Ef,se_final=sef,O_final=Of,
                   O_marshall=OM,flipfrac=ff,beststep=int(best[2])),f,indent=2)
