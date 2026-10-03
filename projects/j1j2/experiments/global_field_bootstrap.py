import math,time,json
import numpy as np
import jax, jax.numpy as jnp
from flax import linen as nn
from flax import serialization
import optax
exec(open('bench_native_fast.py').read().split("# flattened seeds:")[0])

ALPHA=.8; SEED=202609275
Aidx=np.array([x+L*y for y in range(L) for x in range(L) if (x+y)%2==0],dtype=int)
def marshall_x(X):
    ndown=np.sum(X[:,Aidx]<0,axis=1)
    return np.where(ndown%2==0,1.,-1.).astype(np.float32)
def diagonal(s):
    e=0.
    for bb,J in ((NN,1.0),(NNN,J2)):
        for i,j in bb:e+=J*(.25 if (((s>>i)&1)==((s>>j)&1)) else -.25)
    return e

def build_data(states,label):
    states=np.asarray(states,np.uint64);X=bits2x(states).astype(np.float32)
    zx=evalz(X);lax=zx.real
    lists=[];flat=[];maxm=0
    for s in states:
        li=list(neigh(int(s)));lists.append(li);flat.extend([q for q,J in li]);maxm=max(maxm,len(li))
    unq,inv=np.unique(np.asarray(flat,np.uint64),return_inverse=True)
    print(label,'central',len(states),'occ',len(flat),'unique',len(unq),'maxm',maxm,flush=True)
    zu=evalz(bits2x(unq));lau=zu.real
    Xn=np.zeros((len(states),maxm,N),np.float32);C=np.zeros((len(states),maxm),np.float32);M=np.zeros_like(C)
    znim=np.zeros((len(states),maxm),np.float64);m0=marshall_x(X);pos=0
    for i,li in enumerate(lists):
        for k,(q,J) in enumerate(li):
            jj=inv[pos];pos+=1;Xn[i,k]=bits2x(np.array([q],np.uint64))[0]
            C[i,k]=.5*J*np.exp(lau[jj]-lax[i]);M[i,k]=1.;znim[i,k]=zu[jj].imag
    mn=marshall_x(Xn.reshape(-1,N)).reshape(len(states),maxm)
    MP=(m0[:,None]*mn).astype(np.float32)
    D=np.asarray([diagonal(int(s)) for s in states],np.float32)
    lw=(2-ALPHA)*lax;lw-=lw.max();iw=np.exp(lw);iw/=iw.sum()
    print(label,'ESS',1/np.sum(iw*iw),flush=True)
    return dict(X=X,Xn=Xn,C=C,M=M,MP=MP,D=D,zx=zx,znim=znim,iw=iw.astype(np.float32),m0=m0,mn=mn)

st=np.load('global_ticnn_states.npz'); TR=build_data(st['train_states'],'TRAIN'); VA=build_data(st['val_states'],'VAL')
# cache the expensive neighbour tensors
np.savez_compressed('global_field_cache.npz',
    Xtr=TR['X'],Xntr=TR['Xn'],Ctr=TR['C'],Mtr=TR['M'],MPtr=TR['MP'],Dtr=TR['D'],iwtr=TR['iw'],zxtr=TR['zx'],znimtr=TR['znim'],
    Xva=VA['X'],Xnva=VA['Xn'],Cva=VA['C'],Mva=VA['M'],MPva=VA['MP'],Dva=VA['D'],iwva=VA['iw'],zxva=VA['zx'],znimva=VA['znim'])

class TICNN(nn.Module):
    ch:int=20
    @nn.compact
    def __call__(self,x):
        y=x.reshape((-1,L,L,1))
        for _ in range(3):
            y=jnp.pad(y,((0,0),(1,1),(1,1),(0,0)),mode='wrap')
            y=nn.Conv(self.ch,(3,3),padding='VALID')(y);y=nn.tanh(y)
        y=jnp.mean(y,axis=(1,2))
        y=nn.Dense(20)(y);y=nn.tanh(y)
        y=nn.Dense(1,kernel_init=nn.initializers.zeros,
                   bias_init=lambda key,shape,dtype:jnp.ones(shape,dtype)*2.0)(y)
        return y[:,0]
model=TICNN();params=model.init(jax.random.PRNGKey(SEED),jnp.asarray(TR['X'][:4]))['params']
opt=optax.chain(optax.clip_by_global_norm(2.),optax.adamw(1e-3,weight_decay=2e-5));ost=opt.init(params)
rg=np.random.default_rng(SEED)

def corr(params,X,bs=4096):
    out=[]
    for i in range(0,len(X),bs):
        q=np.asarray(model.apply({'params':params},jnp.asarray(X[i:i+bs])))
        out.append(np.where(q>=0,1.,-1.))
    return np.concatenate(out).astype(np.float32)

def current(params,d):
    c0=corr(params,d['X']);cn=corr(params,d['Xn'].reshape(-1,N)).reshape(d['Xn'].shape[:2])
    s0=d['m0']*c0;sn=d['mn']*cn
    field=np.sum(d['M']*d['C']*sn,axis=1)
    starget=np.where(field>0,-1.,1.)
    tcor=starget*d['m0']
    # Confidence: exact single-spin energy scale, clipped for robust fitting.
    conf=np.abs(field); cap=np.quantile(conf,.95); conf=np.minimum(conf,cap)
    wt=d['iw']*conf; wt/=wt.sum()
    el=d['D']+np.sum(d['M']*d['C']*(s0[:,None]*sn),axis=1)
    E=float(np.sum(d['iw']*el))
    return c0,cn,tcor,wt,E,field

def hidden(d,c0):
    w=d['iw'].astype(float);z=d['zx'];phi=.5*np.angle(np.sum(w*np.exp(2j*z.imag)))
    h=np.where(np.cos(np.angle(np.exp(1j*(z.imag-phi))))>=0,1.,-1.)
    truecorr=h*d['m0'];pred=d['m0']*c0
    return dict(O=abs(float(np.sum(w*pred*h))),corr_acc=float(np.mean(c0==truecorr)),
                nonM=float(np.mean(truecorr<0)),phi=float(phi),h=h,truecorr=truecorr)

@jax.jit
def sup_step(params,ost,X,Y,W):
    def lf(p):
        q=model.apply({'params':p},X)
        l=optax.softplus(-Y*q)
        return X.shape[0]*jnp.mean(W*l)
    loss,gr=jax.value_and_grad(lf)(params)
    upd,ost=opt.update(gr,ost,params);params=optax.apply_updates(params,upd)
    return params,ost,loss

def fit_targets(params,ost,d,Y,W,steps=350,batch=64):
    n=len(Y)
    for k in range(steps):
        ids=rg.choice(n,batch,replace=False)
        params,ost,loss=sup_step(params,ost,jnp.asarray(d['X'][ids]),jnp.asarray(Y[ids]),jnp.asarray(W[ids]))
    return params,ost,float(loss)

# Initial physics target quality before any fitting.
for name,d in [('TRAIN',TR),('VAL',VA)]:
    c0,cn,t,w,E,f=current(params,d); hm=hidden(d,c0)
    target_agree=float(np.mean(t==hm['truecorr']))
    target_weighted=float(np.sum(d['iw']*(t==hm['truecorr'])))
    print('OUTER_INIT',name,'E',E,'modelO',hm['O'],'modelCorrAcc',hm['corr_acc'],
          'targetFlipFrac',float(np.mean(t<0)),'targetHiddenAgree',target_agree,
          'targetHiddenAgreePhys',target_weighted,'fieldMedian',float(np.median(np.abs(f))),flush=True)

history=[]
best=(1e99,serialization.to_bytes(params),-1)
for outer in range(10):
    c0,cn,t,w,Etr,field=current(params,TR)
    params,ost,last=fit_targets(params,ost,TR,t,w,steps=350,batch=64)
    ctr,_,_,_,Etr2,_=current(params,TR)
    cv,_,tv,wv,Ev,fv=current(params,VA);hm=hidden(VA,cv)
    target_agree=float(np.mean(tv==hm['truecorr']))
    target_phys=float(np.sum(VA['iw']*(tv==hm['truecorr'])))
    print('OUTER',outer,'Etrain_before',Etr,'Etrain_after',Etr2,'Eval',Ev,
          'O',hm['O'],'corr_acc',hm['corr_acc'],'flipfrac',float(np.mean(cv<0)),
          'nextTargetFlip',float(np.mean(tv<0)),'nextTargetHiddenAgree',target_agree,
          'nextTargetHiddenAgreePhys',target_phys,'supLoss',last,flush=True)
    history.append(dict(outer=outer,Etrain=Etr2,Eval=Ev,O=hm['O'],corr_acc=hm['corr_acc'],
                        flipfrac=float(np.mean(cv<0)),targetHiddenAgree=target_agree,targetHiddenAgreePhys=target_phys))
    if Ev<best[0]:best=(Ev,serialization.to_bytes(params),outer)
    # convergence in model signs on train: targets equal predictions almost everywhere
    if np.mean(ctr==t)>0.995:
        print('CONVERGED',outer,float(np.mean(ctr==t)),flush=True);break
params=serialization.from_bytes(params,best[1])
cv,_,tv,wv,Ev,fv=current(params,VA);hm=hidden(VA,cv)
print('FINAL bestouter',best[2],'Eval',Ev,'metrics', {k:v for k,v in hm.items() if k not in ('h','truecorr')},flush=True)
with open('global_field_params.msgpack','wb') as f:f.write(serialization.to_bytes(params))
with open('global_field_result.json','w') as f:json.dump(dict(bestouter=int(best[2]),Eval=Ev,O=hm['O'],corr_acc=hm['corr_acc'],history=history),f,indent=2)
