import json, numpy as np
import jax, jax.numpy as jnp
from flax import linen as nn
from flax import serialization
import optax
L=6;N=36
ALPHA=.8;SEED=202609276
Aidx=np.array([x+L*y for y in range(L) for x in range(L) if (x+y)%2==0],dtype=int)
def marshall_x(X):
    ndown=np.sum(X[:,Aidx]<0,axis=1)
    return np.where(ndown%2==0,1.,-1.).astype(np.float32)

c=np.load('global_field_cache.npz')
def unpack(p):
    X=c['X'+p];Xn=c['Xn'+p];C=c['C'+p];M=c['M'+p];MP=c['MP'+p];D=c['D'+p];iw=c['iw'+p];zx=c['zx'+p];znim=c['znim'+p]
    return dict(X=X,Xn=Xn,C=C,M=M,MP=MP,D=D,iw=iw,zx=zx,znim=znim)
TR=unpack('tr');VA=unpack('va');TRN=len(TR['X'])
print('CACHE',TRN,len(VA['X']),'ESS',1/np.sum(TR['iw']**2),1/np.sum(VA['iw']**2),flush=True)

class Bridge(nn.Module):
    ch:int=20
    @nn.compact
    def __call__(self,x):
        y=x.reshape((-1,L,L,1))
        for _ in range(3):
            y=jnp.pad(y,((0,0),(1,1),(1,1),(0,0)),mode='wrap')
            y=nn.Conv(self.ch,(3,3),padding='VALID')(y);y=nn.tanh(y)
        y=jnp.mean(y,axis=(1,2))
        y=nn.Dense(20)(y);y=nn.tanh(y)
        # exact g=1 at initialization
        y=nn.Dense(1,kernel_init=nn.initializers.zeros,
                   bias_init=lambda key,shape,dtype:jnp.ones(shape,dtype))(y)
        return y[:,0]

model=Bridge();params=model.init(jax.random.PRNGKey(SEED),jnp.asarray(TR['X'][:4]))['params']
opt=optax.chain(optax.clip_by_global_norm(2.),optax.adamw(5e-4,weight_decay=1e-5));ost=opt.init(params)
rg=np.random.default_rng(SEED)
LAMBDA=.05; MU=8.0; B=48

@jax.jit
def step(params,ost,X,Xn,C,M,MP,D,W):
    def lf(p):
        g0=model.apply({'params':p},X)
        gn=model.apply({'params':p},Xn.reshape((-1,N))).reshape(Xn.shape[:2])
        local_num=D*g0*g0 + g0*jnp.sum(M*C*MP*gn,axis=1)
        num=TRN*jnp.mean(W*local_num)
        norm=TRN*jnp.mean(W*g0*g0)
        amp_pen=TRN*jnp.mean(W*(g0*g0-1.)**2)
        loss=num+MU*(norm-1.)**2+LAMBDA*amp_pen
        return loss,(num,norm,amp_pen,jnp.mean(g0<0),jnp.mean(jnp.abs(g0)))
    (loss,aux),gr=jax.value_and_grad(lf,has_aux=True)(params)
    upd,ost=opt.update(gr,ost,params);params=optax.apply_updates(params,upd)
    return params,ost,loss,aux

def gout(params,X,bs=4096):
    out=[]
    for i in range(0,len(X),bs):out.append(np.asarray(model.apply({'params':params},jnp.asarray(X[i:i+bs]))))
    return np.concatenate(out)

def eval_all(params,d):
    g0=gout(params,d['X']);gn=gout(params,d['Xn'].reshape(-1,N)).reshape(d['Xn'].shape[:2])
    w=d['iw'].astype(float)
    num=np.sum(w*(d['D']*g0*g0+g0*np.sum(d['M']*d['C']*d['MP']*gn,axis=1)))
    den=np.sum(w*g0*g0); Esoft=float(num/den)
    c0=np.where(g0>=0,1.,-1.);cn=np.where(gn>=0,1.,-1.)
    elh=d['D']+np.sum(d['M']*d['C']*d['MP']*(c0[:,None]*cn),axis=1)
    Ehard=float(np.sum(w*elh))
    # paired delta-to-Marshall uncertainty
    elm=d['D']+np.sum(d['M']*d['C']*d['MP'],axis=1)
    dd=elh-elm; dm=float(np.sum(w*dd)); ess=1/np.sum(w*w)
    se=float(np.sqrt(np.sum(w*(dd-dm)**2)/ess))
    ampdev=float(np.sqrt(np.sum(w*(np.abs(g0)-1.)**2)))
    # hidden diagnostic only
    z=d['zx'];phi=.5*np.angle(np.sum(w*np.exp(2j*z.imag)))
    h=np.where(np.cos(np.angle(np.exp(1j*(z.imag-phi))))>=0,1.,-1.)
    pred=marshall_x(d['X'])*c0
    O=abs(float(np.sum(w*pred*h)));acc=float(np.mean(pred==h))
    return dict(Esoft=Esoft,Ehard=Ehard,dEhard=dm,se_dE=se,ampdev=ampdev,O=O,acc=acc,flipfrac=float(np.mean(c0<0)),
                gmin=float(g0.min()),gmax=float(g0.max()),gnorm=float(den))

ev0=eval_all(params,VA);print('INIT',ev0,flush=True)
best=(ev0['Ehard'],serialization.to_bytes(params),0,ev0)
for it in range(1801):
    ids=rg.choice(TRN,B,replace=False)
    params,ost,loss,aux=step(params,ost,jnp.asarray(TR['X'][ids]),jnp.asarray(TR['Xn'][ids]),jnp.asarray(TR['C'][ids]),
        jnp.asarray(TR['M'][ids]),jnp.asarray(TR['MP'][ids]),jnp.asarray(TR['D'][ids]),jnp.asarray(TR['iw'][ids]))
    if it%50==0:
        ev=eval_all(params,VA)
        print('TRAIN',it,'loss',float(loss),'num',float(aux[0]),'norm',float(aux[1]),'pen',float(aux[2]),ev,flush=True)
        if ev['Ehard']<best[0]-1e-6:best=(ev['Ehard'],serialization.to_bytes(params),it,ev)
params=serialization.from_bytes(params,best[1]);ev=eval_all(params,VA)
print('FINAL beststep',best[2],ev,flush=True)
with open('global_bridge_params.msgpack','wb') as f:f.write(serialization.to_bytes(params))
with open('global_bridge_result.json','w') as f:json.dump(dict(beststep=int(best[2]),**ev),f,indent=2)
