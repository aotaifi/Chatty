import math,time,json
import numpy as np
import jax, jax.numpy as jnp
from flax import linen as nn
from flax import serialization
import optax
exec(open('bench_native_fast.py').read().split("# flattened seeds:")[0])

ALPHA=0.8
SEED=202609273
Aidx=np.array([x+L*y for y in range(L) for x in range(L) if (x+y)%2==0],dtype=int)

def marshall_x(X):
    ndown=np.sum(X[:,Aidx]<0,axis=1)
    return np.where(ndown%2==0,1.,-1.).astype(np.float32)

def sample_alpha(seed,nchains,burn,between,nround):
    rg=np.random.default_rng(seed)
    ss=[]
    for _ in range(nchains):
        pos=rg.choice(N,N//2,replace=False);q=0
        for i in pos:q|=1<<int(i)
        ss.append(q)
    la=evalz(bits2x(ss)).real
    out=[];accn=0;tot=0
    for it in range(burn+between*nround):
        cand=[]
        for q in ss:
            up=[i for i in range(N) if (q>>i)&1]
            dn=[i for i in range(N) if not((q>>i)&1)]
            i=int(rg.choice(up));j=int(rg.choice(dn))
            cand.append(q^(1<<i)^(1<<j))
        lb=evalz(bits2x(cand)).real
        ac=np.log(rg.random(nchains))<np.minimum(0,ALPHA*(lb-la))
        accn+=int(np.sum(ac));tot+=nchains
        for k in np.where(ac)[0]:ss[k]=cand[k]
        la=np.where(ac,lb,la)
        if it>=burn and (it-burn+1)%between==0:out.extend(ss)
    print('SAMPLE',seed,len(out),'accept',accn/tot,flush=True)
    return np.asarray(out,np.uint64)

def diagonal(s):
    e=0.
    for bb,J in ((NN,1.0),(NNN,J2)):
        for i,j in bb:
            e+=J*(0.25 if (((s>>i)&1)==((s>>j)&1)) else -0.25)
    return e

def build_data(states,label):
    states=np.asarray(states,np.uint64); X=bits2x(states).astype(np.float32)
    zx=evalz(X); lax=zx.real
    lists=[];maxm=0;flat=[]
    for s in states:
        li=list(neigh(int(s)));lists.append(li);maxm=max(maxm,len(li))
        flat.extend([q for q,J in li])
    unq,inv=np.unique(np.asarray(flat,np.uint64),return_inverse=True)
    print(label,'central',len(states),'neighbor occurrences',len(flat),'unique',len(unq),'maxm',maxm,flush=True)
    zu=evalz(bits2x(unq)); lau=zu.real
    Xn=np.zeros((len(states),maxm,N),np.float32)
    C=np.zeros((len(states),maxm),np.float32); mask=np.zeros_like(C)
    m0=marshall_x(X); pos=0
    zneigh_re=np.zeros((len(states),maxm),np.float64)
    zneigh_im=np.zeros((len(states),maxm),np.float64)
    for i,li in enumerate(lists):
        for k,(q,J) in enumerate(li):
            jj=inv[pos];pos+=1
            Xn[i,k]=bits2x(np.array([q],np.uint64))[0]
            C[i,k]=0.5*J*np.exp(lau[jj]-lax[i]);mask[i,k]=1.
            zneigh_re[i,k]=zu[jj].real; zneigh_im[i,k]=zu[jj].imag
    mn=marshall_x(Xn.reshape(-1,N)).reshape(len(states),maxm)
    MP=(m0[:,None]*mn).astype(np.float32)
    D=np.asarray([diagonal(int(s)) for s in states],np.float32)
    # importance weights q_alpha -> physical a^2
    lw=(2-ALPHA)*lax; lw-=np.max(lw); iw=np.exp(lw); iw/=iw.sum()
    ess=1/np.sum(iw*iw)
    print(label,'ESS',ess,'ESSfrac',ess/len(iw),flush=True)
    return dict(states=states,X=X,Xn=Xn,C=C,M=mask,MP=MP,D=D,zx=zx,
                znre=zneigh_re,znim=zneigh_im,iw=iw.astype(np.float32))

# independent chains for train and validation
train_states=sample_alpha(SEED,16,140,6,128)   # 2048
val_states=sample_alpha(SEED+1,16,140,6,32)    # 512
TR=build_data(train_states,'TRAIN')
VA=build_data(val_states,'VAL')
np.savez_compressed('global_ticnn_states.npz',train_states=train_states,val_states=val_states)

class TICNN(nn.Module):
    ch:int=24
    @nn.compact
    def __call__(self,x):
        y=x.reshape((-1,L,L,1))
        for _ in range(3):
            y=jnp.pad(y,((0,0),(1,1),(1,1),(0,0)),mode='wrap')
            y=nn.Conv(self.ch,(3,3),padding='VALID')(y)
            y=nn.tanh(y)
        y=jnp.mean(y,axis=(1,2))
        y=nn.Dense(24)(y);y=nn.tanh(y)
        y=nn.Dense(1,kernel_init=nn.initializers.zeros,
                   bias_init=lambda key,shape,dtype:jnp.ones(shape,dtype)*1.0)(y)
        return y[:,0]

model=TICNN()
key=jax.random.PRNGKey(SEED)
params=model.init(key,jnp.asarray(TR['X'][:4]))['params']
opt=optax.chain(optax.clip_by_global_norm(2.0),optax.adamw(7e-4,weight_decay=2e-5))
ost=opt.init(params)
rg=np.random.default_rng(SEED+2)

def ste_sign(q,T):
    soft=jnp.tanh(q/T);hard=jnp.where(q>=0,1.,-1.)
    return soft+jax.lax.stop_gradient(hard-soft)

@jax.jit
def batch_loss(params,X,Xn,C,M,MP,D,W,T):
    q0=model.apply({'params':params},X)
    qn=model.apply({'params':params},Xn.reshape((-1,N))).reshape(Xn.shape[:2])
    s0=ste_sign(q0,T);sn=ste_sign(qn,T)
    el=D+jnp.sum(M*C*MP*(s0[:,None]*sn),axis=1)
    # W is global normalized physical importance weight. Multiply by total train N
    # so uniform mini-batch expectation equals weighted physical energy.
    loss=X.shape[0]*0.0 + TRN*jnp.mean(W*el)
    # tiny correction sparsity prior; forward physics still exact hard signs
    pflip=jax.nn.sigmoid(-q0/T)
    loss=loss+2e-3*TRN*jnp.mean(W*pflip)
    return loss,(TRN*jnp.mean(W*el),jnp.mean(q0<0),jnp.mean(jnp.abs(q0)))

@jax.jit
def step(params,ost,X,Xn,C,M,MP,D,W,T):
    (loss,aux),gr=jax.value_and_grad(batch_loss,has_aux=True)(params,X,Xn,C,M,MP,D,W,T)
    upd,ost=opt.update(gr,ost,params);params=optax.apply_updates(params,upd)
    return params,ost,loss,aux

def corr(params,X,bs=4096):
    out=[]
    for i in range(0,len(X),bs):
        q=np.asarray(model.apply({'params':params},jnp.asarray(X[i:i+bs])))
        out.append(np.where(q>=0,1.,-1.)); 
    return np.concatenate(out)

def eval_model(params,dat):
    s0=corr(params,dat['X'])
    sn=corr(params,dat['Xn'].reshape(-1,N)).reshape(dat['Xn'].shape[:2])
    el=dat['D']+np.sum(dat['M']*dat['C']*dat['MP']*(s0[:,None]*sn),axis=1)
    E=float(np.sum(dat['iw']*el))
    # weighted standard-error proxy using effective sample size
    mu=E; var=float(np.sum(dat['iw']*(el-mu)**2)); ess=1/float(np.sum(dat['iw']**2))
    se=(var/ess)**0.5
    return E,se,s0,sn,el

def hidden_metrics(params,dat):
    w=dat['iw'].astype(float)
    z=dat['zx']
    phi=.5*np.angle(np.sum(w*np.exp(2j*z.imag)))
    h0=np.where(np.cos(np.angle(np.exp(1j*(z.imag-phi))))>=0,1.,-1.)
    hn=np.where(np.cos(np.angle(np.exp(1j*(dat['znim']-phi))))>=0,1.,-1.)
    m0=marshall_x(dat['X'])
    mn=marshall_x(dat['Xn'].reshape(-1,N)).reshape(dat['Xn'].shape[:2])
    sc=corr(params,dat['X']); pred=m0*sc
    Ow=abs(float(np.sum(w*pred*h0)))
    acc=float(np.mean(pred==h0))
    corr_true=h0*m0
    corr_pred=sc
    corr_acc=float(np.mean(corr_pred==corr_true))
    nonm=float(np.mean(corr_true<0))
    # hidden fixed-amplitude energy diagnostic
    hel=dat['D']+np.sum(dat['M']*dat['C']*(h0[:,None]*hn),axis=1)
    Eh=float(np.sum(w*hel))
    mel=dat['D']+np.sum(dat['M']*dat['C']*dat['MP'],axis=1)
    Em=float(np.sum(w*mel))
    return dict(Ow=Ow,acc=acc,corr_acc=corr_acc,nonM=nonm,phi=float(phi),Ehidden=Eh,Emarshall=Em)

TRN=len(TR['X'])
B=48
E0,se0,_,_,_=eval_model(params,VA); hm0=hidden_metrics(params,VA)
print('INIT',E0,se0,hm0,flush=True)
best=(E0,serialization.to_bytes(params),0)
for it in range(1601):
    ids=rg.choice(TRN,B,replace=False)
    T=max(.45,1.5*(.9985**it))
    params,ost,loss,aux=step(params,ost,
        jnp.asarray(TR['X'][ids]),jnp.asarray(TR['Xn'][ids]),jnp.asarray(TR['C'][ids]),
        jnp.asarray(TR['M'][ids]),jnp.asarray(TR['MP'][ids]),jnp.asarray(TR['D'][ids]),
        jnp.asarray(TR['iw'][ids]),jnp.asarray(T,np.float32))
    if it%50==0:
        Ev,sev,_,_,_=eval_model(params,VA); hm=hidden_metrics(params,VA)
        print('TRAIN',it,'T',T,'loss',float(loss),'Eval',Ev,'se',sev,
              'O',hm['Ow'],'corr_acc',hm['corr_acc'],'nonM',hm['nonM'],
              'flipfrac',float(aux[1]),'Ehidden',hm['Ehidden'],'Emarshall',hm['Emarshall'],flush=True)
        if Ev<best[0]-1e-5:
            best=(Ev,serialization.to_bytes(params),it)
params=serialization.from_bytes(params,best[1])
Ef,sef,_,_,_=eval_model(params,VA); hmf=hidden_metrics(params,VA)
print('FINAL beststep',best[2],'Eval',Ef,'se',sef,'metrics',hmf,flush=True)
with open('global_ticnn_params.msgpack','wb') as f:f.write(serialization.to_bytes(params))
with open('global_ticnn_result.json','w') as f:
    json.dump(dict(alpha=ALPHA,seed=SEED,E_init=E0,E_final=Ef,se_final=sef,beststep=int(best[2]),**hmf),f,indent=2)
