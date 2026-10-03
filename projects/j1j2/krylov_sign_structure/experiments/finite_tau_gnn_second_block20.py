#!/usr/bin/env python3
import json, math, time
from pathlib import Path
from functools import partial
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla
import jax,jax.numpy as jnp
import optax
from flax import serialization
from flax.training.train_state import TrainState
from finite_tau_matching_20site_exact import build_H,basis,D,N,NN,NNN,state_to_idx
from finite_tau_gnn_densityratio_oracle20 import PairGNN,bits,make_wd,globals_from_wd

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5);HO=(H-sp.diags(diag)).tocsr()
dat=np.load(ROOT/'results/finite_tau_gfmc_uniform_block20.npz')
ys=dat['ys'].astype(int).tolist();train_y=dat['train_y'].astype(int).tolist()
rep1=dat['rep1'].astype(np.int32);rep2=dat['rep2'].astype(np.int32)
wd=make_wd(H,ys)
model=PairGNN();dummy=model.init(jax.random.PRNGKey(0),jnp.zeros((1,N)),jnp.zeros((1,N)),jnp.zeros((1,3)))['params']
old_params=serialization.from_bytes(dummy,(ROOT/'results/finite_tau_gnn_densityratio_walkers20.mpack').read_bytes())
apply_jit=jax.jit(lambda p,x,y,g:model.apply({'params':p},x,y,g))

def systematic(w,rng,M):
    w=np.asarray(w,float);w/=w.sum();c=np.cumsum(w);c[-1]=1
    return np.searchsorted(c,rng.random()/M+np.arange(M)/M,side='right')

class Guide:
    def __init__(self,y,k,tau,params):
        self.y=int(y);self.k=k;self.tau=tau;self.params=params;self.cache={}
        self.yb=bits(np.array([basis[y]]))[0]
    def eval(self,states):
        ss=np.asarray(states,np.int32).ravel(); out=np.empty(len(ss),float)
        uniq=np.unique(ss)
        miss=np.array([int(x) for x in uniq if int(x) not in self.cache],np.int32)
        for a in range(0,len(miss),4096):
            z=miss[a:a+4096];yy=np.repeat(self.yb[None,:],len(z),0)
            gg=globals_from_wd(wd[self.k],self.tau,z)
            vv=np.asarray(apply_jit(self.params,jnp.asarray(bits(basis[z])),jnp.asarray(yy),jnp.asarray(gg)))
            for x,v in zip(z,vv):self.cache[int(x)]=float(v)
        for i,x in enumerate(ss):out[i]=self.cache[int(x)]
        return out

def sign_for(k):
    d=(wd[k]//1000).astype(np.int64)
    return np.where(d%2==0,1,-1).astype(np.int8)

def propagate(start,guide,k,block=.1,tau_max=.005,seed=1):
    rg=np.random.default_rng(seed); walkers=np.asarray(start,np.int32).copy();M=len(walkers);sign=sign_for(k);lc={}
    def local(x):
        x=int(x)
        if x in lc:return lc[x]
        a,b=HO.indptr[x],HO.indptr[x+1];nb=HO.indices[a:b];hs=HO.data[a:b]
        vals=guide.eval(np.concatenate([[x],nb]));rat=np.exp(np.clip(vals[1:]-vals[0],-30,30))
        opp=sign[nb]!=sign[x];rates=hs[opp]*rat[opp];dfn=float(diag[x]+np.sum(hs[~opp]*rat[~opp]))
        lc[x]=(dfn,nb[opp].astype(np.int32),rates.astype(float));return lc[x]
    t=0.;steps=0
    while t<block-1e-14:
        datl=[local(x) for x in walkers];dfn=np.array([z[0] for z in datl]);el=np.array([z[0]-z[2].sum() for z in datl])
        Eref=float(el.mean());mx=max(0.,float(np.max(dfn-Eref)));dt=min(tau_max,block-t,0.8/mx if mx>0 else tau_max)
        nxt=np.empty(M,np.int32);bw=np.empty(M)
        for i,(x,(dv,nb,rate)) in enumerate(zip(walkers,datl)):
            stay=1-dt*(dv-Eref);mv=dt*rate;tot=stay+mv.sum();u=rg.random()*tot
            if u<stay:nxt[i]=x
            else:
                j=np.searchsorted(np.cumsum(mv),u-stay,side='right');nxt[i]=int(nb[min(j,len(nb)-1)])
            bw[i]=tot
        walkers=nxt[systematic(bw,rg,M)];t+=dt;steps+=1
    return walkers,steps,len(guide.cache),len(lc)
ALLB=np.asarray(NN+NNN,np.int32)
def sample_q(guide,starts,nout=4096,nchains=256,burn_sweeps=30,seed=1):
    rg=np.random.default_rng(seed); chains=np.asarray(starts[rg.integers(0,len(starts),size=nchains)],np.int32).copy()
    F=guide.eval(chains); collected=[]
    total_sweeps=burn_sweeps+int(math.ceil(nout/nchains))+4
    for sw in range(total_sweeps):
        for _ in range(N):
            bi=rg.integers(0,len(ALLB),size=nchains);uv=ALLB[bi]
            bs=basis[chains];u=uv[:,0].astype(np.uint32);v=uv[:,1].astype(np.uint32)
            bu=(bs>>u)&1;bv=(bs>>v)&1;flip=bu!=bv
            cand=chains.copy()
            if np.any(flip):
                mask=(np.left_shift(np.uint32(1),u[flip])|np.left_shift(np.uint32(1),v[flip]))
                cb=bs[flip]^mask;cand[flip]=state_to_idx[cb]
            Fc=guide.eval(cand);acc=np.log(rg.random(nchains))<np.minimum(0.,2*(Fc-F))
            chains[acc]=cand[acc];F[acc]=Fc[acc]
        if sw>=burn_sweeps:collected.append(chains.copy())
        if sum(len(x) for x in collected)>=nout:break
    out=np.concatenate(collected)[:nout]
    return out,len(np.unique(out)),len(guide.cache)

def reweight_start(pop,guide,M,seed):
    rg=np.random.default_rng(seed);F=guide.eval(pop);w=np.exp(np.clip(F-F.max(),-30,0))
    return np.asarray(pop[systematic(w,rg,M)],np.int32)

class DR2State(TrainState): bias:jnp.ndarray
@jax.jit
def train_step(st,oldp,x,y,gnew,gold,label,ctx):
    def lf(params,bias):
        fn=st.apply_fn({'params':params},x,y,gnew);fo=st.apply_fn({'params':oldp},x,y,gold)
        z=fn-fo+bias[ctx];return optax.sigmoid_binary_cross_entropy(z,label).mean()
    loss,(gp,gb)=jax.value_and_grad(lf,argnums=(0,1))(st.params,st.bias)
    st=st.apply_gradients(grads=gp);st=st.replace(bias=st.bias-5e-4*gb);return st,loss

def build_dataset(pos,qs,seed):
    XX=[];YY=[];GN=[];GO=[];LL=[];CC=[]
    M=pos.shape[1]
    for k,y in enumerate(train_y):
        ip=pos[k];iq=qs[k];idx=np.concatenate([ip,iq]);yb=bits(np.array([basis[y]]))[0]
        XX.append(bits(basis[idx]));YY.append(np.repeat(yb[None,:],2*M,0))
        GN.append(globals_from_wd(wd[k],.2,idx));GO.append(globals_from_wd(wd[k],.1,idx))
        LL.append(np.concatenate([np.ones(M,np.float32),np.zeros(M,np.float32)]));CC.append(np.full(2*M,k,np.int32))
    return tuple(np.concatenate(z) for z in (XX,YY,GN,GO,LL,CC))
def main():
    M=4096; pos=[];q1=[];posv=[];qv=[]
    for rep,pops,seedbase,PO,QO in [(1,rep1,51000,pos,q1),(2,rep2,61000,posv,qv)]:
      for k,y in enumerate(train_y):
        guide=Guide(y,k,.1,old_params);start=reweight_start(pops[k],guide,M,seedbase+k)
        mix,steps,ncache,nloc=propagate(start,guide,k,.1,.005,seedbase+100+k)
        qs,qu,qcache=sample_q(guide,start,M,256,30,seedbase+200+k)
        PO.append(mix);QO.append(qs)
        print("GEN rep",rep,"y",y,"mixuniq",len(np.unique(mix)),"quniq",qu,"cache",qcache,"local",nloc,flush=True)
    pos=np.stack(pos);q1=np.stack(q1);posv=np.stack(posv);qv=np.stack(qv)
    tr=build_dataset(pos,q1,1);va=build_dataset(posv,qv,2)
    st=DR2State.create(apply_fn=model.apply,params=old_params,tx=optax.adamw(5e-4,weight_decay=1e-6),bias=jnp.zeros(4))
    rg=np.random.default_rng(777);bs=4096;hist=[];bestv=1e9;best=None
    for ep in range(8):
        order=rg.permutation(len(tr[0]));ls=[]
        for i in range(0,len(order),bs):
            ix=order[i:i+bs];args=[jnp.asarray(z[ix]) for z in tr]
            st,l=train_step(st,old_params,*args);ls.append(float(l))
        vl=[]
        for i in range(0,len(va[0]),bs):
            x,y,gn,go,lab,ctx=[z[i:i+bs] for z in va]
            fn=np.asarray(model.apply({'params':st.params},jnp.asarray(x),jnp.asarray(y),jnp.asarray(gn)))
            fo=np.asarray(model.apply({'params':old_params},jnp.asarray(x),jnp.asarray(y),jnp.asarray(go)))
            zz=fn-fo+np.asarray(st.bias)[ctx];vl.append(np.mean(np.logaddexp(0,zz)-lab*zz))
        v=float(np.mean(vl));hist.append([ep,float(np.mean(ls)),v]);print("EPOCH",hist[-1],flush=True)
        if v<bestv:bestv=v;best=jax.tree_util.tree_map(lambda x:np.asarray(x),st.params)
    newp=best;(ROOT/'results/finite_tau_gnn_second_block20.mpack').write_bytes(serialization.to_bytes(newp))
    E=np.zeros((D,len(ys)));E[ys,np.arange(len(ys))]=1
    G=np.asarray(sla.expm_multiply(-.2*H,E),float);G/=np.linalg.norm(G,axis=0,keepdims=True)
    apply2=jax.jit(lambda p,x,y,g:model.apply({'params':p},x,y,g));rows=[]
    for k,y in enumerate(ys):
        yb=bits(np.array([basis[y]]))[0];vals=[]
        for a in range(0,D,4096):
            b=min(D,a+4096);idx=np.arange(a,b);yy=np.repeat(yb[None,:],b-a,0)
            vals.append(np.asarray(apply2(newp,jnp.asarray(bits(basis[idx])),jnp.asarray(yy),jnp.asarray(globals_from_wd(wd[k],.2,idx)))))
        F=np.concatenate(vals);aa=np.exp(np.clip(F-F.max(),-80,0));aa/=np.linalg.norm(aa);tt=np.abs(G[:,k]);tt/=np.linalg.norm(tt)
        row={'split':'train' if k<4 else 'test','y':int(y),'fidelity':float(np.dot(aa,tt)**2)};rows.append(row);print("RESULT",row,flush=True)
    out={'M':M,'history':hist,'rows':rows};path=ROOT/'results/finite_tau_gnn_second_block20.json';path.write_text(json.dumps(out,indent=2))
    np.savez_compressed(ROOT/'results/finite_tau_gnn_second_block20_samples.npz',ys=np.asarray(ys),pos=pos,q=q1,posv=posv,qv=qv)
    print("WROTE",path,flush=True)
if __name__=='__main__':main()
