#!/usr/bin/env python3
"""20-site no-FN-label amplitude learning benchmark.

FN positive projector uses ONLY a callable analytic Jastrow guide g0(x), known signs,
H diagonal and locally evaluated Hamiltonian neighbors. Mixed walkers produce f~g0*phi_FN;
noise-contrastive density-ratio estimation vs independent g0^2 Metropolis draws
learns log(phi_FN/g0) without exact FN wavefunctions/labels.

The 20-site full configuration lookup and ED/FN target arrays are permitted solely
as geometry acceleration / final post-hoc evaluation. This is a prototype: the
full-space lookup and stored sign table must become callable for scaling.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
from collections import defaultdict
import argparse,json,time
import numpy as np
import torch
from torch import nn
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import j1j2_local_query_diagnostic as M
import j1j2_budgeted_orbit20 as B
import j1j2_neural_guide20 as T

P=Path('/mnt/data'); D=M.D; N=M.N; torch.set_num_threads(2)
MASKS=np.array([int(b[0]) for b in M.bonds],np.int32)
HOPS=np.array([b[1] for b in M.bonds],np.float64)
I=np.array([b[2] for b in M.bonds],np.int32)
J=np.array([b[3] for b in M.bonds],np.int32)
NB=len(MASKS)
ALPHA=float(os.environ.get('JASTROW_ALPHA','0.30'))
# The diagonal energy is computable from the spin state; stored 20-site vector is
# a benchmark acceleration and must not be construed as an exact FN amplitude.
LOGG0=(-ALPHA*M.diag).astype(np.float64)
G0=np.exp(LOGG0)
# +/- guide signs are part of the starting point, not ED sign labels.
SIGNS=M.signs

class CountedGuide:
 def __init__(self,model=None,scale=1.0):
  self.model=model;self.scale=scale;self.calls=0;self.cache={}
 def logamp(self,config):
  v=int(config)
  if v not in self.cache:
   ix=int(M.lookup[v]);assert ix>=0
   z=float(LOGG0[ix])
   if self.model is not None:
    with torch.no_grad():
     z+=float(self.scale*self.model.symmetric(T.X[ix:ix+1]).item())
   self.cache[v]=z;self.calls+=1
  return self.cache[v]

# vectorized local-row construction. Rows are calculated ON DEMAND for visited states,
# stored in cache. Each unique row visits only NB Hamiltonian-connected configurations.
class FNProjector:
 def __init__(self,dt=.01,Eref=-10.0):
  self.dt=dt;self.Eref=Eref;self.row_map=np.full(1<<N,-1,np.int32)
  self.rows=[];self.cdf=[];self.z=[];self.visited=set();self.neighbor_probes=0
 def add_rows(self,configs):
  miss=np.unique(configs[self.row_map[configs]<0]);
  if not len(miss):return
  idx=M.lookup[miss];assert (idx>=0).all()
  ys=miss[:,None]^MASKS[None,:]
  valid=(((miss[:,None]>>I[None,:])^(miss[:,None]>>J[None,:]))&1).astype(bool)
  iy=M.lookup[ys];safe=np.maximum(iy,0)
  k=HOPS[None,:]*SIGNS[idx,None]*SIGNS[safe]
  gr=np.exp(LOGG0[safe]-LOGG0[idx,None]);gr=np.where(valid,gr,0.)
  woff=np.where((k<0)&valid,self.dt*(-k)*gr,0.)
  fdiag=M.diag[idx]+np.sum(np.where((k>0)&valid,k*gr,0.),axis=1)
  wstay=1-self.dt*(fdiag-self.Eref)
  if np.any(wstay<=0):raise ValueError('Time-step too large: negative diagonal; min='+str(wstay.min()))
  roww=np.concatenate([wstay[:,None],woff],axis=1)
  z=roww.sum(1)
  if np.any(z<=0):raise ValueError('negative projector weight')
  cdf=np.cumsum(roww/z[:,None],axis=1);cdf[:,-1]=1.
  rid=np.arange(len(self.rows),len(self.rows)+len(miss))
  self.row_map[miss]=rid
  for i,x in enumerate(miss):
   self.rows.append(np.r_[int(x),ys[i]].astype(np.int32));self.cdf.append(cdf[i].astype(np.float32));self.z.append(float(z[i]));self.visited.add(int(x))
  self.neighbor_probes+=int(len(miss)*NB)
 def run(self,seed=13,nwalk=4000,burn=450,steps=500,thin=4):
  rng=np.random.default_rng(seed)
  x=rng.choice(M.states,size=nwalk,replace=True).astype(np.int32)
  pos=[];history=[];accepted=0
  for it in range(burn+steps):
   self.add_rows(x)
   rid=self.row_map[x]
   # CDF lookup on only walker-accessible visited rows.
   batch_cdf=np.stack([self.cdf[int(j)] for j in rid])
   u=rng.random(nwalk)
   col=np.sum(u[:,None]>batch_cdf,axis=1)
   rowlist=self.rows
   yy=np.fromiter((rowlist[int(j)][int(k)] for j,k in zip(rid,col)),dtype=np.int32,count=nwalk)
   accepted+=int(np.sum(yy!=x))
   weights=np.fromiter((self.z[int(j)] for j in rid),dtype=np.float64,count=nwalk)
   cw=np.cumsum(weights);ref=rng.random()/nwalk
   # systematic resampling each projector step
   choices=np.searchsorted(cw,cw[-1]*(ref+np.arange(nwalk)/nwalk),side='left')
   x=yy[np.minimum(choices,nwalk-1)]
   if it>=burn and (it-burn)%thin==0:pos.append(x.copy())
   if it in (0,100,burn-1,burn+steps-1):history.append(dict(step=it+1,distinct_walkers=int(len(np.unique(x))),mean_diag=float(M.diag[M.lookup[x]].mean()),mean_branch=float(weights.mean())))
  samples=np.concatenate(pos)
  return samples,dict(seed=seed,nwalk=nwalk,burn=burn,steps=steps,thin=thin,raw_samples=int(len(samples)),unique_samples=int(len(np.unique(samples))),unique_projector_rows=len(self.rows),hamiltonian_neighbor_probes=self.neighbor_probes,moves_fraction=accepted/(nwalk*(burn+steps)),history=history)

def sample_reference(seed=22,nchains=4000,burn=1200,steps=500,thin=4):
 """Independent g0^2 Markov chains; no FN amplitudes or normalization."""
 rng=np.random.default_rng(seed);x=rng.choice(M.states,size=nchains,replace=True).astype(np.int32)
 samples=[];acc=0
 for it in range(burn+steps):
  j=rng.integers(NB,size=nchains)
  y=x^MASKS[j];valid=(((x>>I[j])^(x>>J[j]))&1).astype(bool)
  ix=M.lookup[x];iy=np.maximum(M.lookup[y],0)
  ratio=np.exp(2*(LOGG0[iy]-LOGG0[ix]))
  accept=valid&(rng.random(nchains)<np.minimum(1.,ratio));x=np.where(accept,y,x);acc+=int(accept.sum())
  if it>=burn and (it-burn)%thin==0:samples.append(x.copy())
 z=np.concatenate(samples)
 return z,dict(raw_samples=len(z),unique_samples=len(np.unique(z)),nchains=nchains,acceptance=acc/(nchains*(burn+steps)),burn=burn,steps=steps)

def train_ratio(pos,neg,steps=900,seed=81):
 """NCE: pos f~g0*phi_FN; neg q~g0^2. log(f/q)=log(phi_FN/g0)+const."""
 torch.manual_seed(seed);rng=np.random.default_rng(seed)
 net=T.Net();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-4)
 # Deduplicated symmetry orbit split for diagnostics; avoid splitting symmetry copies.
 orbit=M.Sampler();upos=np.unique(pos);uneg=np.unique(neg)
 positive_orbits=np.array([orbit.orbit(int(x)) for x in upos]);negative_orbits=np.array([orbit.orbit(int(x)) for x in uneg]);allorbits=np.unique(np.r_[positive_orbits,negative_orbits]);rng.shuffle(allorbits);held=set(allorbits[:max(1,len(allorbits)//10)])
 pm=np.array([orbit.orbit(int(x)) not in held for x in pos]);nm=np.array([orbit.orbit(int(x)) not in held for x in neg]);ptr=pos[pm];ntr=neg[nm];pva=pos[~pm];nva=neg[~nm]
 if len(ptr)<200 or len(ntr)<200: raise ValueError('Too few train samples')
 trace=[];start=time.monotonic()
 for step in range(steps):
  a=ptr[rng.integers(len(ptr),size=128)];b=ntr[rng.integers(len(ntr),size=128)]
  ids=np.r_[M.lookup[a],M.lookup[b]]
  logits=net.symmetric(T.X[ids]);lab=torch.cat([torch.ones(128),torch.zeros(128)])
  loss=nn.functional.binary_cross_entropy_with_logits(logits,lab)
  opt.zero_grad();loss.backward();opt.step()
  if (step+1)%300==0:
   with torch.no_grad():
    na=min(len(pva),300);nb=min(len(nva),300)
    if na and nb:
     z=net.symmetric(T.X[M.lookup[np.r_[pva[:na],nva[:nb]]]])
     vl=nn.functional.binary_cross_entropy_with_logits(z,torch.cat([torch.ones(na),torch.zeros(nb)])).item()
    else:vl=float('nan')
   trace.append(dict(step=step+1,train_bce=float(loss.item()),heldout_bce=float(vl)))
   print('TRAIN',trace[-1],flush=True)
 net.eval()
 return net,dict(method='mixed-walker versus Jastrow-squared noise-contrastive estimation',steps=steps,training_positive=int(len(ptr)),training_negative=int(len(ntr)),heldout_positive=int(len(pva)),heldout_negative=int(len(nva)),unique_sampled_orbits=int(len(allorbits)),trace=trace,elapsed_s=round(time.monotonic()-start,2))

class LocalModelSampler(M.Sampler):
 def __init__(self,guide):
  super().__init__();self.guide=guide
 def amp(self,x):
  x=int(x);self.n_amp_requests+=1
  if x not in self.amps:self.amps[x]=float(np.exp(np.clip(self.guide.logamp(x),-40,25)))
  return self.amps[x]

def eval_trials(net,seedrange=range(301,305),B_orbits=300,limit=40):
 """Model vs analytic Jastrow baseline with same pool from fitted-model walkers."""
 results=[]
 for seed in seedrange:
  rng=np.random.default_rng(seed);fguide=CountedGuide(net);s=LocalModelSampler(fguide)
  x=int(M.states[int(rng.integers(D))]);seeds=[];burn=2000;nseed=100;thin=15
  for it in range(burn+nseed*thin):
   x,_=s.propose(x,rng)
   if it>=burn and (it-burn)%thin==0:seeds.append(x)
  uniq=set(seeds);pool=sorted({s.orbit(x) for x in (set(seeds)|{y for x in uniq for y,_ in s.neighbors(x)})});qs0=len(s.amps)
  proxy={g:0. for g in pool};cnt={g:0 for g in pool}
  for x in uniq:
   ax=s.amp(x);sx=int(SIGNS[s.idx(x)])
   for y,h in s.neighbors(x):
    gy=s.orbit(y);proxy[gy]+=h*sx*int(SIGNS[s.idx(y)])*ax*s.amp(y);cnt[gy]+=1
  eligible=sorted(pool,key=lambda g:(-proxy[g]/max(1,cnt[g]),g))[:B_orbits]
  arms={}
  for method in ('fitted','jastrow'):
   if method=='fitted':sg=s;guide=fguide
   else:guide=CountedGuide();sg=LocalModelSampler(guide);sg.keys=s.keys.copy();sg.members=s.members.copy()
   raw={};edges={}
   for g in eligible:raw[g],edges[g]=B.exact_score_one(sg,g)
   selected=set();tra=[]
   for _ in range(min(limit,len(eligible))):
    gains={g:raw[g]-8*sum(v for h,v in edges[g].items() if h in selected) for g in eligible if g not in selected}
    if not gains:break
    g=max(gains,key=gains.get)
    if gains[g]<=0:break
    selected.add(g);tra.append(float(gains[g]))
   # scoring using current model amplitudes; true FN evaluation done separately
   # Values here *guaranteed* by construction, not an independent discovery.
   pred=sum(tra)
   selected_states=np.concatenate([np.array(s.members[g],dtype=np.int32) for g in selected]) if selected else np.array([],dtype=np.int32)
   ix=M.lookup[selected_states]
   # ED exact ground state sign is ONLY post hoc (does not train, propose or gate)
   correct=float(np.sum((M.psi0[ix]*SIGNS[ix]<0)*M.psi0[ix]**2)) if len(ix) else 0.
   wrong=float(np.sum((M.psi0[ix]*SIGNS[ix]>0)*M.psi0[ix]**2)) if len(ix) else 0.
   arms[method]=dict(n_selected_orbits=len(selected),n_selected_states=len(selected_states),selected_keys=list(map(int,sorted(selected))),guide_predicted_gain=pred,ED_weight_improvement=correct-wrong,unique_guide_queries=guide.calls)
  results.append(dict(seed=seed,pool_orbits=len(pool),eligible_orbits=len(eligible),explore_amplitude_queries=qs0,arms=arms))
  print('TRIAL',seed,'pool',len(pool),'selection',[(m,v['n_selected_orbits'],round(v['guide_predicted_gain'],7),round(v['ED_weight_improvement'],8)) for m,v in arms.items()],flush=True)
 return results

def score_true_amplitude(net,n=5000):
 """Legacy FN amplitude audit only; not strictly same FN operator as analytic g0."""
 rng=np.random.default_rng(331);inds=rng.choice(D,size=n,p=(M.amplitude**2)/(M.amplitude**2).sum())
 x=M.states[inds];y=x^MASKS[rng.integers(NB,size=n)];valid=M.lookup[y]>=0
 xx=inds[valid];yy=M.lookup[y[valid]]
 ids=np.unique(np.r_[xx,yy]);netpred={}
 with torch.no_grad():
  for start in range(0,len(ids),512):
   block=ids[start:start+512];logits=net.symmetric(T.X[block]).numpy()
   for i,v in zip(block,logits):netpred[int(i)]=float(LOGG0[i]+v)
 target=np.log(np.maximum(M.amplitude,1e-15))
 err=np.array([(netpred[int(i)]-netpred[int(j)])-(target[int(i)]-target[int(j)]) for i,j in zip(xx,yy)])
 return dict(n_hops=len(err),RMS_logratio_to_legacy_FN=float(np.sqrt(np.mean(err*err))),warning='Legacy FN amplitudes arise from a different guide; use only as rough scale check.')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--walkers',type=int,default=3500);ap.add_argument('--burn',type=int,default=450);ap.add_argument('--steps',type=int,default=450);ap.add_argument('--train-steps',type=int,default=900);ap.add_argument('--pilot',action='store_true');ap.add_argument('--no-trials',action='store_true');ap.add_argument('--tag',type=str,default='');a=ap.parse_args()
 start=time.monotonic()
 if a.pilot:a.walkers=1300;a.burn=100;a.steps=100;a.train_steps=200
 print('START',vars(a),'Jastrow alpha',ALPHA,flush=True)
 s=FNProjector();pos,ps=s.run(seed=53,nwalk=a.walkers,burn=a.burn,steps=a.steps,thin=4)
 print('POS',json.dumps(ps),'elapsed',round(time.monotonic()-start,1),flush=True)
 neg,ns=sample_reference(seed=61,nchains=a.walkers,burn=max(a.burn*2,500),steps=a.steps,thin=4)
 print('NEG',json.dumps(ns),flush=True)
 net,ts=train_ratio(pos,neg,steps=a.train_steps)
 trials=[] if a.no_trials else eval_trials(net)
 results=dict(protocol='FN power projector mixed walkers vs Jastrow squared baseline, no exact FN amplitude labels',projector=ps,reference=ns,training=ts,sign_trials=trials,limitations=['Sampler uses explicit 20-site state-to-index lookup for computational speed; a scaled version needs callable sign and direct bit operations.','FN sampler uses exact frozen signs from previous 20-site study; not a trained sign network.','Amplitudes for fixed-node sampling are analytic Jastrow guide, not true exact FN solutions.','The sampled FN operator differs from the previous FN-amplitude-oracle experiment.','Model energy gains computed in own amplitude; not sufficient to establish true FN or self-consistent energy gain.','GFMC population control/resampling introduces finite-walker bias.','A second trial on a different sample/cluster is needed to establish generalization.'],elapsed_s=time.monotonic()-start)
 out=P/f'j1j2_walker_only_amplitude20{a.tag}_results.json';out.write_text(json.dumps(results,indent=2));torch.save(dict(model=net.state_dict(),alpha=ALPHA),P/f'j1j2_walker_only_amplitude20{a.tag}_model.pt')
 print('SUMMARY',json.dumps(dict(pos_unique=ps['unique_samples'],fn_rows=ps['unique_projector_rows'],train_pos=ts['training_positive'],energy=[{m:sum(v['arms'][m]['guide_predicted_gain'] for v in trials)/len(trials) for m in ('fitted','jastrow')} if trials else {}],elapsed_s=results['elapsed_s'])),flush=True)