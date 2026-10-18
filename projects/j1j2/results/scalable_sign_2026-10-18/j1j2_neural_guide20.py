#!/usr/bin/env python3
"""Pilot real callable CNN amplitude vs FN oracle for budgeted local orbit corrections.
Amplitudes for training labelled by exact 20-site FN solver, sampling initialization is
from teacher a^1 (not scalable). Candidate selection and gates use CNN predictions only.
Results are diagnostics, not a fully scalable self-consistent algorithm.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time
import numpy as np
import torch
from torch import nn
from collections import defaultdict
import j1j2_local_query_diagnostic as M
import j1j2_budgeted_orbit20 as B
P=Path('/mnt/data');torch.set_num_threads(2)
D=M.D;N=M.N
np.random.seed(504);torch.manual_seed(504)
X=torch.tensor(((((M.states[:,None]>>np.arange(N,dtype=np.uint32))&1).astype(np.float32))*2-1).copy())
# group-equivariant CNN via translation pooling; global spin inversion at output time
nbr=np.array([[M.site(x+dx,yy+dy) for dy in [-1,0,1] for dx in [-1,0,1]] for yy in range(M.Ly) for x in range(M.Lx)])
idx=torch.tensor(nbr,dtype=torch.long)
class Net(nn.Module):
 def __init__(self):
  super().__init__();c=24;self.c0=nn.Linear(9,c);self.c1=nn.Linear(9*c,c);self.c2=nn.Linear(9*c,c);self.l1=nn.LayerNorm(c);self.l2=nn.LayerNorm(c);self.h=nn.Linear(c,1)
 def forward(self,x):
  h=torch.nn.functional.gelu(self.c0(x[:,idx]));h=h+torch.nn.functional.gelu(self.c1(self.l1(h)[:,idx].reshape(-1,N,216)));h=h+torch.nn.functional.gelu(self.c2(self.l2(h)[:,idx].reshape(-1,N,216)));return self.h(h).mean(1).squeeze(-1)
 def symmetric(self,x):return (self.forward(x)+self.forward(-x))*0.5

def train(nseed=1700,nsteps=1200):
 # Teaching points by tempered sampling + one-hop neighbor support; *no ground truth signs*.
 rng=np.random.default_rng(722);pi=M.amplitude.copy();pi/=pi.sum()
 seeds=np.unique(rng.choice(D,size=nseed,replace=True,p=pi))
 reached=set(int(M.states[i]) for i in seeds)
 for i in seeds:
  for y,_ in M.Sampler().neighbors(int(M.states[i])): reached.add(y)
 ids=np.array([int(M.lookup[x]) for x in reached]);rng.shuffle(ids)
 # heldout states selected by orbit, so sym-related samples don't leak
 sam=M.Sampler();uniq=list(set(sam.orbit(int(M.states[i])) for i in ids));rng.shuffle(uniq);keep=set(uniq[:int(.9*len(uniq))]);training=np.array([i for i in ids if sam.orbit(int(M.states[i])) in keep]);valid=np.array([i for i in ids if sam.orbit(int(M.states[i])) not in keep]);
 target=np.log(np.maximum(M.amplitude,1e-14));mu=float(target[training].mean());scale=float(target[training].std());
 y=torch.tensor(((target-mu)/scale).astype('float32'))
 model=Net();opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-4)
 start=time.monotonic();h=[]
 for step in range(nsteps):
  sub=training[rng.integers(len(training),size=192)];z=model.symmetric(X[sub]);loss=nn.functional.smooth_l1_loss(z,y[sub]);opt.zero_grad();loss.backward();opt.step()
  if (step+1)%300==0:
   with torch.no_grad():
    q=model.symmetric(X[valid[:1024]]);vl=float(nn.functional.smooth_l1_loss(q,y[valid[:1024]]))
   h.append([step+1,round(float(loss.item()),6),round(vl,6)]);print('TRAIN',h[-1],'elapsed',round(time.monotonic()-start,1),flush=True)
 model.eval()
 return model,mu,scale,dict(training_ids=len(training),label_queries=len(reached),training_orbits=len(keep),steps=nsteps,history=h,train_elapsed_s=round(time.monotonic()-start,1))

class Neural(M.Sampler):
 def __init__(self,model,mu,scale):
  super().__init__();self.model=model;self.mu=mu;self.scale=scale
 def amp(self,x):
  x=int(x);self.n_amp_requests+=1
  if x not in self.amps:
   with torch.no_grad():
    pred=float(self.model.symmetric(X[self.idx(x):self.idx(x)+1]).item());self.amps[x]=float(np.exp(np.clip(self.mu+self.scale*pred,-40,4)))
  return self.amps[x]

def true_deltaE(chosen,s):
 selected=set(chosen);e=0.
 for g in selected:
  n=len(s.members[g]);ax=M.amplitude[s.idx(g)];sx=int(M.signs[s.idx(g)])
  for y,h in s.neighbors(g):
   if s.orbit(y) in selected:continue
   e-=4*n*h*ax*M.amplitude[s.idx(y)]*sx*int(M.signs[s.idx(y)])
 return float(e)

def trial(seed,net,mu,scale,Budget=300,limit=40):
 rng=np.random.default_rng(seed);s=Neural(net,mu,scale)
 x=int(M.states[int(rng.integers(D))]);seeds=[];accepted=0;burn=2000;nseed=100;thin=15
 for it in range(burn+nseed*thin):
  x,ac=s.propose(x,rng);accepted+=ac
  if it>=burn and (it-burn)%thin==0: seeds.append(x)
 distinct=set(seeds);candidate=set(seeds)
 for x in distinct:
  for y,h in s.neighbors(x):candidate.add(y)
 pool=sorted({s.orbit(x) for x in candidate});n_explore=len(s.amps)
 # observed signed bond proxy only uses NQS amplitudes at visited endpoints
 proxy={g:0. for g in pool};degree={g:0 for g in pool}
 for x in distinct:
  ax=s.amp(x);sx=int(M.signs[s.idx(x)])
  for y,h in s.neighbors(x):
   gy=s.orbit(y);proxy[gy]+=h*sx*int(M.signs[s.idx(y)])*ax*s.amp(y);degree[gy]+=1
 elig=sorted(pool,key=lambda g:(-proxy[g]/max(1,degree[g]),g))[:Budget]
 raw={};edges={}
 for g in elig:raw[g],edges[g]=B.exact_score_one(s,g)
 chosen=[];picked=set()
 for j in range(limit):
  gains={g:raw[g]-8*sum(v for h,v in edges[g].items() if h in picked) for g in elig if g not in picked}
  if not gains:break
  g=max(gains,key=gains.get)
  if gains[g]<=0:break
  picked.add(g);chosen.append(g)
 pred=-sum(raw[g] for g in chosen)+4*sum(v for g in picked for h,v in edges[g].items() if h in picked)
 true=true_deltaE(chosen,s)
 return dict(seed=seed,pool=len(pool),candidate_orbits=len(elig),selected=len(chosen),walker_accept=accepted/(burn+nseed*thin),queries_explore=n_explore,queries_total=len(s.amps),predicted_gain=-float(pred),true_gain=-float(true))

def ratio_audit(net,mu,scale,n=5000):
 # *Evaluation-only* samples of physical bonds; not consumed in candidate selection.
 rng=np.random.default_rng(1234);pi=M.amplitude**2;pi/=pi.sum();inds=rng.choice(D,n,p=pi)
 pairs=[]
 for ix in inds:
  x=int(M.states[ix]);neigh=list(M.Sampler().neighbors(x));y,h=neigh[int(rng.integers(len(neigh)))];pairs.append((ix,int(M.lookup[y])))
 keys=np.unique(np.array(pairs).ravel());pred=np.empty(len(keys))
 with torch.no_grad():
  for start in range(0,len(keys),512):
   k=keys[start:start+512];pred[start:start+len(k)]=(mu+scale*net.symmetric(X[k]).numpy())
 lookup={int(k):float(q) for k,q in zip(keys,pred)}
 true=np.log(np.maximum(M.amplitude,1e-14));errs=np.array([(lookup[x]-lookup[y])-(true[x]-true[y]) for x,y in pairs]);
 return dict(n_pairs=len(pairs),rmse_logratio=float(np.sqrt(np.mean(errs**2))),mae_logratio=float(np.mean(np.abs(errs))),p90_abs_logratio=float(np.quantile(np.abs(errs),.9)),predictive_query_count=len(keys))

if __name__=='__main__':
 start=time.time();net,mu,scale,learn=train(int(os.environ.get('TRAIN_SAMPLES','1700')),int(os.environ.get('STEPS','1200')))
 print('TRAIN_DONE',json.dumps(learn),flush=True)
 audit=ratio_audit(net,mu,scale);print('RATIO_AUDIT',json.dumps(audit),flush=True)
 trials=[]
 for seed in range(301,305):
  v=trial(seed,net,mu,scale);trials.append(v);print('TRIAL',json.dumps(v),'elapsed',round(time.time()-start,1),flush=True)
 out=dict(training=learn,ratio_audit=audit,trials=trials,summary=dict(mean_actual_gain=float(np.mean([v['true_gain'] for v in trials])),mean_predicted_gain=float(np.mean([v['predicted_gain'] for v in trials])),negative_runs=int(np.sum([v['true_gain']<0 for v in trials])),mean_queries=float(np.mean([v['queries_total'] for v in trials]))),limitations=['FN amplitude labels used for training, acquired via exact 20-site teacher; not a scalable FN sampling method.','Training configurations sampled from a^1 exact amplitude; this is an oracle training distribution.','Walker uses CNN estimated amplitude; local bond discovery and correction use CNN estimates only.','Signed guide signs still available through frozen array, not trained signs.','FN amplitude target very small tail is hard, log-amplitude ratio errors measured on physical a^2 bonds only.','Energy is fixed-clean-amplitude after flips, not a self-consistent FN energy or neural-wavefunction energy.'])
 p=P/'j1j2_neural_guide20_results.json';p.write_text(json.dumps(out,indent=2));print('FINAL',json.dumps(out['summary']))
 torch.save({'model':net.state_dict(),'mu':mu,'scale':scale},str(P/'j1j2_neural_guide20_model.pt'))