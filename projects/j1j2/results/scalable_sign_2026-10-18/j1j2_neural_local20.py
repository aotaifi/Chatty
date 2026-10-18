#!/usr/bin/env python3
"""20-site NN-guide scalability diagnostic: train on FN-labelled sampled orbits,
then collect new candidate bonds using only the callable NN. ED/exact FN vector
is used only as the labelled training teacher and as a POST-DECISION oracle.
No claim that direct FN amplitude labels are available in a realistic GFMC run.
"""
import os,time,json,math,argparse
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import j1j2_local_query_diagnostic as M
import j1j2_budgeted_orbit20 as B
P=Path('/mnt/data');torch.set_num_threads(2)
N=M.N; D=M.D
nbr=np.array([[M.site(x+dx,y+dy) for dy in (-1,0,1) for dx in (-1,0,1)] for y in range(M.Ly) for x in range(M.Lx)],dtype=np.int64)
NT=torch.as_tensor(nbr,dtype=torch.int64)

class CNN(nn.Module):
 def __init__(self):
  super().__init__();self.c0=nn.Linear(9,16);self.c1=nn.Linear(144,16);self.c2=nn.Linear(144,16);self.n1=nn.LayerNorm(16);self.n2=nn.LayerNorm(16);self.out=nn.Linear(16,1)
 def once(self,x):
  h=torch.nn.functional.gelu(self.c0(x[:,NT]));h=h+torch.nn.functional.gelu(self.c1(self.n1(h)[:,NT].reshape(-1,N,144)));h=h+torch.nn.functional.gelu(self.c2(self.n2(h)[:,NT].reshape(-1,N,144)))
  return self.out(h).sum(1).squeeze(-1)/math.sqrt(N)
 def forward(self,x):
  # exact global spin-flip invariance, translation-invariant network architecture
  return (self.once(x)+self.once(-x))/2

def spin_x(states):
 v=np.asarray(states,dtype=np.uint32)
 return torch.as_tensor((2*((v[:,None]>>np.arange(N))&1).astype(np.int8)-1).astype(np.float32))

def observed_train_orbits(seed,n_orbits=600):
 # Samples drawn from true-FN amplitude tempered MCMC: optimistic training oracle.
 # Need to replace in future with FN mixed-walker data or unbiased ratio estimators.
 rng=np.random.default_rng(seed);sam=M.Sampler();discovered=[];visited=set()
 start=int(M.states[int(rng.integers(D))]);x=start
 for chain in range(30):
  if chain>0:x=int(M.states[int(rng.integers(D))])
  for step in range(1500):
   x,_=sam.propose(x,rng)
   if step>=150 and step%9==0:
    k=sam.orbit(x)
    if k not in visited:visited.add(k);discovered.append(k)
  if len(discovered)>=n_orbits:break
 # Supplement only by one-hop discovered neighbors, never global selection by FN oracle.
 # Teacher amplitudes for chosen training orbits are explicitly counted.
 if len(discovered)<n_orbits:
  visited_pool=set(discovered)
  for g in list(discovered):
   for y,h in sam.neighbors(g):
    ky=sam.orbit(y)
    if ky not in visited_pool:visited_pool.add(ky);discovered.append(ky)
    if len(discovered)>=n_orbits:break
   if len(discovered)>=n_orbits:break
 train=discovered[:n_orbits]
 # Repeated amplitude requests are cached by the sampler; count distinct oracle queries.
 y=np.array([math.log(sam.amp(g)) for g in train],dtype=np.float32)
 X=spin_x(train)
 pos={int(g):i for i,g in enumerate(train)}
 ei=[];ej=[];ew=[]
 for i,g in enumerate(train):
  for yb,h in sam.neighbors(g):
   j=pos.get(sam.orbit(yb),-1)
   if j>=0 and j>i:ei.append(i);ej.append(j);ew.append(h)
 return train,X,y,np.asarray(ei,np.int32),np.asarray(ej,np.int32),dict(teacher_distinct_amplitude_queries=len(sam.amps),initial_discovered_orbits=len(visited),training_orbits=len(train),training_edges=len(ei),proposals=1500*(chain+1))

def train(seed,objective='point',steps=900,ntrain=600):
 keys,X,y,ei,ej,meta=observed_train_orbits(100+seed,ntrain)
 torch.manual_seed(10000+seed);net=CNN();net.train()
 mean=y.mean();sd=max(float(y.std()),0.05);Y=torch.tensor((y-mean)/sd)
 # If training pairs are scarce, fall back to pointwise loss rather than make a spurious claim.
 E_i=torch.tensor(ei,dtype=torch.long);E_j=torch.tensor(ej,dtype=torch.long)
 opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=1e-4)
 rng=np.random.default_rng(800+seed)
 t=time.time();last={}
 for step in range(steps):
  ib=torch.from_numpy(rng.integers(len(keys),size=128))
  pred=net(X[ib]);err=pred-Y[ib]
  loss=torch.mean(torch.nn.functional.smooth_l1_loss(pred,Y[ib],reduction='none',beta=0.2))
  pair_loss=torch.zeros(())
  if objective=='edge' and len(ei)>0:
   ij=torch.from_numpy(rng.integers(len(ei),size=128));i=E_i[ij];j=E_j[ij]
   v=net(torch.cat([X[i],X[j]],dim=0));f,g=v[:128],v[128:]
   pair_loss=torch.mean(torch.nn.functional.smooth_l1_loss(f-g,Y[i]-Y[j],reduction='none',beta=0.1))
   loss=loss+1.5*pair_loss
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
  if step%300==0 or step==steps-1:last=dict(step=step,loss=float(loss.detach()),pair=float(pair_loss.detach()))
 net.eval()
 return net,mean,sd,dict(**meta,training_seconds=round(time.time()-t,2),final_loss=last,model_parameters=sum(x.numel() for x in net.parameters()),train_seed=seed,objective=objective,train_keys=keys)

class LearnedGuide(M.Sampler):
 def __init__(self,net,mean,sd):
  super().__init__();self.net=net;self.mean=mean;self.sd=sd;self.orb_amps={};self.batches=0
 @torch.inference_mode()
 def amp(self,x):
  x=int(x);self.n_amp_requests+=1
  if x in self.amps:return self.amps[x]
  g=self.orbit(x)
  if g not in self.orb_amps:
   y=float(self.net(spin_x([g]))[0]);self.orb_amps[g]=float(math.exp(float(self.mean+self.sd*y)));self.batches+=1
  self.amps[x]=self.orb_amps[g]
  return self.amps[x]

# Construct identical candidate pool for all scoring methods, discovered using only the learned sampler.
def discover(net,mean,sd,seed,n_seed=120,burn=1200,thin=12):
 rng=np.random.default_rng(seed);s=LearnedGuide(net,mean,sd)
 x=int(M.states[int(rng.integers(D))]);samples=[];accepted=0
 for t in range(burn+n_seed*thin):
  x,ok=s.propose(x,rng);accepted+=int(ok)
  if t>=burn and (t-burn)%thin==0:samples.append(x)
 seeds=set(samples);allstates=set(seeds)
 for x in seeds:
  for y,h in s.neighbors(x):allstates.add(y)
 pool=sorted({s.orbit(x) for x in allstates})
 proxy={g:0. for g in pool};cnt={g:0 for g in pool}
 for x in seeds:
  ax=s.amp(x);sx=int(M.signs[s.idx(x)])
  for y,h in s.neighbors(x):
   g=s.orbit(y);proxy[g]+=h*sx*int(M.signs[s.idx(y)])*ax*s.amp(y);cnt[g]+=1
 return s,pool,{g:proxy[g]/cnt[g] if cnt[g] else 0 for g in pool},dict(acceptance=accepted/(burn+n_seed*thin),distinct_seed_states=len(seeds),candidate_orbits=len(pool),model_orbit_calls_preselection=s.batches,model_state_requests_preselection=s.n_amp_requests)

def evaluate_selection(s,prio,maxflips=40,confidence=0):
 # Only selected shortlist orbit representatives and their local neighbors queried.
 bonds={};raw={}
 for g in prio:
  G,edge=B.exact_score_one(s,g);raw[g]=G;bonds[g]=edge
 selected=set();sequence=[];pred_gains=[]
 for _ in range(maxflips):
  eligible=[]
  for g in prio:
   if g in selected:continue
   edges=bonds[g];P=sum(max((-v if h in selected else v),0.) for h,v in edges.items());Q=sum(max((-v if h in selected else v)*-1,0.) for h,v in edges.items())
   marg=4*(math.exp(-confidence)*P-math.exp(confidence)*Q)
   eligible.append((marg,g,P,Q))
  if not eligible:break
  marg,g,P,Q=max(eligible)
  if marg<=0:break
  selected.add(g);sequence.append(int(g));pred_gains.append(float(4*(P-Q)))
 # predicted and oracle true energy via graph boundary, recompute oracle AFTER decisions.
 est=sum(raw[g] for g in selected)-4*sum(v for g in selected for h,v in bonds[g].items() if h in selected)
 # Oracle evaluation uses *true* exact FN amplitudes, NOT used in scoring or gating
 tru=M.Sampler();tru.keys=s.keys.copy();tru.members=s.members.copy()
 actual=-sum(B.exact_score_one(tru,g)[0] for g in selected)
 # Internal selected-to-selected graph edges counted twice; correct via +4 sum directed w.
 if selected:
  true_edges={g:B.exact_score_one(tru,g)[1] for g in selected}
  actual+=4*sum(v for g in selected for h,v in true_edges[g].items() if h in selected)
 return dict(n_selected=len(selected),predicted_gain=float(est),actual_gain=float(-actual),chosen_orbits=sequence,predicted_marginal_gains=pred_gains,model_orbit_calls=getattr(s,'batches',0),model_state_requests=s.n_amp_requests,oracle_posthoc_query_count=len(tru.amps))

def eval_edge_ratios(model,mu,sd,seed,nsample=800,excluded_orbits=()):
 # Post-hoc evaluation pool: a separate learned-guide walker; target FN labels are only queried *after* model predictions.
 s,pool,proxy,info=discover(model,mu,sd,seed,n_seed=90,burn=1500,thin=12)
 subset=pool[:min(nsample,len(pool))]
 excluded=set(excluded_orbits)
 predicted=[];actual=[];signed_w=[];seen=set()
 for g in subset:
  for y,h in s.neighbors(g):
   gy=s.orbit(y)
   if gy==g or g in excluded or gy in excluded:continue
   pair=(min(g,gy),max(g,gy))
   if pair in seen:continue
   seen.add(pair)
   pr=math.log(s.amp(y)/s.amp(g))
   tr=math.log(float(M.amplitude[s.idx(y)])/float(M.amplitude[s.idx(g)]))
   predicted.append(pr);actual.append(tr)
   signed_w.append(abs(h*float(M.amplitude[s.idx(g)])*float(M.amplitude[s.idx(y)])))
 predicted=np.array(predicted);actual=np.array(actual);signed_w=np.array(signed_w)
 diff=predicted-actual
 return dict(n_test_edges=len(diff),edge_logratio_rmse=float(np.sqrt(np.mean(diff**2))),edge_logratio_mae=float(np.mean(np.abs(diff))),edge_logratio_weighted_rmse=float(np.sqrt(np.average(diff**2,weights=signed_w))),edge_abs_error_p90=float(np.quantile(abs(diff),.9)),edge_abs_error_p99=float(np.quantile(abs(diff),.99)),model_calls_for_test=s.batches)

def run(ns=2,nsteps=900,ntrain=600,ntrials=3,budget=300):
 records=[]
 for seed in range(ns):
  for objective in ('point','edge'):
   t=time.time();print('TRAIN_START',seed,objective,flush=True)
   model,mu,sd,metadata=train(seed,objective,steps=nsteps,ntrain=ntrain)
   torch.save(dict(state_dict=model.state_dict(),mean=float(mu),sd=float(sd),training_meta=metadata),P/f'j1j2_neural_guide20_{seed}_{objective}.pt')
   scores=[]
   for trial in range(ntrials):
    sid=3000+seed*100+trial
    s,pool,proxy,info=discover(model,mu,sd,sid)
    eligible=sorted(pool,key=lambda g:(-proxy[g],g))[:min(budget,len(pool))]
    case=evaluate_selection(s,eligible)
    # Sensitivity to predeclared conservative local-ratio margin; no oracle in acceptance.
    gates={}
    for eps in (0.05,0.1,0.2,0.4):
     check=evaluate_selection(s,eligible,confidence=eps)
     gates[str(eps)]={k:check[k] for k in ('n_selected','actual_gain','predicted_gain')}
    case['gates']=gates
    # Inaccessible upper bound, evaluated strictly AFTER selection for comparison.
    oracle=M.Sampler();oracle.keys=s.keys.copy();oracle.members=s.members.copy()
    optimum=evaluate_selection(oracle,eligible)
    case['oracle_same_pool_gain']=optimum['actual_gain']
    case.update(dict(seed=sid,**info))
    scores.append(case)
    print('TEST',seed,objective,trial,'gain',round(case['actual_gain'],8),'pred',round(case['predicted_gain'],8),'flips',case['n_selected'],'calls',case['model_orbit_calls'],flush=True)
   edge=eval_edge_ratios(model,mu,sd,4000+seed*100,excluded_orbits=metadata['train_keys'])
   rec=dict(seed=seed,objective=objective,train={k:v for k,v in metadata.items() if k!='train_keys'},ratio_test=edge,per_trial=scores,mean_actual_gain=float(np.mean([c['actual_gain'] for c in scores])),mean_predicted_gain=float(np.mean([c['predicted_gain'] for c in scores])),n_trials_worse=int(sum(c['actual_gain']<0 for c in scores)),wall_seconds=round(time.time()-t,2))
   records.append(rec)
   (P/'j1j2_neural_local20_results.json').write_text(json.dumps(dict(in_progress=True,records=records,limitations=['FN amplitude labels for training were supplied by an exact on-demand full-space oracle, NOT by a realistic walker mixed estimator.','The current sign guide is supplied by an exact finite-size table rather than a callable sign network.','The 20-site torus is small and symmetry-orbit closure compresses its size.','Exact FN amplitudes are accessed only for training labels and post-decision test scoring, not for online proposal or flip selection.','Only fixed-amplitude variational energy evaluated; FN next-step energy not established.','True FN-based Metropolis is used for the training data acquisition; production would require GFMC/importance-weighted sampling.']),indent=2))
   print('FINISHED',seed,objective,'mean gain',rec['mean_actual_gain'],'ratio rmse',edge['edge_logratio_weighted_rmse'],'wall',rec['wall_seconds'],flush=True)
 out=json.loads((P/'j1j2_neural_local20_results.json').read_text());out['in_progress']=False;(P/'j1j2_neural_local20_results.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=2);ap.add_argument('--steps',type=int,default=900);ap.add_argument('--train-orbits',type=int,default=600);ap.add_argument('--trials',type=int,default=3);ap.add_argument('--budget',type=int,default=300);a=ap.parse_args();run(a.seeds,a.steps,a.train_orbits,a.trials,a.budget)
