#!/usr/bin/env python3
"""Does budgeted local sign correction survive imperfections in callable guide amplitudes?
Each orbit has a single Gaussian log-amplitude perturbation, respecting model symmetries.
Candidate pool fixed by exact-amplitude MCMC, so this isolates the effect on ranking and gating.
True energy is measured post-selection using clean guide amplitudes; ED not used for selection.
"""
import os,json
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from collections import defaultdict
import j1j2_budgeted_orbit20 as B
import j1j2_local_query_diagnostic as M
P='/mnt/data/'
class Noisy(M.Sampler):
 def __init__(self,base,seed,sigma):
  super().__init__();self.sigma=sigma;self.seed=seed;self.keys=base.keys.copy();self.members=base.members.copy()
  self.amps={};self.zcache={}
  for x in base.amps:self.amp(x)
 def amp(self,x):
  x=int(x);self.n_amp_requests+=1
  if x not in self.amps:
   k=self.orbit(x)
   if k not in self.zcache:self.zcache[k]=float(np.random.default_rng(self.seed*10000000+k).normal())
   self.amps[x]=float(M.amplitude[self.idx(x)]*np.exp(self.sigma*self.zcache[k]))
  return self.amps[x]

def true_deltaE(selected,structure):
 sel=set(selected);e=0.
 for g in sel:
  n=len(structure.members[g]);ax=M.amplitude[structure.idx(g)];sx=int(M.signs[structure.idx(g)])
  for y,h in structure.neighbors(g):
   if structure.orbit(y) in sel:continue
   e+=-4*n*h*ax*M.amplitude[structure.idx(y)]*sx*int(M.signs[structure.idx(y)])
 return float(e)

def test(seed,sigma,Budget=300,limit=40):
 base,pool,rep,proxy,sm=B.generate(seed)
 # For controlled noise, repeat an IDENTICAL baseline sampler candidate list with noisy amplitudes.
 s=Noisy(base,seed=seed+300,sigma=sigma)
 # reconstruct discovered seed states via an independent reproducible MH rerun
 rng=np.random.default_rng(seed);x=int(M.states[int(rng.integers(M.D))]);distinct=set()
 burn=2500;n_seed=120;thin=15
 for step in range(burn+n_seed*thin):
  maskv,_,_,_=M.bonds[int(rng.integers(len(M.bonds)))];y=x^int(maskv)
  if M.lookup[int(y)]>=0:
   a=float(M.amplitude[M.lookup[int(x)]]);b=float(M.amplitude[M.lookup[int(y)]])
   if rng.random()<b/max(a,1e-300):x=y
  if step>=burn and (step-burn)%thin==0:distinct.add(x)
 scores={g:0. for g in pool};count={g:0 for g in pool}
 for x in distinct:
  ax=s.amp(x);sx=int(M.signs[s.idx(x)])
  for y,h in s.neighbors(x):
   gy=s.orbit(y)
   scores[gy]+=h*sx*int(M.signs[s.idx(y)])*ax*s.amp(y)
   count[gy]+=1
 elig=sorted(pool,key=lambda g:(-scores[g]/max(1,count[g]),g))[:Budget]
 raw={};edges={}
 for g in elig:raw[g],edges[g]=B.exact_score_one(s,g)
 chosen=[];sel=set()
 for _ in range(limit):
  gains={g:raw[g]-8*sum(v for h,v in edges[g].items() if h in sel) for g in elig if g not in sel}
  if not gains:break
  g=max(gains,key=gains.get)
  if gains[g]<=0:break
  chosen.append(g);sel.add(g)
 predicted=-sum(raw[g] for g in sel)+4*sum(v for g in sel for h,v in edges[g].items() if h in sel)
 truth=true_deltaE(sel,s)
 return dict(seed=seed,sigma=sigma,n_selected_orbits=len(sel),predicted_deltaE=float(predicted),true_deltaE=float(truth),n_amplitudes_queried=len(s.amps),pool=len(pool))

if __name__=='__main__':
 out=[]
 for seed in range(201,213):
  rows=[]
  for sigma in (0.,0.1,0.3,0.5):
   r=test(seed,sigma);out.append(r);rows.append((sigma,round(-r['true_deltaE'],7),r['n_selected_orbits']))
  print('SEED',seed,'noise(gain,n)',rows,flush=True)
 summary={}
 for sigma in (0.,0.1,0.3,0.5):
  rr=[r for r in out if r['sigma']==sigma];gain=np.array([-r['true_deltaE'] for r in rr]);pred=np.array([-r['predicted_deltaE'] for r in rr])
  summary[str(sigma)]={'gain_mean':float(gain.mean()),'gain_median':float(np.median(gain)),'gain_min':float(gain.min()),'gain_max':float(gain.max()),'wrong_direction_runs':int((gain<0).sum()),'predicted_gain_mean':float(pred.mean()),'mean_queries':float(np.mean([r['n_amplitudes_queried'] for r in rr])),'mean_selected_orbits':float(np.mean([r['n_selected_orbits'] for r in rr]))}
 res={'summary':summary,'per_seed':out,'limits':['noise oracle perturbs true FN amplitudes; is not a trained NQS amplitude model','candidate pool was sampled from clean FN amplitudes; only screening and greedy gate are noisy','noise is per symmetry orbit, log-normal with controlled sigma','no ED signs used for ranking']}
 open(P+'j1j2_noisy_oracle_diagnostic_results.json','w').write(json.dumps(res,indent=2))
 print('SUMMARY',json.dumps(summary,indent=2),flush=True)