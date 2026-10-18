#!/usr/bin/env python3
"""Budgeted sign-correction diagnostic on exact J1-J2 20-site data.
Uses only on-demand guide-amplitude queries at sampler-discovered representatives and their hops.
No global basis scan for selection, no ED labels in selection. Full FN amplitude is oracle BACKEND.
"""
import os,json,time
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from collections import defaultdict
from pathlib import Path
import j1j2_local_query_diagnostic as M
P=Path('/mnt/data'); D=M.D

def generate(seed,n_seed=120,burn=2500,thin=15):
 rng=np.random.default_rng(seed);s=M.Sampler();x=int(M.states[int(rng.integers(D))]);seeds=[];accepted=0
 for step in range(burn+n_seed*thin):
  x,ac=s.propose(x,rng);accepted+=ac
  if step>=burn and (step-burn)%thin==0:seeds.append(x)
 distinct=set(seeds);seen=set(seeds)
 for x in distinct:
  for y,h in s.neighbors(x): seen.add(y)
 pool=sorted({s.orbit(x) for x in seen})
 # query each candidate orbit's representative amplitude, which an evaluable guide could supply
 rep_amp={g:s.amp(g) for g in pool}
 # screen candidate orbits from ONLY edges reached by the walker seed states.
 # Edge-score amplitude queries are counted in the common base cost.
 max_frustrated={g:0. for g in pool}; seen_signed={g:0. for g in pool}; edge_n={g:0 for g in pool}
 for x in distinct:
  sx=int(M.signs[s.idx(x)]);ax=s.amp(x)
  for y,h in s.neighbors(x):
   gy=s.orbit(y); sy=int(M.signs[s.idx(y)]); ay=s.amp(y)
   w=h*sx*sy*ax*ay
   if w>max_frustrated[gy]:max_frustrated[gy]=w
   seen_signed[gy]+=w;edge_n[gy]+=1
 edgeproxies={'max_frustrated':max_frustrated,'observed_signed':{g:(seen_signed[g]/edge_n[g] if edge_n[g] else 0.) for g in pool}}
 return s,pool,rep_amp,edgeproxies,dict(seed=seed,n_seed=n_seed,accepted_fraction=accepted/(burn+n_seed*thin),unique_seed_states=len(distinct),seen_states=len(seen),candidate_orbits=len(pool),preselection_amplitude_queries=len(s.amps))

def exact_score_one(sam,g):
 ax=sam.amp(g);sx=int(M.signs[sam.idx(g)]);norb=len(sam.members[g]);w=defaultdict(float)
 for y,h in sam.neighbors(g):
  gy=sam.orbit(y)
  if gy!=g:
   w[gy]+=norb*h*ax*sam.amp(y)*sx*int(M.signs[sam.idx(y)])
 return 4*sum(w.values()),dict(w)

def run_arm(base,pool,rep_amp,seed,eligible,limit=40):
 s=M.Sampler();s.amps=base.amps.copy();s.keys=base.keys.copy();s.members=base.members.copy()
 base_q=len(s.amps)
 edges={};raw={}
 for g in eligible:
  raw[g],edges[g]=exact_score_one(s,g)
 newq=len(s.amps)-base_q
 # Greedy selection; each marginal energy gain recomputed after previous sign flips.
 selected=[];selected_set=set();tr=[]
 for _ in range(min(limit,len(eligible))):
  gains={g:raw[g]-8*sum(v for h,v in edges[g].items() if h in selected_set) for g in eligible if g not in selected_set}
  g=max(gains,key=gains.get)
  if gains[g]<=0:break
  selected.append(g);selected_set.add(g);tr.append(gains[g])
 e=-sum(raw[g] for g in selected_set)+4*sum(v for g in selected_set for h,v in edges[g].items() if h in selected_set)
 assert abs(-sum(tr)-e)<2e-9,(-sum(tr),e)
 indices=np.concatenate([np.array([s.idx(x) for x in s.members[g]]) for g in selected]) if selected else np.array([],int)
 ww=float(np.sum(M.psi0[indices]**2*(M.signs[indices]*M.psi0[indices]>0))) if len(indices) else 0.
 wf=float(np.sum(M.psi0[indices]**2*(M.signs[indices]*M.psi0[indices]<0))) if len(indices) else 0.
 return dict(selected_keys=[int(g) for g in selected],deltaE=float(e),n_orbits=len(selected),n_states=len(indices),extra_amp_queries=int(newq),total_amp_queries=len(s.amps),wrong_ed_weight_change=ww-wf,n_krylov_flip=int(M.target[indices].sum()))

def run(seed,limits=(100,300),max_select=40):
 s,pool,rep_amp,edgeproxies,summary=generate(seed)
 rng=np.random.default_rng(seed+10000);result={}
 for B in limits:
  if B>len(pool):continue
  # fixed candidate pool and equal count of exactly evaluated orbit scores.
  # cheap sign-only proxy: field assuming all adjacent amplitude ratios are 1.
  # This queries only guide signs, not FN amplitudes.
  signed_score={}
  for g in pool:
   sx=int(M.signs[s.idx(g)]); sc=0.0
   for y,h in s.neighbors(g):
    if s.orbit(y)!=g:sc+=h*sx*int(M.signs[s.idx(y)])
   signed_score[g]=sc
  selectors={
    'max_frustrated':sorted(pool,key=lambda g:(-edgeproxies['max_frustrated'][g],g))[:B],
    'observed_signed':sorted(pool,key=lambda g:(-edgeproxies['observed_signed'][g],g))[:B],
    'sign_only_field':sorted(pool,key=lambda g:(-signed_score[g],g))[:B],
    'random':list(rng.choice(pool,B,replace=False)),
    'large_amplitude': sorted(pool,key=lambda g:(-rep_amp[g],g))[:B],
    'small_amplitude': sorted(pool,key=lambda g:(rep_amp[g],g))[:B],
  }
  result[str(B)]={p:run_arm(s,pool,rep_amp,seed,v,limit=max_select) for p,v in selectors.items()}
 summary['arms']=result
 return summary

if __name__=='__main__':
 results=[]
 for seed in range(201,213):
  t=time.time();r=run(seed);results.append(r)
  print('SEED',seed,'pool',r['candidate_orbits'],'base_queries',r['preselection_amplitude_queries'],'results',[(B,p,round(v['deltaE'],8),v['total_amp_queries']) for B,rows in r['arms'].items() for p,v in rows.items()],'time',round(time.time()-t,2),flush=True)
 summary={}
 for B in ('100','300'):
  summary[B]={}
  for p in ('random','large_amplitude','small_amplitude','sign_only_field','max_frustrated','observed_signed'):
   summary[B][p]={f:float(np.mean([r['arms'][B][p][f] for r in results])) for f in ('deltaE','n_orbits','n_states','extra_amp_queries','total_amp_queries','wrong_ed_weight_change')}
 out={'scope':'20-site guide with global FN amplitudes exposed only as on-demand oracle; sampler-discovered candidates and orbit-exact local-neighbor scores; exploratory 4-chain test','summary':summary,'per_seed':results,'limitations':['Data backend is globally precomputed exact FN, so amplitude availability is still an oracle; neural guide needed for scaling.','Metropolis uses local swaps targeting amp^1, not FN walker / GFMC.','Greedy fixed-amplitude energy monotonicity does not imply convergence after amplitude updates, nor neural sign learning.','Symmetry orbit energy score uses identical amplitudes under the translations+global spin inversion.','Evaluation ED labels only used after selecting flips.']}
 (P/'j1j2_budgeted_orbit20_results.json').write_text(json.dumps(out,indent=2))
 print('SUMMARY',json.dumps(summary,indent=2),flush=True)