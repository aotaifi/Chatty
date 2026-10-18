#!/usr/bin/env python3
"""Symmetry-orbit Ising-graph sign correction with *exact* marginal energy gate.
Controlled exact 20-site diagnostic. No ED target used to select flips.
The orbit graph requires full-space guide a,s and is NOT a scalable algorithm on its own.
"""
import os,json
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.sparse import coo_matrix
from pathlib import Path
P=Path('/mnt/data')
src=(P/'j1j2_orbit_priority20.py').read_text();exec(src[:src.index('\nres=[]')]);P=Path('/mnt/data')
from scipy.sparse import csr_matrix
rind=[];cind=[];vdata=[]
for i,j,h in bonds:
 active=(((states>>i)^(states>>j))&1).astype(bool)
 ix=np.flatnonzero(active);jx=lookup[states[ix]^np.uint32((1<<i)|(1<<j))]
 rind.append(grp[ix]);cind.append(grp[jx]);vdata.append(h*s[ix]*s[jx]*a[ix]*a[jx])
W=coo_matrix((np.concatenate(vdata),(np.concatenate(rind),np.concatenate(cind))),shape=(ng,ng)).tocsr()
W.setdiag(0);W.eliminate_zeros()
err=max(abs(W-W.T).data,default=0.)
assert err<1e-11,err
initial_local=np.asarray(W@np.ones(ng))
# check exact orbit sign energy gradients for random sets
random=np.random.default_rng(11)
checks=[]
for k in (1,5,10):
 gs=random.choice(ng,size=k,replace=False);sel=np.concatenate([orbit_indices[z] for z in gs]);eta=np.ones(ng);eta[gs]*=-1
 dE_ising=float(eta@W@eta-np.ones(ng)@W@np.ones(ng))
 dE_direct=deltaE(sel)
 checks.append(dict(k=k,dE_ising=dE_ising,dE_direct=dE_direct,err=abs(dE_ising-dE_direct)))
assert max(x['err'] for x in checks)<2e-8,checks
print('GRAPH',ng,'n_sparse',W.nnz,'symmetry_error',err,'energy checks',checks,flush=True)
# Avoid expensive CSC retrieval by precompute CSC; O(degree) incremental gain update.
Wc=W.tocsc();policies=['FN_suppression','Krylov_field','individual_flip_gain','orbital_exact_gain','greedy_energy']
rankings={**scoreorbit,'orbital_exact_gain':4*initial_local}
results=[]
for seed in range(8):
 rng=np.random.default_rng(550+seed);prob=np.maximum(a,1e-250);prob/=prob.sum();seeds=np.unique(rng.choice(D,size=600,replace=True,p=prob))
 seedmask=np.zeros(D,bool);seedmask[seeds]=True;candidate=seedmask.copy()
 for ix,jx in neighbor_states:candidate[jx[seedmask[ix]]]=True
 pool=np.unique(grp[candidate]);
 for policy in policies:
  if policy!='greedy_energy':
   ordered=pool[np.lexsort((pool,-rankings[policy][pool]))]
   for budget in (5,10,20,40):
    gs=ordered[:budget];selected=np.concatenate([orbit_indices[z] for z in gs])
    results.append(dict(seed=seed,policy=policy,orbits=budget,deltaE=deltaE(selected),n_selected_states=len(selected),n_true_krylov=int(target[selected].sum()),n_ed_wrong_after=float(np.sum(p[old_wrong])+np.sum((1-2*old_wrong[selected])*p[selected]))))
  else:
   eta=np.ones(ng,dtype=np.float64);field0=initial_local.copy();chosen=[];used=np.zeros(ng,bool);dE_trajectory=0.
   for step in range(1,41):
    gain=4*eta*field0
    gain[used]=-np.inf
    local_pool=pool[np.argmax(gain[pool])];best=float(gain[local_pool])
    if best<=0:break
    # ΔE = -best, update signs and neighbors' local fields, then allow further corrections.
    chosen.append(int(local_pool));used[local_pool]=True;dE_trajectory-=best
    oldeta=eta[local_pool];eta[local_pool]*=-1
    start,end=Wc.indptr[local_pool:local_pool+2]
    field0[Wc.indices[start:end]]-=2*oldeta*Wc.data[start:end]
    if step in (5,10,20,40):
     selected=np.concatenate([orbit_indices[z] for z in chosen]);dE_check=deltaE(selected)
     assert abs(dE_trajectory-dE_check)<1e-8,(dE_trajectory,dE_check)
     results.append(dict(seed=seed,policy=policy,orbits=step,deltaE=dE_check,n_selected_states=len(selected),n_true_krylov=int(target[selected].sum()),n_ed_wrong_after=float(np.sum(p[old_wrong])+np.sum((1-2*old_wrong[selected])*p[selected]))))
summary={}
for budget in (5,10,20,40):
 summary[str(budget)]={}
 for pol in policies:
  r=[row for row in results if row['orbits']==budget and row['policy']==pol]
  if r:summary[str(budget)][pol]={k:float(np.mean([x[k] for x in r])) for k in ('deltaE','n_selected_states','n_true_krylov','n_ed_wrong_after')}
print('SUMMARY',json.dumps(summary,indent=2),flush=True)
with open(P/'j1j2_orbit_greedy20_results.json','w') as f:json.dump({'graph':dict(n_orbits=ng,nnz=W.nnz,energy_checks=checks),'summary':summary,'per_seed':results,'warning':'uses globally known guide amplitudes and scores; this is exact energy sign optimization on 20 sites, not scalable yet'},f,indent=2)