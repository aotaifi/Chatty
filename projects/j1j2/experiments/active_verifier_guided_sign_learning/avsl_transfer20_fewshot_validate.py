"""Independently verify a fixed subset of few-shot selected-orbit energy changes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np, json
from pathlib import Path
import avsl_transfer20 as T
D=json.load(open('/mnt/data/avsl_transfer20_fewshot_results.json'))
a=T.AMPS;diag=T.D0['diag'];sts=T.STATES
Hbond=[]
for mask,h,i,j in T.BONDS:
 ix=np.flatnonzero(((sts>>i)^(sts>>j))&1)
 iy=T.LOOK[sts[ix]^np.uint32(mask)]
 Hbond.append((ix,iy,h))
def energy(sign):
 u=sign*a;out=float(np.dot(diag,a*a))
 for ix,iy,h in Hbond:out+=float(h*np.dot(u[ix],u[iy]))
 return out
E0=energy(T.SK)
norm=float(np.dot(a,a))
rows=[]
for b in ('6','20','40','110'):
 for mode in ('warm','scratch'):
  for run in D['arms'][b]['variants'][mode]['runs']:
   for t in run['trials'][:4]:
    db=T.Backend();chosen=[]
    # Recompute exact per-orbit scores and greedy acceptance in stored TOP (22)
    raw={};edges={}
    for g in t['top']:
     db.key(g);raw[g],edges[g]=T.score(db,g,'post_krylov')
    gain,chosen=T.greedy(raw,edges,t['top'])
    inds=[]
    for g in chosen:
     db.key(g);inds.extend(T.LOOK[list(db.members[g])])
    after=T.SK.copy();after[np.asarray(inds,dtype=int)]*=-1
    true=(E0-energy(after))/norm;stored=t['gain']
    rows.append(dict(budget=b,mode=mode,seed=run['seed'],episode=t['seed'],stored=stored,direct=true,abs_error=abs(stored-true),accepted=len(chosen)))
mx=max(r['abs_error'] for r in rows)
assert mx<1e-9,mx
out={'n_independent_checks':len(rows),'max_abs_error':mx,'rows':rows}
Path('/mnt/data/avsl_transfer20_fewshot_validation.json').write_text(json.dumps(out,indent=2))
print('VALIDATED',len(rows),'max error',mx)