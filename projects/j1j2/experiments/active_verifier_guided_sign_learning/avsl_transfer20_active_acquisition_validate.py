#!/usr/bin/env python3
"""Independent direct-Hamiltonian check of active acquisition experiment outputs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,json
from pathlib import Path
import avsl_transfer20 as T

D=json.load(open('/mnt/data/avsl_transfer20_active_acquisition_results.json'))
assert D['label_budget_each']==360
for arm,rec in D['arms'].items():
    assert rec['labels']==360
    assert sum(c['label_queries'] for c in rec['training_cost'])==360
    for run in rec['models']:
        assert len(run['trials'])==20
        for trial in run['trials']:
            assert all(T.stable_split(g)==0 for g in trial['top'])

states=T.STATES;a=T.AMPS;diag=T.D0['diag'];norm=float(np.dot(a,a))
hops=[]
for mask,h,i,j in T.BONDS:
    ix=np.flatnonzero(((states>>i)^(states>>j))&1)
    iy=T.LOOK[states[ix]^np.uint32(mask)]
    hops.append((ix,iy,h))
def energy(s):
    u=s*a;v=float(np.dot(diag,a*a))
    for ix,iy,h in hops:v+=float(h*np.dot(u[ix],u[iy]))
    return v
E0=energy(T.SK);checks=[]
for arm,rec in D['arms'].items():
    for run in rec['models']:
        for t in run['trials'][::4]:
            db=T.Backend();inds=[]
            for g in t['selected_keys']:
                db.key(g);inds.extend(T.LOOK[list(db.members[g])])
            s=T.SK.copy();s[np.asarray(inds,dtype=int)]*=-1
            true=(E0-energy(s))/norm
            assert abs(true-t['gain'])<1e-9,(arm,true,t['gain'])
            checks.append(dict(arm=arm,seed=run['seed'],episode=t['seed'],
                               direct=true,stored=t['gain'],err=abs(true-t['gain'])))
res={'n_checked':len(checks),'max_abs_error':max(z['err'] for z in checks),'checks':checks}
Path('/mnt/data/avsl_transfer20_active_acquisition_validation.json').write_text(json.dumps(res,indent=2))
print('VALIDATED',len(checks),'max error',res['max_abs_error'])
