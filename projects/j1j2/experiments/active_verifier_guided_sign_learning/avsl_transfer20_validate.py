"""Independent direct-H validation of AVSL 20-site symmetry-orbit greedy energy-gain scores."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,json,sys
sys.path.insert(0,'/mnt/data')
import avsl_transfer20 as q
from pathlib import Path
D=json.loads(Path('/mnt/data/avsl_transfer20_results.json').read_text())
a=q.AMPS;diag=q.D0['diag'];sts=q.STATES
Hbond=[]
for mask,h,i,j in q.BONDS:
    idx=np.flatnonzero(((sts>>i)^(sts>>j))&1)
    iy=q.LOOK[sts[idx]^np.uint32(mask)]
    Hbond.append((idx,iy,h))

def energy(sign):
    u=sign*a;out=float(np.dot(diag,a*a))
    for ix,iy,h in Hbond:out+=float(h*np.dot(u[ix],u[iy]))
    return out
rng=np.random.default_rng(3141)
rows=[];err=[]
for trial in D['test_episodes'][::5]:
    for chamber in ('marshall','post_krylov'):
        s0=q.SM.copy() if chamber=='marshall' else q.SK.copy()
        E0=energy(s0)
        for method in trial['arms'][chamber]:
            entry=trial['arms'][chamber][method];keys=entry['selected_keys']
            b=q.Backend()
            inds=[]
            for g in keys:
                b.key(g)
                inds.extend(q.LOOK[list(b.members[g])])
            after=s0.copy();after[np.asarray(inds,dtype=int)]*=-1
            E1=energy(after); delta=E0-E1
            e=delta-entry['fixed_a_gain'];err.append(abs(e))
            rows.append(dict(seed=trial['seed'],chamber=chamber,method=method,n_flips=len(keys),stored_gain=entry['fixed_a_gain'],direct_gain=delta,abs_error=abs(e)))
assert max(err)<1e-9,max(err)
# Check actual guide symmetry invariance for random complete orbits.
for k in rng.choice(q.STATES,100,replace=False):
    b=q.Backend();g=b.key(int(k));ii=q.LOOK[list(b.members[g])]
    assert np.ptp(q.AMPS[ii])<2e-9,(g,np.ptp(q.AMPS[ii]))
    for sg in [q.SM,q.SK]:
        assert len(set(sg[ii]))==1,(g,len(set(sg[ii])))
P=Path('/mnt/data/avsl_transfer20_validation.json');P.write_text(json.dumps({'validation_rows':rows,'max_abs_energy_error':max(err),'symmetry_orbits_checked':100},indent=2))
print('Direct energy checks',len(rows),'max |diff|',max(err),'symmetry sample 100 groups OK')