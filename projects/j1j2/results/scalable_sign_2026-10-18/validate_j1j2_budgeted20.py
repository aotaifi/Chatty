"""Independent post-selection full-space energy audit; this script is not used for choosing corrections."""
import numpy as np,json
import j1j2_local_query_diagnostic as M
r=json.load(open('/mnt/data/j1j2_budgeted_orbit20_results.json'))
a=M.amplitude;s=M.signs;D=M.D
psi=a*s
# construct Hamiltonian diagonal/offdiagonal expectation independently from selections
def energy(v):
 out=float(np.dot(M.diag,v*v))
 for bit,h,i,j in M.bonds:
  active=((M.states>>i)^(M.states>>j))&1
  ix=np.flatnonzero(active); jx=M.lookup[M.states[ix]^bit]
  out+=float(h*np.dot(v[ix],v[jx]))
 return out
E0=energy(psi);checks=[]
for trial in r['per_seed'][:3]:
 for B in ('100','300'):
  for policy in ('random','observed_signed','sign_only_field'):
   x=M.Sampler();choose=trial['arms'][B][policy]['selected_keys']
   states=set()
   for k in choose:
    assert x.orbit(k)==k
    states.update(x.members[k])
   ix=np.array([x.idx(j) for j in states],dtype=np.int32)
   next_psi=psi.copy();next_psi[ix]*=-1
   observed=energy(next_psi)-E0
   expected=trial['arms'][B][policy]['deltaE']
   checks.append(dict(seed=trial['seed'],B=B,policy=policy,from_direct_E=observed,from_local_gate=expected,abs_error=abs(observed-expected)))
mx=max(q['abs_error'] for q in checks)
print('Reference energy:',E0,'max_error:',mx,'checked',len(checks));assert mx<1e-9
json.dump(checks,open('/mnt/data/validate_j1j2_budgeted20_results.json','w'),indent=2)