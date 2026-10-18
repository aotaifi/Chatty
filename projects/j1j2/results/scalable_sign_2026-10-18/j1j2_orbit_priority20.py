#!/usr/bin/env python3
"""Symmetry-respecting controlled ranking of 20-site FN/Krylov candidate corrections.
Uses known guide amplitudes, exact FN suppression and ED labels for evaluation only.
Does not establish realistic acquisition cost (whole configuration space already known).
"""
import os, json
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.stats import rankdata
P='/mnt/data/'
d=np.load(P+'sign20_training_data.npz');u=np.load(P+'sign20_suppression.npz');audit=np.load(P+'audit_sign20_targets.npz')
states=d['states'].astype(np.uint32);a=d['amp'];s=d['guide'];psi0=d['psi0'];rate=u['rate'];target=audit['optimal_flip'].astype(bool)
N=20;Lx,Ly=4,5;D=len(states)
lookup=np.full(1<<20,-1,dtype=np.int32);lookup[states]=np.arange(D)
def site(x,y):
 q,r=divmod(y,Ly);return (x-q)%Lx + Lx*r
bonds=[]
for yy in range(Ly):
 for xx in range(Lx):
  i=site(xx,yy)
  for dx,dy,h in [(1,0,.5),(0,1,.5),(1,1,.25),(-1,1,.25)]:
   bonds.append((i,site(xx+dx,yy+dy),h))
off=np.zeros(D)
neighbor_states=[]
for i,j,h in bonds:
 active=(((states>>i)^(states>>j))&1).astype(bool)
 ix=np.flatnonzero(active);jx=lookup[states[ix]^np.uint32((1<<i)|(1<<j))]
 off[ix]+=h*s[ix]*s[jx]*a[jx]
 neighbor_states.append((ix,jx))
field=d['diag']+off/np.maximum(a,1e-290)
gain=4*a*off
p=psi0**2
old_wrong=(s*psi0<0)
w_old=float(p[old_wrong].sum())

def deltaE(sel):
 inside=np.zeros(D,dtype=bool);inside[sel]=True
 cross_with_internal_twice=float(np.sum(a[sel]*off[sel]))
 interior=0.
 for i,j,h in bonds:
  active=(((states[sel]>>i)^(states[sel]>>j))&1).astype(bool)
  ix=sel[active]; jx=lookup[states[ix]^np.uint32((1<<i)|(1<<j))]
  both=inside[jx];ix=ix[both];jx=jx[both]
  interior+=float(np.sum(h*s[ix]*s[jx]*a[ix]*a[jx]))
 return -4*(cross_with_internal_twice-interior)

def directE(flips):
 y=s*a;y=y.copy();y[flips]*=-1
 E=float(np.dot(d['diag'],a*a))
 for i,j,h in bonds:
  active=(((states>>i)^(states>>j))&1).astype(bool)
  ix=np.flatnonzero(active);jx=lookup[states[ix]^np.uint32((1<<i)|(1<<j))]
  E+=float(h*np.dot(y[ix],y[jx]))
 return E
probe=np.flatnonzero(target)[:400]
ver={'initialE':directE(np.array([],dtype=int)),'direct_change':directE(probe)-directE(np.array([],dtype=int)),'analytic_change':deltaE(probe)}
assert abs(ver['direct_change']-ver['analytic_change']) < 1e-9,ver
print('Energy verification',ver,flush=True)
# translation and global spin flip symmetry orbit keys
coords={site(xx,yy):(xx,yy) for yy in range(Ly) for xx in range(Lx)}
keys=np.full(D,np.uint32((1<<N)-1),dtype=np.uint32)
for dy in range(Ly):
 for dx in range(Lx):
  perm=[site(coords[i][0]+dx,coords[i][1]+dy) for i in range(N)]
  v=np.zeros(D,np.uint32)
  for i,j in enumerate(perm):v|=((states>>i)&1)<<int(j)
  keys=np.minimum(keys,np.minimum(v,v^np.uint32((1<<N)-1)))
unique,grp=np.unique(keys,return_inverse=True)
ng=len(unique);orbit_indices=[np.flatnonzero(grp==i) for i in range(ng)]
print('Symmetry orbits',ng,'sizes percentiles',np.quantile(np.bincount(grp),[0,.5,1]),flush=True)
# Check invariance: all members of an orbit should have identical target and nearly same scores.
quality={}
for name,z in {'field':field,'rate':rate,'gain':gain,'amplitude':a,'target':target.astype(float)}.items():
 avg=np.bincount(grp,weights=z)/np.bincount(grp)
 dev=np.abs(z-avg[grp])
 quality[name]=dict(max_abs_deviation=float(np.max(dev)), rms_abs_deviation=float(np.sqrt(np.mean(dev**2))))
print('Orbit invariance',json.dumps(quality),flush=True)
# Scores of orbits (average over members; each member physically equivalent)
scoreconf={'FN_suppression':rate,'Krylov_field':field,'individual_flip_gain':gain,'small_amplitude':-np.log(np.maximum(a,1e-250))}
scoreorbit={name:np.bincount(grp,weights=z)/np.bincount(grp) for name,z in scoreconf.items()}
res=[]
for seed in range(8):
 rng=np.random.default_rng(550+seed)
 prob=np.maximum(a,1e-250);prob/=prob.sum()
 seeds=np.unique(rng.choice(D,size=600,replace=True,p=prob))
 mask=np.zeros(D,bool);mask[seeds]=True
 seedmask=mask.copy()
 for ix,jx in neighbor_states:mask[jx[seedmask[ix]]]=True
 pool=np.unique(grp[np.flatnonzero(mask)])
 total=int(np.sum(target[np.isin(grp,pool)])); totalwt=float(np.sum(p[target & np.isin(grp,pool)]))
 for budget in (5,10,20,40):
  for name,z in [*scoreorbit.items(),('random',rng.uniform(size=ng))]:
   gs=pool[np.lexsort((pool,-z[pool]))[:budget]]
   sel=np.concatenate([orbit_indices[i] for i in gs])
   flips=int(target[sel].sum());ww=float(p[sel][target[sel]].sum())
   false=int(np.sum(target[sel] & ~old_wrong[sel]))
   res.append(dict(seed=seed,pool_n_orbits=len(pool),selected_orbits=budget,selected_states=len(sel),method=name,precision=float(flips/len(sel)),target_recall=float(flips/total),target_weight_recall=float(ww/totalwt) if totalwt else None,incorrect_ed_flips=false,deltaE=deltaE(sel),wrong_weight_after=w_old+float(np.sum((1-2*old_wrong[sel])*p[sel]))))
summary={}
for budget in (5,10,20,40):
 summary[str(budget)]={}
 for name in [*scoreorbit,'random']:
  rr=[v for v in res if v['selected_orbits']==budget and v['method']==name]
  summary[str(budget)][name]={k:float(np.mean([v[k] for v in rr])) for k in ('selected_states','precision','target_recall','target_weight_recall','incorrect_ed_flips','deltaE','wrong_weight_after')}
print('Orbit summary',json.dumps(summary,indent=2),flush=True)
result={'verification':ver,'n_orbits':ng,'orbit_invariance':quality,'per_seed':res,'summary':summary,'notes':['whole symmetry orbits flipped, including states outside candidate pool (symmetry closure)','score computed using exact full-space FN and guide amplitudes; oracle diagnostic','ED used for scoring only, not ranking','at equal number of orbits, configurations considered similar but not identical in count']}
with open(P+'j1j2_orbit_priority20_results.json','w') as f:json.dump(result,f,indent=2)