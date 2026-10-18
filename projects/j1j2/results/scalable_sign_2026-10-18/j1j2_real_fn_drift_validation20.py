#!/usr/bin/env python3
"""Independent full-space H expectation audit of old-selected flips at next FN amplitude."""
import numpy as np
import j1j2_real_fn_drift20 as R
import j1j2_budgeted_orbit20 as B
import j1j2_local_query_diagnostic as M
from pathlib import Path

base = np.asarray(R.NEXT*M.signs,dtype=float)
states=M.states;lookup=M.lookup

def full_hop_energy(wave):
 energy=0.
 for mask,h,_,_ in M.bonds:
  ix=lookup[np.bitwise_xor(states,mask)]
  good=ix>=0
  energy+=h*float(np.dot(wave[good],wave[ix[good]]))
 return energy

base_e=full_hop_energy(base)
for seed in [201,211]:
 s,pool,amps,proxy,info=B.generate(seed)
 prio=sorted(pool,key=lambda g:(-proxy['observed_signed'][g],g))[:300]
 chosen,margins=R.choose(prio,s)
 n=R.NextGuide();n.keys=s.keys.copy();n.members=s.members.copy()
 flips=np.zeros(M.D,bool)
 for g in chosen:
  flips[np.array([s.idx(x) for x in s.members[g]],int)]=True
 after=base*np.where(flips,-1,1)
 direct=base_e-full_hop_energy(after)
 local=R.eval_new(chosen,n)
 print('seed',seed,'n_orbits',len(chosen),'energy_diff direct',direct,'local',local,'difference',direct-local)
 assert np.isclose(direct,local,atol=1e-10,rtol=1e-8)