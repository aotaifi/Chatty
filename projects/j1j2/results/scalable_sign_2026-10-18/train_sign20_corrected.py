#!/usr/bin/env python3
"""Matched-pool CPU pilot: actual Krylov labels, one-hop neighbor support, symmetry holdout.
Ablation compares uniform baseline, measured FN-suppression priority, individual-flip energy gain priority.
WARNING: scoring uses full-space exact FN/guide amplitude arrays, so this is an oracle priority experiment.
"""
import os,json,time
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np, torch
from torch import nn
torch.set_num_threads(2)
P=Path('/mnt/data');started=time.time()
# Reuse verified geometry, local scores, symmetry orbits and energy evaluator, but not ranking experiments.
src=(P/'j1j2_orbit_priority20.py').read_text()
exec(src[:src.index('\nres=[]')])
P=Path('/mnt/data')
# Actual energy-optimized Krylov update, not the original g>0 proxy.
newsign=s.copy();newsign[target]*=-1
marshall=np.where(np.bitwise_count(states&int(d['maskA'][0]))%2==0,1,-1).astype(np.int8)
y=(newsign*marshall>0).astype('float32')
# Split entire symmetry orbits, not individual states.
rng=np.random.default_rng(812);shuffle=rng.permutation(ng);held=np.zeros(ng,bool);held[shuffle[:ng//5]]=True
is_test=held[grp];poolstates=np.flatnonzero(~is_test);test=np.flatnonzero(is_test)
# Same candidate pool for all methods: amplitude-tempered seeds + one-hop neighbors.
rng=np.random.default_rng(898);pi=np.maximum(a[poolstates],1e-250);pi/=pi.sum()
seeds=np.unique(rng.choice(poolstates,size=600,replace=True,p=pi))
seedmask=np.zeros(D,bool);seedmask[seeds]=True
mask=seedmask.copy()
for ix,jx in neighbor_states:mask[jx[seedmask[ix]]]=True
mask &= ~is_test
pool=np.unique(grp[mask]);rep=np.array([orbit_indices[g][0] for g in pool]);M=len(rep)
W={'baseline':np.ones(M,dtype=float)/M}
for method,z in [('suppression',np.maximum(scoreorbit['FN_suppression'][pool],0)),('energy_gain',np.maximum(scoreorbit['individual_flip_gain'][pool],0))]:
 z=np.sqrt(z);q=z/(z.sum()+1e-300)
 W[method]=.7*W['baseline']+.3*q
print('SETUP',json.dumps(dict(n_sites=N,states=D,orbits=ng,pool_orbits=M,seed_states=len(seeds),test_states=len(test),methods=list(W),target_flips=int(target.sum()))),flush=True)
X=torch.from_numpy(((((states[:,None]>>np.arange(N))&1).astype(np.float32))*2-1).copy());Y=torch.from_numpy(y)
nbr=np.array([[site(x+dx,yy+dy) for dy in [-1,0,1] for dx in [-1,0,1]] for yy in range(Ly) for x in range(Lx)])
nbrt=torch.from_numpy(nbr.astype(np.int64))
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.c0=nn.Linear(9,16);self.c1=nn.Linear(144,16);self.c2=nn.Linear(144,16);self.l1=nn.LayerNorm(16);self.l2=nn.LayerNorm(16);self.h=nn.Linear(16,1)
 def forward(self,x):
  h=torch.nn.functional.gelu(self.c0(x[:,nbrt]));h=h+torch.nn.functional.gelu(self.c1(self.l1(h)[:,nbrt].reshape(-1,N,144)));h=h+torch.nn.functional.gelu(self.c2(self.l2(h)[:,nbrt].reshape(-1,N,144)));return self.h(h).sum(1).squeeze(-1)/np.sqrt(N)
nsteps=int(os.environ.get('NSTEPS','1000'));nseeds=int(os.environ.get('NSEEDS','2'))
records=(json.loads((P/'sign20_corrected_cnn_results.json').read_text())['runs'] if os.getenv('APPEND_RESULTS')=='1' and (P/'sign20_corrected_cnn_results.json').exists() else []);ww=psi0**2;E_start=directE(np.array([],dtype=int));E_target=E_start+deltaE(np.flatnonzero(target))
for seed in range(int(os.getenv('START_SEED','0')),int(os.getenv('START_SEED','0'))+nseeds):
 for method in ('baseline','suppression','energy_gain'):
  t0=time.time();torch.manual_seed(101+seed);model=Net();opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
  r=np.random.default_rng(3001+seed)
  for step in range(nsteps):
   ids=rep[r.choice(M,size=128,replace=True,p=W[method])];xb=X[ids];yb=Y[ids]
   if r.random()<.5:xb=-xb
   loss=nn.functional.binary_cross_entropy_with_logits(model(xb),yb)
   opt.zero_grad();loss.backward();opt.step()
   if (step+1)%250==0:print('TRAIN',seed,method,step+1,round(float(loss.detach()),6),'sec',round(time.time()-t0,1),flush=True)
  outs=[];model.eval()
  with torch.no_grad():
   for k in range(0,D,2048):
    xb=X[k:k+2048];outs.append(((model(xb)+model(-xb))/2).numpy())
  logits=np.concatenate(outs);pred=np.where(logits>=0,1,-1)*marshall
  err=(pred[test]!=newsign[test]);edwrong=(pred*psi0<0)
  ee=E_start+deltaE(np.flatnonzero(pred!=s))
  rec=dict(seed=seed,method=method,steps=nsteps,test_err=float(err.mean()),test_weighted_target_err=float(ww[test][err].sum()/ww[test].sum()),whole_wrong_ED_weight=float(ww[edwrong].sum()),energy_fixed_amp=float(ee),seconds=round(time.time()-t0,1))
  records.append(rec);print('RESULT',json.dumps(rec),flush=True)
  (P/'sign20_corrected_cnn_results.json').write_text(json.dumps(dict(protocol=dict(n_pool_orbits=M,n_train_states=int((~is_test).sum()),n_test_states=len(test),nsteps=nsteps,nseeds=nseeds,guide_energy=E_start,target_energy=E_target,guide_wrong_weight=float(ww[old_wrong].sum()),target_wrong_weight=float(ww[newsign*psi0<0].sum()),warning='all score arrays were computed globally using exact FN and guide amplitudes; prioritization still oracle-cost'),runs=records),indent=2))
print('DONE',round(time.time()-started,1),flush=True)