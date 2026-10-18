#!/usr/bin/env python3
"""Unlabeled sign-frustration-weighted NCE on walker-generated mixed and Jastrow^2 draws.
Same weight is applied to both NCE classes at each x, preserving the density-ratio
population minimizer; changes only finite-data fitting emphasis. No FN amplitude labels.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
import json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import j1j2_walker_only_amplitude20 as W
from j1j2_walker_only_amplitude20 import M,T
P=Path('/mnt/data');torch.set_num_threads(2)

def local_flip_pressure(configs):
 ix=M.lookup[configs];assert (ix>=0).all();ans=np.zeros(len(ix));
 for mask,h,i,j in M.bonds:
  y=configs^int(mask);iy=M.lookup[y];valid=iy>=0;safe=np.maximum(0,iy)
  ans += np.where(valid,h*W.SIGNS[ix]*W.SIGNS[safe]*np.exp(W.LOGG0[safe]-W.LOGG0[ix]),0.)
 return ans

def train_focus(pos,neg,steps=1200,seed=81,strength=4.0):
 torch.manual_seed(seed);rng=np.random.default_rng(seed)
 net=T.Net();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-4)
 # Weight by positive sign-pressure evaluated only on sampled configurations.
 up,un=np.unique(pos),np.unique(neg)
 allu=np.unique(np.r_[up,un]);pressure=local_flip_pressure(allu)
 score=np.maximum(pressure,0.)/(1+np.maximum(pressure,0.))
 weights=1+strength*score
 lookupweights=np.zeros(1<<W.N,dtype=np.float32);lookupweights[allu]=weights.astype(np.float32)
 # Same exact holdout/orbit selection as the unweighted training function.
 orbit=M.Sampler();upos=np.unique(pos);uneg=np.unique(neg)
 positive_orbits=np.array([orbit.orbit(int(x)) for x in upos]);negative_orbits=np.array([orbit.orbit(int(x)) for x in uneg]);allorbits=np.unique(np.r_[positive_orbits,negative_orbits]);rng.shuffle(allorbits);held=set(allorbits[:max(1,len(allorbits)//10)])
 pm=np.array([orbit.orbit(int(x)) not in held for x in pos]);nm=np.array([orbit.orbit(int(x)) not in held for x in neg]);ptr=pos[pm];ntr=neg[nm];pva=pos[~pm];nva=neg[~nm]
 trace=[];start=time.monotonic()
 for step in range(steps):
  a=ptr[rng.integers(len(ptr),size=128)];b=ntr[rng.integers(len(ntr),size=128)]
  ids=np.r_[M.lookup[a],M.lookup[b]];x=np.r_[a,b];wt=torch.from_numpy(lookupweights[x])
  logits=net.symmetric(T.X[ids]);labels=torch.cat([torch.ones(128),torch.zeros(128)])
  loss=(nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')*wt).mean()/wt.mean()
  opt.zero_grad();loss.backward();opt.step()
  if (step+1)%400==0:
   with torch.no_grad():
    na=min(len(pva),500);nb=min(len(nva),500)
    z=net.symmetric(T.X[M.lookup[np.r_[pva[:na],nva[:nb]]]])
    vl=nn.functional.binary_cross_entropy_with_logits(z,torch.cat([torch.ones(na),torch.zeros(nb)])).item()
   trace.append(dict(step=step+1,loss=float(loss),heldout_unweighted_bce=float(vl)))
   print('TRAIN',trace[-1],flush=True)
 return net,dict(n_positive=int(len(pos)),n_negative=int(len(neg)),steps=steps,strength=strength,trace=trace,mean_weight=float(weights.mean()),median_weight=float(np.median(weights)),elapsed_s=time.monotonic()-start)

if __name__=='__main__':
 t=time.monotonic();p,pr=W.FNProjector().run(seed=53,nwalk=3500,burn=450,steps=450,thin=4)
 q,qr=W.sample_reference(seed=61,nchains=3500,burn=900,steps=450,thin=4)
 print('SAMPLING',len(p),len(q),'time',round(time.monotonic()-t,1),flush=True)
 net,tr=train_focus(p,q)
 torch.save(dict(model=net.state_dict(),alpha=W.ALPHA),P/'j1j2_walker_focus20_model.pt')
 (P/'j1j2_walker_focus20_train.json').write_text(json.dumps(dict(sampling=pr,reference=qr,training=tr),indent=2))
 print('DONE',round(time.monotonic()-t,1),flush=True)