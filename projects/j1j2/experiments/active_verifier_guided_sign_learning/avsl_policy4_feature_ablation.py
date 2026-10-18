#!/usr/bin/env python3
"""Train/evaluate 4x4 AVSL with no geometric or no observed bond features."""
import json
import numpy as np
import torch
from pathlib import Path
import avsl_learned_policy4 as L
from torch import nn
class ApplyMask(nn.Module):
 def __init__(self,model,kept):
  super().__init__();self.model=model; self.kept=kept
 def forward(self,x):
  z=x.clone();z[:,self.kept==0]=0;return self.model(z)

def run():
 torch.set_num_threads(2)
 tr=[L.get_labeled_run(s,k,160,80) for k in [.75,.85,.9,1.,1.1] for s in range(300,309)]
 setups={
  'all':list(range(13)),
  'geometry_only':list(range(8,13)),
  'observed_bonds_only':list(range(8))+[12],
  'no_geometry_except_kappa':list(range(8))+[12],
  'without_kappa':list(range(12)),
 }
 res={}
 for name,keep in setups.items():
  keep=np.asarray(keep,dtype=int); mask=np.zeros(13,dtype=float);mask[keep]=1
  data=[{'X':r['X']*mask,'Y':r['Y']} for r in tr]
  net,mu,sigma,stat=L.train_model(data,42,800)
  net=ApplyMask(net,mask)
  rows=[L.evaluate_model(s,k,net,mu,sigma,B=22,n_seeds=160) for k in [.8,.9,1.] for s in range(2100,2112)]
  gains=np.array([r['methods']['learned']['gain_numerator'] for r in rows]);good=np.array([r['methods']['learned']['positive_tested'] for r in rows])
  res[name]={'mean_gain_numerator':float(gains.mean()),'mean_positive_tested':float(good.mean()),'train_count':stat['train_count'],'train_positive':stat['positive_count'],
             'wins_vs_signed':int(sum(r['methods']['learned']['gain_numerator']>r['methods']['signed']['gain_numerator']+1e-9 for r in rows)),
             'mean_signed_gain_numerator':float(np.mean([r['methods']['signed']['gain_numerator'] for r in rows]))}
  print(name,res[name],flush=True)
 Path('/mnt/data/avsl_policy4_feature_ablation_results.json').write_text(json.dumps(res,indent=2))
if __name__=='__main__':run()
