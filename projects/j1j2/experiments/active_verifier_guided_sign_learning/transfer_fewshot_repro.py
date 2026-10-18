#!/usr/bin/env python3
"""Few-shot post-Krylov adaptation of a Marshall-trained 20-site AVSL ranker.

Prerequisites in working directory:
- avsl_transfer20.py (same geometry and train/test split)
- sign20_training_data.npz, produced by the archived preparation code
- avsl_transfer20_results.pt, from source Marshall train:
  python avsl_transfer20.py --train-episodes 18 --test-episodes 20
      --train-budget 110 --verify-budget 22 --epochs 1000 --seed 19

Run from a scratch directory with those files, e.g.
  python transfer_fewshot_repro.py --per-episode 20 --out result.json

No ED sign labels are used in training. This is an exact amplitude-oracle
post-hoc *ranking* benchmark, NOT deployable GFMC, RL, or an FN energy gain.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import avsl_transfer20 as T

p=argparse.ArgumentParser()
p.add_argument('--per-episode',type=int,default=20,choices=[6,20,40,110])
p.add_argument('--steps',type=int,default=700)
p.add_argument('--out',default='fewshot_repro_results.json')
args=p.parse_args()
old=torch.load('avsl_transfer20_results.pt',weights_only=False,map_location='cpu')
torch.set_num_threads(2)

# Exactly the fixed train/test frontier and orbit splits of original transfer.
train=[];seen=set()
for seed in range(20260,20278):
    db,roots,pool,_=T.frontier(seed,140)
    full,geo,_=T.local_features(db,roots,pool,'post_krylov')
    ii=[i for i,g in enumerate(pool) if T.stable_split(g)!=0 and g not in seen]
    selected=np.random.default_rng(seed*47).permutation(ii)[:110]
    yy=[]
    for j in selected:
        g=pool[j]
        gain,_=T.score(db,g,'post_krylov')
        yy.append(float(gain>1e-13))
        seen.add(g)
    train.append((full[selected][:args.per_episode],geo[selected][:args.per_episode],
                  np.asarray(yy[:args.per_episode],dtype=np.float32)))
tests=[]
for seed in range(30260,30280):
    db,roots,pool,_=T.frontier(seed,140)
    candidates=[g for g in pool if T.stable_split(g)==0]
    full,geo,_=T.local_features(db,roots,pool,'post_krylov')
    index={g:i for i,g in enumerate(pool)}
    tests.append((seed,db,candidates,full[[index[g] for g in candidates]],
                  geo[[index[g] for g in candidates]]))

def fit(features,labels,kind,mode,seed):
    torch.manual_seed(seed)
    dim=features.shape[1]
    model=nn.Sequential(nn.Linear(dim,32),nn.Tanh(),nn.Linear(32,32),
                        nn.Tanh(),nn.Linear(32,1))
    if mode=='warm':
        model.load_state_dict(old[kind])
        mu=old[kind+'_mu'];std=old[kind+'_std'];lr=.0008
    else:
        mu=features.mean(axis=0);std=np.maximum(features.std(axis=0),.1);lr=.003
    xx=torch.tensor((features-mu)/std,dtype=torch.float32)
    yy=torch.tensor(labels)
    pos=float(labels.mean())
    weight=float(min(15,max(1,((1-pos)/max(pos,1e-5))**.5)))
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.002)
    rng=np.random.default_rng(seed*389)
    for step in range(args.steps):
        inds=rng.choice(len(labels),min(256,len(labels)),replace=True)
        loss=criterion(model(xx[inds]).flatten(),yy[inds])
        opt.zero_grad();loss.backward();opt.step()
    model.eval()
    return (model,mu,std)

def evaluate(model,kind):
    norm=float(np.dot(T.AMPS,T.AMPS))
    gains=[]
    for seed,db,candidates,full,geo in tests:
        logits=T.predict(model,full if kind=='full' else geo)
        top=[candidates[j] for j in np.argsort(-logits,kind='stable')[:22]]
        raw={};edges={}
        for g in top:
            raw[g],edges[g]=T.score(db,g,'post_krylov')
        gain,_=T.greedy(raw,edges,top)
        gains.append(float(gain/norm))
    return gains

records=[]
for kind in ('full','geo'):
    i=0 if kind=='full' else 1
    x=np.concatenate([r[i] for r in train])
    y=np.concatenate([r[2] for r in train])
    for mode in ('warm','scratch'):
        for seed in (19,29):
            model=fit(x,y,kind,mode,seed)
            gains=evaluate(model,kind)
            records.append(dict(features=kind,mode=mode,seed=seed,
                                labels=len(y),positives=int(y.sum()),
                                mean_gain=float(np.mean(gains)),episode_gains=gains))
            print(kind,mode,seed,'mean fixed-a gain',records[-1]['mean_gain'],flush=True)

Path(args.out).write_text(json.dumps({
    'protocol':'20 heldout orbit-pool episodes, 22 expensive orbit gains per episode, exact 20-site FN amplitude oracle',
    'source_training_labels':1980,'post_labels':len(y),'post_positive_labels':int(y.sum()),
    'FN_reoptimization_performed':False,'results':records},indent=2))
