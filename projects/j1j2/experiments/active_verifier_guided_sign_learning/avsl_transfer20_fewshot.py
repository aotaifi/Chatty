#!/usr/bin/env python3
"""Counterfactual offline few-shot adaptation of the Marshall-trained AVSL candidate ranker.
Same 20-site chamber, amplitude guide, train-root episodes and orbit holdout as
avsl_transfer20.py; compares scratch training versus warm-start on only a few
new-chamber labels. The previous Marshall training cost is sunk, NOT free.
No label reads from 20% heldout orbits, but FN oracle amplitudes are callable.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time
from pathlib import Path
import numpy as np,torch
import avsl_transfer20 as T
from torch import nn

BUDGETS=[6,20,40,110]   # labels / training episode, 18 episodes
NTRAIN=18; EPOCH=700
OUT='/mnt/data/avsl_transfer20_fewshot_results.json'

# Build the same finite pre-existing offline labeling dataset as the full post-only comparison.
arrays=[]; unique=set();t0=time.time();counts=[]
for seed in range(20260,20260+NTRAIN):
    db, roots, pool,_=T.frontier(seed,n_root=140)
    f,geo,signed=T.local_features(db,roots,pool,'post_krylov')
    idx=[i for i,g in enumerate(pool) if T.stable_split(g)!=0 and g not in unique]
    rr=np.random.default_rng(seed*47).permutation(idx)[:110]
    y=[]
    for i in rr:
        g=pool[i];val,_=T.score(db,g,'post_krylov');y.append(float(val>1e-13));unique.add(g)
    arrays.append((f[rr],np.asarray(y,np.float32)))
    counts.append(dict(seed=seed,n=len(rr),positive=sum(y),unique_amplitudes=len(db.amps)))
print('DATA',len(unique),int(sum(y.sum() for _,y in arrays)),time.time()-t0,flush=True)

old=torch.load('/mnt/data/avsl_transfer20_results.pt',map_location='cpu',weights_only=False)
torch.set_num_threads(2)
def build_model(seed,init=None):
    torch.manual_seed(seed)
    m=nn.Sequential(nn.Linear(13,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))
    if init is not None:m.load_state_dict(init)
    return m

def fit(X,Y,method,seed):
    if method=='warm':
        m=build_model(seed,old['full']);mu=old['full_mu'];std=old['full_std']
        lr=0.0008
    else:
        m=build_model(seed);mu=X.mean(0);std=np.maximum(X.std(0),.1);lr=.003
    w=float(min(15,max(1,((1-Y.mean())/max(Y.mean(),1e-5))**0.5)))
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=.002)
    loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w))
    x=torch.tensor((X-mu)/std,dtype=torch.float32);y=torch.tensor(Y)
    rng=np.random.default_rng(seed*389)
    for k in range(EPOCH):
        ix=rng.choice(len(Y),min(256,len(Y)),replace=True)
        z=loss(m(x[ix]).flatten(),y[ix]);opt.zero_grad();z.backward();opt.step()
    m.eval();return (m,mu,std),float(z.item())

# Need not score all candidate orbits until a model selects them; all arms share pool/query root data.
tests=[]
for seed in range(30260,30280):
    db,roots,pool,_=T.frontier(seed,n_root=140)
    held=[g for g in pool if T.stable_split(g)==0]
    assert len(held)>=22
    f,geo,signed=T.local_features(db,roots,pool,'post_krylov')
    ii={g:j for j,g in enumerate(pool)}
    tests.append((seed,db,held,f[[ii[g] for g in held]]))
print('TEST POOLS',len(tests),time.time()-t0,flush=True)

def assess(model):
    records=[]
    for seed,db,held,feat in tests:
        logits=T.predict(model,feat)
        top=[held[i] for i in np.argsort(-logits,kind='stable')[:22]]
        raw={};edge={}
        for g in top:raw[g],edge[g]=T.score(db,g,'post_krylov')
        gn,selected=T.greedy(raw,edge,top)
        records.append(dict(seed=seed,gain=gn/float(np.dot(T.AMPS,T.AMPS)),positives=sum(v>1e-13 for v in raw.values()),selected=len(selected),top=top))
    return dict(mean_gain=float(np.mean([r['gain'] for r in records])),positive_mean=float(np.mean([r['positives'] for r in records])),n_positive_trials=int(sum(r['gain']>1e-13 for r in records)),trials=records)

result={'note':'Frozen-guide source-to-target adaptation, offline label-budget ablation. Only target labels from sampled training orbits enter training; exact FN amplitudes remain an on-demand backend; counts omit SOURCE pretraining cost. Re-evaluation after FN and RL training not tested.', 'source_pretrain_labels':1980,'post_label_dataset':counts,'epochs_per_arm':EPOCH,'seeds':[19,29],'arms':{}}
result['arms']['source_zero_shot']={'mean_gain':json.load(open('/mnt/data/avsl_transfer20_results.json'))['summary']['post_krylov']['learned_bonds']['fixed_a_gain']}
result['arms']['target_full']={'mean_gain':json.load(open('/mnt/data/avsl_transfer20_train_post_results.json'))['summary']['post_krylov']['learned_bonds']['fixed_a_gain']}
for cap in BUDGETS:
    X=np.concatenate([f[:cap] for f,y in arrays]);Y=np.concatenate([y[:cap] for f,y in arrays]);entry={'labels':len(Y),'positives':int(Y.sum()),'variants':{}}
    for method in ('warm','scratch'):
        reps=[]
        for seed in (19,29):
            model,loss=fit(X,Y,method,seed)
            met=assess(model);met['seed']=seed;met['loss']=loss;reps.append(met)
        entry['variants'][method]={'avg_gain':float(np.mean([t['mean_gain'] for t in reps])),'runs':reps}
        print('RESULT',cap,method,'pos',int(Y.sum()),'gains', [round(x['mean_gain'],9) for x in reps],flush=True)
    result['arms'][str(cap)]=entry
    Path(OUT).write_text(json.dumps(result,indent=2))
print('SAVED',OUT,'wall',time.time()-t0,flush=True)