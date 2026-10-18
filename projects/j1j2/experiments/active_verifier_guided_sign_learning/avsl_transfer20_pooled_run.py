#!/usr/bin/env python3
"""AVSL clean chamber-transfer test, J1-J2 20-site tilted torus.

Learn which symmetry orbits have positive fixed-a marginal sign-flip gain under
Marshall signs; test zero-shot on a different post-Krylov chamber, and on
held-out symmetry orbits. Both chambers share the same FIXED amplitude vector,
walker seed/candidate pools, and local-query budgets. ED never trains labels.

On-demand full 20-site amplitude/sign arrays are ORACLE backends. This tests
transfer across sign chambers, NOT a scalable GFMC/guide learned solution.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import argparse, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn

D0=np.load('/mnt/data/sign20_training_data.npz')
STATES=D0['states'].astype(np.uint32); AMPS=D0['amp'].astype(float); SK=D0['guide'].astype(np.int8)
PSI=D0['psi0'].astype(float); N=20; Lx,Ly=4,5; DIM=len(STATES); MASK=(1<<N)-1
LOOK=np.full(1<<N,-1,np.int32); LOOK[STATES]=np.arange(DIM,dtype=np.int32)
msk=int(D0['maskA'][0]); b=np.zeros(DIM,dtype=np.int8)
for i in range(N):
    if msk>>i&1:b^=((STATES>>i)&1).astype(np.int8)
SM=np.where(b==0,1,-1).astype(np.int8)

def site(x,y):
    q,r=divmod(y,Ly); return (x-q)%Lx+Lx*r
BONDS=[]
for y in range(Ly):
    for x in range(Lx):
        i=site(x,y)
        for dx,dy,h in ((1,0,.5),(0,1,.5),(1,1,.25),(-1,1,.25)):
            j=site(x+dx,y+dy)
            BONDS.append(((1<<i)|(1<<j),h,i,j))
xy={site(x,y):(x,y) for y in range(Ly) for x in range(Lx)}
PERMS=[]
for dy in range(Ly):
    for dx in range(Lx):
        PERMS.append(tuple(site(xy[i][0]+dx,xy[i][1]+dy) for i in range(N)))

class Backend:
    def __init__(self):
        self.amps={};self.signs={};self.key_cache={};self.members={};self.amp_calls=0;self.sign_calls=0
    def a(self,x):
        x=int(x);self.amp_calls+=1
        if x not in self.amps:self.amps[x]=float(AMPS[LOOK[x]])
        return self.amps[x]
    def s(self,x,chamber):
        x=int(x);self.sign_calls+=1
        k=(x,chamber)
        if k not in self.signs:self.signs[k]=int((SM if chamber=='marshall' else SK)[LOOK[x]])
        return self.signs[k]
    def key(self,x):
        x=int(x)
        if x in self.key_cache:return self.key_cache[x]
        z=[]
        for p in PERMS:
            v=0
            for i,j in enumerate(p):v|=((x>>i)&1)<<j
            z.extend((v,v^MASK))
        vals=tuple(sorted(set(z)));k=vals[0]
        for q in vals:self.key_cache[q]=k
        self.members[k]=vals
        return k
    def neigh(self,x):
        for bit,h,i,j in BONDS:
            y=x^bit
            if LOOK[y]>=0:yield y,h
    def counts(self):return dict(unique_amp=len(self.amps),unique_sign=len(self.signs),unique_orbit=len(self.members),amp_requests=self.amp_calls,sign_requests=self.sign_calls)

def stable_split(g):return ((int(g)*2654435761+1013904223)%997)%5

def frontier(seed,n_root=140,burn=2200,thin=13):
    rng=np.random.default_rng(seed);db=Backend();x=int(STATES[rng.integers(DIM)]); roots=[];nacc=0
    for it in range(burn+n_root*thin):
        bit,h,i,j=BONDS[int(rng.integers(len(BONDS)))];y=x^bit
        if LOOK[y]>=0:
            ax=db.a(x);ay=db.a(y)
            if rng.random()<min(1,ay/max(ax,1e-300)):x=y;nacc+=1
        if it>=burn and (it-burn)%thin==0:roots.append(x)
    roots=sorted(set(roots));pool=set(db.key(x) for x in roots)
    for x in roots:
        for y,h in db.neigh(x):pool.add(db.key(y))
    # Query all pool representative amplitudes; common to policies.
    for g in pool:db.a(g)
    return db,roots,sorted(pool),dict(seed=seed,root_count=len(roots),pool_count=len(pool),acceptance=nacc/(burn+n_root*thin))

def local_features(db,roots,pool,chamber):
    # Aggregate only bonds emanating from actually observed walker roots; NOT full score of candidate orbits.
    v={g:[] for g in pool}
    for x in roots:
        ax=db.a(x);sx=db.s(x,chamber)
        for y,h in db.neigh(x):
            g=db.key(y)
            if g in v:v[g].append(h*sx*db.s(y,chamber)*ax*db.a(y))
    feat=[];geo=[];signed={}
    for g in pool:
        k=int(g);a=db.a(k);s=db.s(k,chamber)
        vals=np.asarray(v[g],dtype=float)
        # Signed contribution features are only from encountered walker-root edges.
        pos=vals[vals>0];neg=-vals[vals<0]
        scale=float(abs(vals).sum()+1e-25)
        nn_=nnn_=0
        for _,h,i,j in BONDS:
            if (k>>i&1)!=(k>>j&1):
                if h==.5:nn_+=1
                else:nnn_+=1
        common=[np.log1p(len(db.members[k])),float(nn_)/40.,float(nnn_)/40.,np.log(max(a,1e-240))/15.,float(s)]
        bonds=[np.log1p(len(vals)),np.log1p(pos.sum()),np.log1p(neg.sum()),(pos.sum()-neg.sum())/scale,
               (pos.sum()/scale),np.log1p(pos.max()) if len(pos) else 0.,
               np.log1p(neg.max()) if len(neg) else 0.,float(np.mean(vals)) if len(vals) else 0.]
        feat.append(common+bonds);geo.append(common)
        signed[k]=float(np.mean(vals)) if len(vals) else 0.
    return np.asarray(feat,np.float32),np.asarray(geo,np.float32),signed

def score(db,g,chamber):
    # Representative orbit-specific crossing contribution; each state in orbit has equivalent bonds.
    ax=db.a(g);sx=db.s(g,chamber);mult=len(db.members[g]); w=defaultdict(float)
    for y,h in db.neigh(g):
        ky=db.key(y)
        if ky==g:continue
        w[ky]+=mult*h*sx*db.s(y,chamber)*ax*db.a(y)
    return float(4*sum(w.values())),dict(w)

def train(records,dim,seed=10,epochs=700):
    X=np.concatenate([r[0] for r in records]);Y=np.concatenate([r[1] for r in records]);
    mu=X.mean(0);std=np.maximum(X.std(0),.1)
    torch.manual_seed(seed);torch.set_num_threads(2)
    XX=torch.tensor((X-mu)/std);YY=torch.tensor(Y)
    net=nn.Sequential(nn.Linear(dim,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))
    p=Y.mean();wt=float(min(15,max(1,((1-p)/max(p,1e-5))**.5)))
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(wt));opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.002)
    rng=np.random.default_rng(seed*389)
    for i in range(epochs):
        ix=rng.choice(len(X),min(256,len(X)),replace=True)
        l=loss_fn(net(XX[ix]).flatten(),YY[ix]);opt.zero_grad();l.backward();opt.step()
    net.eval()
    return (net,mu,std),dict(training_labels=len(Y),positives=int(Y.sum()),positive_fraction=float(p),final_loss=float(l.item()),positive_weight=wt)

def predict(model,X):
    net,mu,std=model
    with torch.no_grad():return net(torch.tensor((X-mu)/std)).flatten().numpy()

def greedy(raw,edges,subset,max_flip=12):
    chosen=[];keep=set();gross=0.
    for _ in range(min(len(subset),max_flip)):
        scores={g:raw[g]-8*sum(z for j,z in edges[g].items() if j in keep) for g in subset if g not in keep}
        if not scores:break
        g=max(scores,key=scores.get)
        if scores[g]<=1e-13:break
        gross+=scores[g];chosen.append(g);keep.add(g)
    cross=4*sum(z for g in chosen for h,z in edges[g].items() if h not in keep)
    assert np.isclose(gross,cross,atol=1e-10), (gross,cross)
    return float(gross),chosen

def evaluate_episode(seed,models,B=22,n_root=140):
    db,roots,pool,meta=frontier(seed,n_root=n_root)
    # fixed, disjoint orbit holdout from both chambers
    test=[g for g in pool if stable_split(g)==0]
    if len(test)<B:return None
    rng=np.random.default_rng(seed+313)
    common_pre=db.counts()
    data={};results={}
    for chamber in ('marshall','post_krylov'):
        features,geom,signed=local_features(db,roots,pool,chamber)
        indices={g:i for i,g in enumerate(pool)}
        p_full=predict(models['full'],features);p_geo=predict(models['geo'],geom)
        order={
          'learned_bonds':sorted(test,key=lambda g:(-p_full[indices[g]],g)),
          'local_geometry_amp':sorted(test,key=lambda g:(-p_geo[indices[g]],g)),
          'observed_signed_bonds':sorted(test,key=lambda g:(-signed[g],g)),
          'random':list(rng.permutation(test))
        }
        # Apply every method at identical number of orbit score evaluations; cache shared verifier scores
        arms={}
        for method,ordered in order.items():
            base_n=len(db.amps);raw={};edges={}
            verifier=Backend()
            verifier.amps=db.amps.copy();verifier.signs=db.signs.copy()
            verifier.key_cache=db.key_cache.copy();verifier.members=db.members.copy()
            for g in ordered[:B]:raw[g],edges[g]=score(verifier,int(g),chamber)
            gain,selected=greedy(raw,edges,ordered[:B]);
            arms[method]=dict(beneficial_candidates=sum(v>1e-13 for v in raw.values()),
                   fixed_a_gain=gain,verified_orbits=B,accepted_orbits=len(selected),
                   selected_keys=[int(x) for x in selected],
                   marginal_extra_amp_queries=len(verifier.amps)-base_n,total_amp_queries=len(verifier.amps))
        results[chamber]=arms
    full_cost=db.counts()
    return dict(seed=seed,common=meta,frontier_cost=common_pre,cost_after=full_cost,test_orbits=len(test),arms=results)

def run(args):
    st=time.time();train_data_full=[];train_data_geo=[];seen=set();accum=[]
    per_chamber_seen={"marshall":set(),"post_krylov":set()}
    for seed in range(args.train_first,args.train_first+args.train_episodes):
        db,roots,pool,meta=frontier(seed,n_root=args.roots)
        for chamber in ("marshall","post_krylov"):
            f,geo,signed=local_features(db,roots,pool,chamber)
            idx=[i for i,g in enumerate(pool) if stable_split(g)!=0 and g not in per_chamber_seen[chamber]]
            rr=np.random.default_rng(seed*47).permutation(idx)[:args.train_budget//2]
            labs=[]
            for i in rr:
                g=pool[i];raw,_=score(db,g,chamber);labs.append(float(raw>1e-13));per_chamber_seen[chamber].add(g)
            if len(rr):
                train_data_full.append((f[rr],np.asarray(labs,np.float32)))
                train_data_geo.append((geo[rr],np.asarray(labs,np.float32)))
            accum.append(dict(**meta,chamber=chamber,queried_labels=len(rr),positive_labels=int(sum(labs)),total_unique_amp=len(db.amps)))
    models={};stats={}
    for typ,dat in [('full',train_data_full),('geo',train_data_geo)]:
        models[typ],stats[typ]=train(dat,13 if typ=='full' else 5,seed=args.seed,epochs=args.epochs)
    tests=[]
    for seed in range(args.test_first,args.test_first+args.test_episodes):
        e=evaluate_episode(seed,models,B=args.verify_budget,n_root=args.roots)
        if e:tests.append(e)
        print('TEST',seed, 'marshall', {m:round(v['fixed_a_gain'],7) for m,v in e['arms']['marshall'].items()} if e else None,
              'post_krylov', {m:round(v['fixed_a_gain'],7) for m,v in e['arms']['post_krylov'].items()} if e else None,flush=True)
    summary={}
    for chamber in ('marshall','post_krylov'):
        summary[chamber]={}
        for method in ('learned_bonds','local_geometry_amp','observed_signed_bonds','random'):
            dd=[t['arms'][chamber][method] for t in tests]
            if not dd:continue
            summary[chamber][method]={k:float(np.mean([v[k] for v in dd])) for k in ('beneficial_candidates','fixed_a_gain','accepted_orbits')}
            summary[chamber][method]['positive_gain_trials']=int(sum(v['fixed_a_gain']>1e-12 for v in dd))
    result=dict(description=__doc__,args=vars(args),training=stats,train_cost=accum,
        training_unique_labeled_orbits={k:len(v) for k,v in per_chamber_seen.items()},test_count=len(tests),summary=summary,test_episodes=tests,
        wall_s=time.time()-st,limitations=[
        'Identical frozen exact FN amplitudes in two chambers, not reoptimized FN transfer.',
        'Global precomputed amplitude and sign arrays used solely as on-demand oracle backends.',
        'Metropolis root proposer samples a^1, not the mixed FN distribution; not actual GFMC.',
        'The heldout symmetry orbit split excludes label leakage by orbit, but orbit neighbors and their sign values remain queryable.',
        'Common candidate pool uses walker roots and one-hop expansions; no learned multi-step exploration policy yet.',
        'Budget is equal expensive orbit-score count, with independently cloned on-demand query caches per method; training label and common frontier costs are reported separately.',
        'Pooled labels from both chambers; training classifier predicts positive *single-orbit frozen-a gain*, not exact ground-state signs or optimized Krylov label.',
        'The 20-site symmetry class is small; source and target differ in signs but not configurations or amplitudes.' ])
    Path(args.out).write_text(json.dumps(result,indent=2))
    torch.save({'full':models['full'][0].state_dict(),'geo':models['geo'][0].state_dict(),
                'full_mu':models['full'][1],'full_std':models['full'][2],
                'geo_mu':models['geo'][1],'geo_std':models['geo'][2]},Path(args.out).with_suffix('.pt'))
    print('SUMMARY',json.dumps(summary,indent=2),'TRAIN',stats,'TIME',round(time.time()-st,1),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train-first',type=int,default=20260);ap.add_argument('--train-episodes',type=int,default=14)
    ap.add_argument('--test-first',type=int,default=30260);ap.add_argument('--test-episodes',type=int,default=12)
    ap.add_argument('--train-budget',type=int,default=110);ap.add_argument('--verify-budget',type=int,default=22)
    ap.add_argument('--roots',type=int,default=140);ap.add_argument('--epochs',type=int,default=900);ap.add_argument('--seed',type=int,default=19)
    ap.add_argument('--out',type=str,default='/mnt/data/avsl_transfer20_results.json')
    args=ap.parse_args();run(args)