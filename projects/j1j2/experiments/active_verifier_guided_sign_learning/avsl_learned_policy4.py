#!/usr/bin/env python3
"""AVSL contextual-bandit candidate ranking, 4x4 square-lattice J1-J2.

Trains a small MLP on energy outcomes of queried orbit actions and tests on
UNSEEN symmetry orbits (hash split), as well as held-out walker seeds.

CRUCIAL: inputs = already observed signed edges plus *local* spin/guide
features. Labels = extra on-demand one-hop orbit energy query. No ground-state
ED/FN amplitudes, no full-space energy table in decisions. Includes no-flip
baseline y=0 and retains all negative examples. Counts expensive queries.

This is contextual-bandit supervised reward modelling, NOT policy-gradient
GRPO. Compare with cheap signed-edge heuristic and random at equal B.
"""
from __future__ import annotations
import argparse,json,sys,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent))
import avsl_probe4 as p

def split(gid):
    # Stable, reproducible split by full symmetry-orbit key. Never use state ID as feature.
    return ((int(gid)*2654435761+1013904223) % 997) % 5

def make_features(g,pool,scores):
    feats=[]
    for k in pool:
        a,s=g.evaluate(k)
        vals=np.asarray(scores.get(k,[]),dtype=float)
        pos=vals[vals>0];neg=-vals[vals<0]
        # Local spin bond geometry; no neighbor guide calls here.
        nn_flips=0;nnn_flips=0
        for i,j,J,h,mask in p.BONDS:
            if (k>>i&1)!=(k>>j&1):
                if J==1:nn_flips+=1
                else:nnn_flips+=1
        total=pos.sum()+neg.sum()+1e-12
        feats.append([
            np.log1p(len(vals)),
            np.log1p(pos.sum()),np.log1p(neg.sum()),
            (pos.sum()-neg.sum())/total,
            np.log1p(pos.max()) if len(pos) else 0.,
            np.log1p(neg.max()) if len(neg) else 0.,
            len(pos)/(len(vals)+1e-12),
            np.log1p(sum(vals*vals)**.5),
            np.log1p(len(p.orbit(k))),
            float(nn_flips)/32.,float(nnn_flips)/32.,
            float(np.log(a))/30.,
            float(g.kappa),
        ])
    return np.asarray(feats,dtype=np.float32)

class Policy(nn.Module):
    def __init__(self,d):
        super().__init__();self.f=nn.Sequential(nn.Linear(d,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,x):return self.f(x).squeeze(-1)

def train_model(train_data,seed,steps=650):
    torch.manual_seed(seed)
    X=np.concatenate([r['X'] for r in train_data]).astype('float32')
    Y=np.concatenate([r['Y'] for r in train_data]).astype('float32')
    # Rank learned from reward (positive vs keep 0) using ALL outcomes incl no-flip.
    # Positive weighting corrects rare-event imbalance without discarding negatives.
    mu=X.mean(0);sigma=X.std(0).clip(0.1,None)
    XX=torch.from_numpy((X-mu)/sigma);YY=torch.from_numpy(Y)
    net=Policy(X.shape[1]);optimizer=torch.optim.AdamW(net.parameters(),lr=0.003,weight_decay=0.002)
    rng=np.random.default_rng(seed+1819)
    positive=float(Y.mean());weight_pos=min(12., max(1.,(1.-positive)/max(positive,1e-4))**.5)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight_pos))
    net.train();t0=time.time()
    for step in range(steps):
        idx=rng.choice(len(X),min(256,len(X)),replace=True)
        loss=criterion(net(XX[idx]),YY[idx]);optimizer.zero_grad();loss.backward();optimizer.step()
    net.eval()
    return net,mu,sigma,{'train_count':int(len(Y)),'positive_count':int(Y.sum()),'positive_fraction':float(positive),'positive_weight':float(weight_pos),'last_loss':float(loss.detach()),'train_sec':time.time()-t0}

def get_labeled_run(seed,kappa,n_seeds,train_cap=80):
    g,pool,scores,meta=p.generate(seed=seed,kappa=kappa,n_seeds=n_seeds)
    allowed=[k for k in pool if split(k)!=0]
    rng=np.random.default_rng(seed*19+int(round(kappa*1000)))
    # Logged, randomized acquisition so policy can learn from the full proposal distribution;
    # different from dynamic success-only filtering (selection bias).
    if len(allowed)>train_cap:allowed=list(rng.choice(allowed,size=train_cap,replace=False))
    x=make_features(g,allowed,scores)
    count0=len(g.asked)
    gains=[]
    for k in allowed:
        val,_=p.score_orbit(g,int(k));gains.append(float(val))
    return {'X':x,'Y':(np.asarray(gains)>1e-10).astype('float32'),
            'total_queries':len(g.asked),'extra_queries':len(g.asked)-count0,
            'common_queries':count0,'seed':seed,'kappa':kappa}

def evaluate_model(seed,kappa,net,mu,sigma,B=22,n_seeds=160,mode='orbit_holdout'):
    g,pool,scores,meta=p.generate(seed=seed,kappa=kappa,n_seeds=n_seeds)
    allowed=[k for k in pool if mode=='new_seeds' or split(k)==0]
    feats=make_features(g,allowed,scores)
    if len(allowed)<B:return None
    with torch.no_grad():pred=net(torch.from_numpy((feats-mu)/sigma)).numpy()
    rng=np.random.default_rng(seed+6611)
    indexed=dict(zip(allowed,range(len(allowed))))
    rs={m:None for m in ['learned','signed','random','max_violation']}
    methods={
        'learned':sorted(allowed,key=lambda k:(-pred[indexed[k]],k)),
        'signed':sorted(allowed,key=lambda k:(-np.mean(scores[k]) if scores[k] else float('inf'),k)),
        'max_violation':sorted(allowed,key=lambda k:(-max(scores[k]) if scores[k] else float('inf'),k)),
        'random':list(rng.permutation(allowed))
    }
    # Common metadata has already paid every frontier query. Each arm begins from identical cached calls.
    for m,order in methods.items():
        guide=p.Oracle(kappa);guide.asked=dict(g.asked)
        raw={};cross={}
        for k in order[:B]:raw[int(k)],cross[int(k)]=p.score_orbit(guide,int(k))
        # Choose greedily among queried complete symmetry orbits, measuring nonlinear interactions.
        picks=[];gain=0.;max_flips=12
        for it in range(max_flips):
            cand={k:raw[k]-8*sum(w for o,w in cross[k].items() if o in picks) for k in raw if k not in picks}
            if not cand:break
            opt=max(cand,key=cand.get);delta=cand[opt]
            if delta<=1e-10:break
            picks.append(opt);gain+=float(delta)
        # Bond identity check using complete orbit crossing contributions.
        check=4*sum(w for k in picks for h,w in cross[k].items() if h not in picks)
        assert abs(gain-check)<1e-7,(gain,check)
        rs[m]={'positive_tested':int(sum(v>1e-10 for v in raw.values())),
               'negative_tested':int(sum(v<=1e-10 for v in raw.values())),
               'gain_numerator':float(gain),'selected_orbits':len(picks),
               'total_guide_queries':len(guide.asked),'extra_verifier_queries':len(guide.asked)-len(g.asked),
               'query_per_gain':float((len(guide.asked))/max(gain,1e-8)),
               'selected_orbit_keys':[int(v) for v in picks]}
    return {'seed':int(seed),'kappa':kappa,'mode':mode,'candidate_pool':len(allowed),
            'common_queries':len(g.asked),'n_seeds':n_seeds,'methods':rs}

def run(args):
    torch.set_num_threads(2)
    train=[]
    kappas=[.75,.85,.9,1.0,1.1]
    for k in kappas:
        for seed in range(args.train_seed_start,args.train_seed_start+args.train_runs):
            train.append(get_labeled_run(seed,k,args.n_seeds,args.train_cap))
    net,mu,sigma,stat=train_model(train,args.seed,args.steps)
    results=[]
    for seed in range(args.test_seed_start,args.test_seed_start+args.test_runs):
        for k in [.8,.9,1.0]:
            row=evaluate_model(seed,k,net,mu,sigma,B=args.B,n_seeds=args.n_seeds,mode=args.holdout_mode)
            if row is not None:results.append(row)
    methods=['learned','signed','random','max_violation']
    summary={}
    for m in methods:
        gains=np.asarray([r['methods'][m]['gain_numerator'] for r in results]);pos=np.asarray([r['methods'][m]['positive_tested'] for r in results]);queries=np.asarray([r['methods'][m]['total_guide_queries'] for r in results])
        summary[m]={'mean_gain_numerator':float(np.mean(gains)),'median_gain_numerator':float(np.median(gains)),
         'mean_positive_tested':float(pos.mean()),'mean_total_queries':float(queries.mean()),
         'mean_gain_per_1000_queries':float(np.mean(gains/queries)*1e3),
         'wins_vs_random':int(sum(r['methods'][m]['gain_numerator']>r['methods']['random']['gain_numerator']+1e-10 for r in results)),
         'wins_vs_signed':int(sum(r['methods'][m]['gain_numerator']>r['methods']['signed']['gain_numerator']+1e-10 for r in results))}
    result={'setup':vars(args),'train':stat,'total_train_guide_queries':int(sum(r['total_queries'] for r in train)),
        'total_train_verifier_queries':int(sum(r['extra_queries'] for r in train)),
        'n_eval':len(results),'methods':summary,'per_eval':results,
        'limitations':['Frozen callable analytic-guide energy numerator, not projected FN or exact sign accuracy',
         '4x4 finite-size; active sampling support not tested at large N','Network trained on exact known-guide energy rewards, not unreliable FN neural guide',
         'Reward labels require charged neighbor guide queries; training amortization is not free',
         'Split by translation+global spin-flip orbit; deterministic hash partition with approximately 20% test; no orbit ID feature',
         'Fully expanded common seed frontier is expensive and can dominate verification query budget',
         'Data are logged randomly; this is a contextual-bandit reward model, not a GRPO policy-gradient optimizer.']}
    Path(args.out).write_text(json.dumps(result,indent=2));torch.save({'state':net.state_dict(),'mean':mu,'std':sigma,'stats':stat},args.out.replace('.json','.pt'))
    print('TRAIN',stat,flush=True)
    print('SUMMARY',json.dumps(summary,indent=2),flush=True)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-seed-start',type=int,default=300)
    ap.add_argument('--train-runs',type=int,default=9)
    ap.add_argument('--test-seed-start',type=int,default=2100)
    ap.add_argument('--test-runs',type=int,default=12)
    ap.add_argument('--n-seeds',type=int,default=160)
    ap.add_argument('--train-cap',type=int,default=80)
    ap.add_argument('--B',type=int,default=22)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--steps',type=int,default=650)
    ap.add_argument('--holdout-mode',choices=['orbit_holdout','new_seeds'],default='orbit_holdout')
    ap.add_argument('--out',default='/mnt/data/avsl_learned_policy4_results.json')
    a=ap.parse_args();run(a)
