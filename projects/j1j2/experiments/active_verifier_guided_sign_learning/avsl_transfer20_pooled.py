#!/usr/bin/env python3
"""Mixed-chamber transfer control: same code, 1980 total expensive labels, 990 each chamber.
Target is positive one-orbit fixed-amplitude energy gain, NOT Krylov label.
Caveat: assumes known on-demand exact guide amplitudes/sign arrays, uses toy Metropolis frontier.
"""
from pathlib import Path
import re, subprocess, sys
p=Path('/mnt/data/avsl_transfer20.py')
s=p.read_text()
begin=s.index('    for seed in range(args.train_first,args.train_first+args.train_episodes):')
end=s.index('    models={};stats={}',begin)
rep='''    per_chamber_seen={\"marshall\":set(),\"post_krylov\":set()}
    for seed in range(args.train_first,args.train_first+args.train_episodes):
        db,roots,pool,meta=frontier(seed,n_root=args.roots)
        for chamber in (\"marshall\",\"post_krylov\"):
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
'''
s=s[:begin]+rep+s[end:]
s=s.replace("training_unique_labeled_orbits=len(seen)","training_unique_labeled_orbits={k:len(v) for k,v in per_chamber_seen.items()}")
s=s.replace("'Training classifier predicts positive", "'Pooled labels from both chambers; training classifier predicts positive")
Path('/mnt/data/avsl_transfer20_pooled_run.py').write_text(s)
cmd=[sys.executable,'/mnt/data/avsl_transfer20_pooled_run.py','--train-episodes','18','--test-episodes','20','--train-budget','110','--verify-budget','22','--epochs','1000','--seed','19','--out','/mnt/data/avsl_transfer20_pooled_results.json']
print('COMMAND',cmd,flush=True)
subprocess.run(cmd,check=True)