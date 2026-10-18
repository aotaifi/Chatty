#!/usr/bin/env python3
"""Regenerate AVSL Marshall/post-Krylov train-chamber comparisons.

The original benchmark is unchanged; we patch *only* its two training-label
callsites. Holdout pools, 22-orbit score budget, features, and evaluation are
identical. Requires prepared 20-site data and dependencies of avsl_transfer20.py.
"""
from pathlib import Path
import argparse, subprocess, sys
ap=argparse.ArgumentParser()
ap.add_argument('--source',type=Path,default=Path(__file__).with_name('avsl_transfer20.py'))
ap.add_argument('--scratch',type=Path,default=Path('/mnt/data'))
ap.add_argument('--seed',type=int,default=19)
ap.add_argument('--train-chamber',choices=('marshall','post_krylov'),default='post_krylov')
ap.add_argument('--train-budget',type=int,default=110)
args=ap.parse_args()
code=args.source.read_text()
if args.train_chamber=='post_krylov':
    old1="f,geo,signed=local_features(db,roots,pool,'marshall')"
    new1="f,geo,signed=local_features(db,roots,pool,'post_krylov')"
    old2="g=pool[i];raw,_=score(db,g,'marshall');labs.append(float(raw>1e-13));seen.add(g)"
    new2="g=pool[i];raw,_=score(db,g,'post_krylov');labs.append(float(raw>1e-13));seen.add(g)"
    assert code.count(old1)==code.count(old2)==1
    code=code.replace(old1,new1).replace(old2,new2)
args.scratch.mkdir(parents=True,exist_ok=True)
script=args.scratch/f'avsl_transfer20_train_{args.train_chamber}.py'
script.write_text(code)
out=args.scratch/f'avsl_transfer20_train_{args.train_chamber}_b{args.train_budget}_seed{args.seed}.json'
subprocess.run([sys.executable,str(script),'--train-episodes','18','--test-episodes','20',
    '--train-budget',str(args.train_budget),'--verify-budget','22','--epochs','1000',
    '--seed',str(args.seed),'--out',str(out)],check=True)
print('RESULT',out)
