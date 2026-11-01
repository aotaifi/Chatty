#!/usr/bin/env python3
"""Regenerate the fixed 20-site reference guide used by AVSL-06/07/08.

This runs the ALREADY ARCHIVED exact 20-site J1-J2 preparation (J2=0
ground-state amplitude, 6 Marshall FN relaxations, K1 sign optimization,
3 post-K1 FN relaxations). J2=.5 ED vector psi0 is retained ONLY for
independent diagnostics; it must never train proposal policies.

Warning: scipy eigensolves on 184,756 basis states can require minutes/RAM.
The resulting a is approximate, NOT guaranteed an exact FN eigenvector.
"""
import argparse
import numpy as np
from pathlib import Path

def run(output):
    root=Path(__file__).resolve().parents[2]
    source=root/'results/scalable_sign_2026-10-18/fn_collapse_exact20_krylov_v2.py'
    src=source.read_text()
    # Execute only benchmark definitions/preparation, not its multicase main.
    cutoff=src.index('import sys\nNITER=')
    ns={'__file__':str(source),'__name__':'avsl08_reference_preparation'}
    exec(compile(src[:cutoff],str(source),'exec'),ns)
    a=ns['a0'].copy()
    s=ns['sM'].copy()
    for k in range(6):
        _,a,_=ns['fn_step'](a,s,f'AVSL08-pre{k}',tol=1e-8)
    s,_=ns['krylov_step'](a,s,'AVSL08-K1')
    for k in range(3):
        _,a,_=ns['fn_step'](a,s,f'AVSL08-post{k}',tol=1e-8)
    g=s*(ns['Hoff']@(s*a))
    flip=g>0
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,
         states=ns['states'],amp=a,guide=s.astype('int8'),
         flip=flip.astype('int8'),psi0=ns['psi0'],
         diag=ns['diag'],maskA=np.array([ns['maskA']]))
    print('SAVED',output,'n_positive',int(flip.sum()))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    run(p.parse_args().out)
