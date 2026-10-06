#!/usr/bin/env python3
"""Writes the run grid (one task file per N) for sl_train.py --tasks.
G1 fixed budget n=1e4 (targets x beta, + seed 1); G2 budget sweep (beta 0.5); G4 ablations (aux, no-nbr,
full-sign targets m2/m3, 4x longer training); G3 capacity sweep (n=3e4). Size M unless noted."""
import sys
NS = [16, 20, 24, 28, 32, 36]
T4 = ['c1', 'c2', 'c3', 'gs']
out = sys.argv[1] if len(sys.argv) > 1 else '.'
for N in NS:
    L = []
    f = lambda **k: L.append(f'--N {N} ' + ' '.join(f'--{a.replace("_", "-")} {v}' for a, v in k.items()))
    for t in T4:                                   # G1
        for b in (0.5, 1.0, 0.3):
            f(target=t, beta=b, seed=0, n=10000, size='M')
        f(target=t, beta=0.5, seed=1, n=10000, size='M')
    for n in (1000, 100000, 300, 3000, 30000, 100):  # G2
        for t in T4:
            f(target=t, beta=0.5, seed=0, n=n, size='M')
    for t in ('c1', 'c2', 'c3'):                   # G4
        f(target=t, beta=0.5, seed=0, n=10000, size='M', lam=0.1)
    for t in ('c1', 'gs'):
        for n in (10000, 100000):
            f(target=t, beta=0.5, seed=0, n=n, size='M', nbr=0)
    for t in ('m2', 'm3'):
        f(target=t, beta=0.5, seed=0, n=10000, size='M')
    for t in ('c1', 'c2', 'gs'):
        f(target=t, beta=0.5, seed=0, n=10000, size='M', epochs=40, max_steps=160000)
    for s in ('XS', 'S', 'L'):                     # G3
        for t in ('c1', 'gs', 'c2'):
            f(target=t, beta=0.5, seed=0, n=30000, size=s)
    open(f'{out}/tasks_N{N}.txt', 'w').write('\n'.join(L) + '\n')
    print(N, len(L))
