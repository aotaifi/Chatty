#!/usr/bin/env python3
"""Supplementary grid (added after the first N=28 results showed that the full Krylov sign s_k*Marshall (targets m2, m3)
is far easier to learn than the step factor c_k): budget sweep for m2/m3 and a large net for m2."""
import sys
out = sys.argv[1] if len(sys.argv) > 1 else '.'
for N in [16, 20, 24, 28, 32, 36]:
    L = []
    for n in (1000, 100000, 300, 3000, 30000):
        for t in ('m2', 'm3'):
            L.append(f'--N {N} --target {t} --beta 0.5 --seed 0 --n {n} --size M')
    L.append(f'--N {N} --target m2 --beta 0.5 --seed 0 --n 30000 --size L')
    open(f'{out}/tasks_supp_N{N}.txt', 'w').write('\n'.join(L) + '\n')
