#!/usr/bin/env python3
"""Supplementary grid 3 (after N36 showed that 4x more optimizer steps lowers the M-net error): fixed numbers of
optimizer steps (2e4, 8e4; default is max(5000, 10 epochs)) for gs, c1, m2 at n=1e4, size M, N = 28, 32, 36."""
import sys
out = sys.argv[1] if len(sys.argv) > 1 else '.'
for N in [28, 32, 36]:
    L = []
    for st in (20000, 80000):
        for t in ('gs', 'c1', 'm2'):
            L.append(f'--N {N} --target {t} --beta 0.5 --seed 0 --n 10000 --size M --min-steps {st} --max-steps {st}')
    open(f'{out}/tasks_supp3_N{N}.txt', 'w').write('\n'.join(L) + '\n')
