#!/usr/bin/env python3
"""Supplementary grid 2 (after N32/N36 showed a steep capacity dependence): XL nets (128 ch x 6, ~0.74M params),
4x longer training of L nets, and L nets at a larger sample budget, for the exact sign (gs) and the first step label (c1)."""
import sys
out = sys.argv[1] if len(sys.argv) > 1 else '.'
for N in [24, 28, 32, 36]:
    L = []
    for t in ('gs', 'c1'):
        L.append(f'--N {N} --target {t} --beta 0.5 --seed 0 --n 30000 --size XL')
    if N >= 32:
        for t in ('gs', 'c1'):
            L.append(f'--N {N} --target {t} --beta 0.5 --seed 0 --n 100000 --size L')
            L.append(f'--N {N} --target {t} --beta 0.5 --seed 0 --n 30000 --size L --epochs 40 --max-steps 160000')
    open(f'{out}/tasks_supp2_N{N}.txt', 'w').write('\n'.join(L) + '\n')
