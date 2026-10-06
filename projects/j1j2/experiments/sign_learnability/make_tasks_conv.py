#!/usr/bin/env python3
"""'Converged' protocol (added after supp3 showed the default max(5000 steps, 10 epochs) under-trains: N28 M gs
1.1e-4 -> 1.9e-5 -> 1.2e-5 for 5k / 20k / 80k steps): every run gets exactly 8e4 Adam steps (batch 1024).
 A capacity : sizes S, M, L (+ XL at N32/36) for gs and c1, n = 3e4
 B budget   : n = 1e3, 1e5 for gs and c1, size M (n = 1e4 from supp3, n = 3e4 from A)
 C targets  : c2, c3, m2 with M and c2 with L, n = 3e4
Writes tasks_conv_<group>.txt, one file per GPU job."""
import sys
out = sys.argv[1] if len(sys.argv) > 1 else '.'
ST = '--min-steps 80000 --max-steps 80000'
def t(N, tg, n, sz): return f'--N {N} --target {tg} --beta 0.5 --seed 0 --n {n} --size {sz} {ST}'
G = {}
for N in (24, 28, 32, 36):
    rest = [t(N, tg, 30000, sz) for sz in ('M', 'S', 'L') for tg in ('gs', 'c1')]
    rest += [t(N, tg, n, 'M') for n in (1000, 100000) for tg in ('gs', 'c1')]
    rest += [t(N, tg, 30000, 'M') for tg in ('m2', 'c2', 'c3')] + [t(N, 'c2', 30000, 'L')]
    if N == 36:
        G['36xg'] = [t(N, 'gs', 30000, 'XL')]; G['36xc'] = [t(N, 'c1', 30000, 'XL')]
        G['36a'] = rest[:6]; G['36b'] = rest[6:]
    elif N == 32:
        G['32x'] = [t(N, 'gs', 30000, 'XL'), t(N, 'c1', 30000, 'XL')]
        G['32a'] = rest
    else:
        G[f'{N}a'] = rest
for k, v in G.items():
    open(f'{out}/tasks_conv_{k}.txt', 'w').write('\n'.join(v) + '\n')
    print(k, len(v))
