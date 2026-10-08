"""Guide-evaluation cost of the FEAT write-back per sample: number of distinct configurations one hop (V, W, SI target,
network inputs at x) and two hops (network inputs at the neighbours y entering local edge terms) from x, J1-J2 hops
(NN + NNN spin exchanges), L x L periodic. Samples: Neel state with k random valid exchanges (k ~ typical distance of
sampled configurations from Neel), and random Sz = 0 states.  python wt_hopcount.py
"""
import json, os, sys
import numpy as np


def bonds(L):
    b = []
    for y in range(L):
        for x in range(L):
            i = x + L * y
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                b.append((i, ((x + dx) % L) + L * ((y + dy) % L)))
    return np.array(b)


def nbrs(c, B):
    v = c[B[:, 0]] != c[B[:, 1]]
    out = []
    for (i, j) in B[v]:
        d = c.copy(); d[i], d[j] = d[j], d[i]; out.append(d)
    return out


def count(L, kind, n, rng):
    B = bonds(L); N = L * L
    neel = np.array([(x + y) % 2 for y in range(L) for x in range(L)], np.int8)
    r1, r2 = [], []
    for _ in range(n):
        if kind == 'random':
            c = np.zeros(N, np.int8); c[rng.permutation(N)[:N // 2]] = 1
        else:
            c = neel.copy()
            for _ in range(int(kind)):
                nb = nbrs(c, B); c = nb[rng.integers(len(nb))]
        one = nbrs(c, B)
        two = {d.tobytes() for y in one for d in nbrs(y, B)}
        two |= {y.tobytes() for y in one}; two.discard(c.tobytes())
        r1.append(len(one)); r2.append(len(two))
    return dict(L=L, kind=kind, n=n, one_hop_mean=float(np.mean(r1)), two_hop_mean=float(np.mean(r2)))


if __name__ == '__main__':
    rng = np.random.default_rng(0); out = []
    for L in (6, 8, 10):
        for kind in ('4', '12', 'random'):
            r = count(L, kind, 60 if L < 10 else 20, rng); out.append(r); print(r, flush=True)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'hopcount.json', 'w'), indent=1)
