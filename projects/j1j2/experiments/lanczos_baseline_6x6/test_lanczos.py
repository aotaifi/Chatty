"""Verify lanczos_lib on (1) a random dense symmetric matrix, (2) the Sector class on a synthetic sector, (3) a tiny
J1-J2 cluster (4x3 dense ED, 4x4 sparse ED): alpha vs brute-force scan of the Rayleigh quotient of (1 + alpha H) v,
p-step Ritz energy/variance vs explicit Krylov basis, variance extrapolation vs ED."""
import sys, os
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, scipy.optimize
sys.path.insert(0, os.path.dirname(__file__))
from lanczos_lib import lanczos_ritz, extrapolate

rng = np.random.default_rng(0)


def explicit(Hd, v, p):
    """span{v,..,H^p v} via QR, then eigh of the projected matrix: reference E, var."""
    K = [v]
    for _ in range(p): K.append(Hd @ K[-1])
    Q, _ = np.linalg.qr(np.stack(K, 1))
    ev, ec = np.linalg.eigh(Q.T @ Hd @ Q)
    psi = Q @ ec[:, 0]
    E = psi @ Hd @ psi; Hp = Hd @ psi
    return E, Hp @ Hp - E * E, psi


def check(Hd, v, tag):
    v = v / np.linalg.norm(v)
    res = lanczos_ritz(lambda x: Hd @ x, v, 2)
    for p in (1, 2):
        E, var, psi = explicit(Hd, v, p)
        r = res[p]
        assert abs(r['E'] - E) < 1e-9 * max(1, abs(E)) and abs(r['var'] - var) < 1e-8 * max(1, abs(var)), (tag, p, r['E'], E, r['var'], var)
        assert abs(abs(psi @ r['psi']) - 1) < 1e-8
    # alpha vs brute-force minimisation of the Rayleigh quotient of (1 + alpha H) v
    def rq(a):
        w = v + a * (Hd @ v); return (w @ Hd @ w) / (w @ w)
    a1 = res[1]['alpha']
    emax = max(1.0, np.abs(np.linalg.eigvalsh(Hd)).max())
    grid = np.concatenate([np.linspace(-60, 60, 24001) / emax, a1 * (1 + np.linspace(-0.5, 0.5, 2001))])
    vals = np.array([rq(a) for a in grid])
    assert abs(rq(a1) - res[1]['E']) < 1e-10, (tag, rq(a1), res[1]['E'])
    assert rq(a1) <= vals.min() + 1e-12, (tag, a1, rq(a1), vals.min(), grid[np.argmin(vals)])    # alpha is the global minimiser
    assert rq(a1) <= rq(a1 * (1 + 1e-4)) + 1e-14 and rq(a1) <= rq(a1 * (1 - 1e-4)) + 1e-14
    a_opt = a1
    print(f'{tag}: E0..2 {[round(r["E"], 8) for r in res]} alpha {a1:.6f} (global minimiser of the Rayleigh quotient over a grid + local check) OK')
    return res


# (1) random dense
D = 300
A = rng.normal(size=(D, D)); Hd = 0.5 * (A + A.T)
check(Hd, rng.normal(size=D), 'random dense')

# (2) the Sector class on a synthetic sector (jax)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stall_6x6'))
import st6_sector as SS
import jax.numpy as jnp
n = rng.integers(1, 9, D).astype(float)
Hs = np.zeros((D, D))
for _ in range(6 * D):
    i, j = rng.integers(0, D, 2)
    if i != j: x = rng.random(); Hs[i, j] += x; Hs[j, i] += x
Hs[np.arange(D), np.arange(D)] = rng.normal(0, 2, D)
ui, uc, rows = [], [], []
for i in range(D):
    for j in range(i, D):
        if Hs[i, j] != 0: ui.append(j); uc.append(8 * Hs[i, j] * np.sqrt(n[j] / n[i])); rows.append(i)
inc = np.diff(np.concatenate([[0], np.array(rows)])).astype(np.uint8)
ew, ev = np.linalg.eigh(Hs)
S = SS.Sector.__new__(SS.Sector)
S._setup(n, np.array(ui, np.int32), np.array(uc), inc, np.arange(D, dtype=np.uint64), ev[:, 0] / np.sqrt(n), ew[0], ch=257)
v = rng.normal(size=D); v /= np.linalg.norm(v)
rj = lanczos_ritz(lambda x: S.Hm(x), jnp.asarray(v), 2)
rn = check(Hs, v, 'Sector (numpy ref)')
for p in (0, 1, 2):
    assert abs(rj[p]['E'] - rn[p]['E']) < 1e-10 and abs(rj[p]['var'] - rn[p]['var']) < 1e-9, p
print('Sector GPU-class Lanczos == dense: OK')


# (3) tiny J1-J2 clusters, Sz = 0
def j1j2(Lx, Ly, J2=0.5):
    N = Lx * Ly; s = lambda x, y: (x % Lx) + Lx * (y % Ly)
    bonds = []
    for y in range(Ly):
        for x in range(Lx):
            i = s(x, y)
            for j, J in ((s(x + 1, y), 1), (s(x, y + 1), 1), (s(x + 1, y + 1), J2), (s(x + 1, y - 1), J2)):
                if j != i: bonds.append((i, j, J))
    from itertools import combinations
    states = [sum(1 << i for i in c) for c in combinations(range(N), N // 2)]
    idx = {st: k for k, st in enumerate(states)}
    r, c, val = [], [], []
    for st, k in idx.items():
        d = 0.0
        for i, j, J in bonds:
            bi = (st >> i) & 1; bj = (st >> j) & 1
            if bi == bj: d += 0.25 * J
            else:
                d -= 0.25 * J; r.append(idx[st ^ ((1 << i) | (1 << j))]); c.append(k); val.append(0.5 * J)
        r.append(k); c.append(k); val.append(d)
    return sp.csr_matrix((val, (r, c)), shape=(len(states),) * 2), states


for (Lx, Ly) in ((4, 3), (4, 4)):
    H, states = j1j2(Lx, Ly)
    E0 = sla.eigsh(H, k=1, which='SA')[0][0]
    # Marshall-like start with noise
    N = Lx * Ly
    rngp = np.random.default_rng(3)
    sg = np.array([(-1) ** (sum(((st >> (x + Lx * y)) & 1) for x in range(Lx) for y in range(Ly) if (x + y) % 2 == 0)) for st in states], float)
    v = sg * np.exp(0.3 * rngp.normal(size=len(states)))
    v /= np.linalg.norm(v)
    if len(states) < 2000:
        check(H.toarray(), v, f'J1J2 {Lx}x{Ly} dense')
    res = lanczos_ritz(lambda x: H @ x, v, 2)
    E = [r['E'] for r in res]; var = [r['var'] for r in res]
    e01 = extrapolate(E[:2], var[:2]); e012 = extrapolate(E, var)
    print(f'J1J2 {Lx}x{Ly} (N={N}, D={len(states)}): E0={E0:.8f}  E_p={np.round(E, 6)}  var_p={np.round(var, 4)}  extrap(0,1)={e01:.6f} extrap(0,1,2)={e012:.6f}')
    assert E[0] > E[1] > E[2] >= E0 - 1e-12 and abs(e01 - E0) < abs(E[1] - E0) and abs(e012 - E0) < abs(E[2] - E0) + 1e-3
print('ALL LANCZOS TESTS PASSED')
