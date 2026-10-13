"""Locality vs J2: exact ground state of the 4x6 periodic J1-J2 model (Sz = 0) and the double-swap interaction
I(b,b') = la(x^{bb'}) + la(x) - la(x^b) - la(x^{b'}) and sign non-additivity vs bond distance (README). CPU only.
  python ed_pairint.py OUT.json J2 [J2 ...]
"""
import sys, json, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from itertools import combinations
Lx, Ly = 6, 4; N = Lx * Ly; ONE = np.uint64(1)
q = lambda x, y: (x % Lx) + Lx * (y % Ly)
NN, NNN = [], []
for y in range(Ly):
    for x in range(Lx):
        i = q(x, y); NN += [(i, q(x + 1, y)), (i, q(x, y + 1))]; NNN += [(i, q(x + 1, y + 1)), (i, q(x + 1, y - 1))]
B = NN + NNN; NB = len(B)
BI = np.array([b[0] for b in B], np.uint64); BJ = np.array([b[1] for b in B], np.uint64)
MASKS = (ONE << BI) | (ONE << BJ); ISNN = np.array([1] * len(NN) + [0] * len(NNN), bool)


def sdist(p, r):
    dx = abs(p % Lx - r % Lx); dy = abs(p // Lx - r // Lx); dx = min(dx, Lx - dx); dy = min(dy, Ly - dy)
    return np.sqrt(dx * dx + dy * dy)
DB = np.round(np.array([[min(sdist(p, r) for p in B[a] for r in B[c]) for c in range(NB)] for a in range(NB)]), 3)
keys = sorted(set(DB[DB > 0].tolist()))
# basis: all 24-bit words with 12 ones
t0 = time.time()
basis = np.array(sorted(sum(1 << np.array(c, dtype=np.uint64)) for c in combinations(range(N), N // 2)), np.uint64) \
    if False else None
idx = np.arange(1 << N, dtype=np.uint64)
pc = np.zeros(1 << N, np.uint8)
for k in range(N): pc += ((idx >> np.uint64(k)) & ONE).astype(np.uint8)
basis = idx[pc == N // 2]; del idx, pc
D = len(basis); print('basis', D, f'{time.time() - t0:.0f}s', flush=True)
valid = np.empty((NB, D), bool)
for b in range(NB): valid[b] = (((basis >> BI[b]) ^ (basis >> BJ[b])) & ONE).astype(bool)
cols_b = []
for b in range(NB):
    r = np.nonzero(valid[b])[0]
    cols_b.append((r.astype(np.int32), np.searchsorted(basis, basis[r] ^ MASKS[b]).astype(np.int32)))
print('neighbours built', f'{time.time() - t0:.0f}s', flush=True)
res = dict(cluster='4x6', D=D, J2={})
rng = np.random.default_rng(3)
for J2 in [float(a) for a in sys.argv[2:]]:
    JB = np.where(ISNN, 1.0, J2)
    diag = np.zeros(D)
    for b in range(NB): diag += np.where(valid[b], -0.25, 0.25) * JB[b]
    rows = np.concatenate([c[0] for c in cols_b] + [np.arange(D, dtype=np.int32)])
    cols = np.concatenate([c[1] for c in cols_b] + [np.arange(D, dtype=np.int32)])
    data = np.concatenate([np.full(len(c[0]), 0.5 * JB[b], np.float64) for b, c in enumerate(cols_b)] + [diag])
    H = sp.csr_matrix((data, (rows, cols)), shape=(D, D)); del rows, cols, data
    t1 = time.time(); ev, vec = eigsh(H, k=1, which='SA', tol=1e-10); del H
    psi = vec[:, 0]; psi = psi * np.sign(psi[np.argmax(np.abs(psi))])
    la = np.log(np.maximum(np.abs(psi), 1e-300)); s = np.sign(psi)
    p = psi ** 2; p /= p.sum()
    samp = rng.choice(D, size=2000, p=p)
    acc = {k: np.zeros(4) for k in keys}
    nbr = np.stack([np.searchsorted(basis, basis[samp] ^ MASKS[b]) for b in range(NB)], 1)    # (S, NB)
    for si, ix in enumerate(samp):
        x = basis[ix]; v = valid[:, ix]
        a_, c_ = np.triu_indices(NB, 1); m = v[a_] & v[c_] & (DB[a_, c_] > 0); a_, c_ = a_[m], c_[m]
        ibb = np.searchsorted(basis, x ^ MASKS[a_] ^ MASKS[c_])
        ia, ic = nbr[si, a_], nbr[si, c_]
        I = la[ibb] + la[ix] - la[ia] - la[ic]; sig = s[ibb] * s[ix] * s[ia] * s[ic]
        dk = DB[a_, c_]
        for k in keys:
            mm = dk == k
            if mm.any(): acc[k] += np.array([mm.sum(), np.sum(I[mm] ** 2), np.sum(np.abs(I[mm])), np.sum(sig[mm] < 0)])
    out = dict(E0_site=float(ev[0]) / N, sec=time.time() - t1,
               by_distance=[dict(dist=k, n=int(a[0]), rms_I=float(np.sqrt(a[1] / a[0])), mean_abs_I=float(a[2] / a[0]),
                                 sign_defect_frac=float(a[3] / a[0])) for k, a in acc.items() if a[0] > 0])
    res['J2'][str(J2)] = out
    print(J2, out['E0_site'], [(r['dist'], round(r['rms_I'], 3), round(r['sign_defect_frac'], 4)) for r in out['by_distance']], flush=True)
    json.dump(res, open(sys.argv[1], 'w'), indent=1)
print('DONE', time.time() - t0)
