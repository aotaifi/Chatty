"""C6b-T0 (PLAN_ansatz.md): product of 2-site J2 gates as a non-unitary similarity, exact 4x4.
  python3 c6b_t0.py OUT_JSON
K(a) = prod_b (1 + a h_b) over J2 bonds (fixed order), Ht = K^{-1} H K. Sign exponent per site in the Marshall frame:
g(M) = (E0 - lambda_min(M_s))/N, M_s = D M D with off-diagonals -> -|.|. Reference g(H); also W_M of Phi = K^{-1} psi0
and the off-diagonal weight of Ht by Hamming distance.
"""
import sys, json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
LX = LY = 4; N = 16; J2 = 0.5; OUT = sys.argv[1]; T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.1f}s]', *a, flush=True)
def site(x, y): return (x % LX) + LX * (y % LY)
NN = [(site(x, y), site(x + 1, y)) for y in range(LY) for x in range(LX)] + [(site(x, y), site(x, y + 1)) for y in range(LY) for x in range(LX)]
NNN = [(site(x, y), site(x + 1, y + 1)) for y in range(LY) for x in range(LX)] + [(site(x, y), site(x + 1, y - 1)) for y in range(LY) for x in range(LX)]
allx = np.arange(1 << N, dtype=np.int64); pc = np.zeros(1 << N, np.int8)
for i in range(N): pc += ((allx >> i) & 1).astype(np.int8)
basis = allx[pc == N // 2]; D = basis.size
def SS(i, j, J=1.0):
    """J S_i.S_j on the Sz=0 basis (sparse)."""
    anti = ((basis >> i) & 1) != ((basis >> j) & 1)
    nbr = np.minimum(np.searchsorted(basis, basis ^ ((1 << i) | (1 << j))), D - 1)
    r = np.nonzero(anti)[0]
    off = sp.csr_matrix((np.full(r.size, 0.5 * J), (r, nbr[r])), shape=(D, D))
    return (off + sp.diags(J * np.where(anti, -0.25, 0.25))).tocsr()
H = sum(SS(i, j) for i, j in NN) + sum(SS(i, j, J2) for i, j in NNN); H = H.tocsr()
w, v = sla.eigsh(H, k=1, which='SA', tol=1e-12); E0 = w[0]; psi0 = v[:, 0]
MA = sum(1 << site(x, y) for y in range(LY) for x in range(LX) if (x + y) % 2 == 0)
cnt = np.zeros(D, np.int64)
for i in range(N):
    if (MA >> i) & 1: cnt += (basis >> i) & 1
sN = (1 - 2 * (cnt & 1)).astype(float)
log('E0/N', E0 / N)
def WM(x):
    x = x / np.linalg.norm(x); g = np.sign(np.sum(x * np.abs(x) * sN)) or 1.0
    return float(np.sum(x ** 2 * ((np.sign(x) * g) != sN)))
def g_of(M):
    """(E0 - lambda_min(M_s))/N for dense M (Marshall frame, stoquastised)."""
    Ms = -np.abs(sN[:, None] * M * sN[None, :]); np.fill_diagonal(Ms, np.diag(M))
    c = np.max(np.diag(Ms)) + 1.0                                    # c - Ms is entrywise non-negative
    lam = sla.eigs(c * np.eye(D) - Ms, k=1, which='LR', tol=1e-10)[0][0].real
    return float((E0 - (c - lam)) / N)
hb = [SS(i, j, J2) for i, j in NNN]
SSb = [SS(i, j, 1.0) for i, j in NNN]
I = sp.identity(D, format='csr')
def Ht_of(a):
    """K^{-1} H K with K = prod_b (1 + a h_b); (1 + a h)^{-1} = Pt/(1 + a J2/4) + Ps/(1 - 3 a J2/4)."""
    K = np.eye(D)
    for G in hb: K = (I + a * G) @ K                                 # K = G_last ... G_1
    X = H @ K
    ct, cs = 1.0 / (1 + a * J2 / 4), 1.0 / (1 - 3 * a * J2 / 4)
    for S in SSb:                                                    # K^{-1} = G_1^{-1} ... G_last^{-1}, applied last-first
        Ginv = (ct * (0.75 * I + S) + cs * (0.25 * I - S))
        X = Ginv @ X
    # check K^{-1} K = 1 on one column
    return X, K
res = dict(E0_site=E0 / N, W_M_psi0=WM(psi0))
Hd = H.toarray(); res['g_H'] = g_of(Hd); log('g(H)', res['g_H'])
# check: J2 = 0 is Marshall-stoquastic -> g = 0
H0 = (sum(SS(i, j) for i, j in NN)).toarray(); E0s = E0; E0 = sla.eigsh(sp.csr_matrix(H0), k=1, which='SA')[0][0]
res['check_g_J2=0'] = g_of(H0); E0 = E0s; log('check g(J2=0)', res['check_g_J2=0'])
# Hamming distance between basis states (in units of spin flips / 2)
def ham_profile(M):
    out = {}
    for c0 in range(0, D, 2000):
        xs = basis[c0:c0 + 2000]; x = xs[:, None] ^ basis[None, :]; d = np.zeros(x.shape, np.int8)
        for i in range(N): d += ((x >> i) & 1).astype(np.int8)
        A = np.abs(M[c0:c0 + 2000]); np.fill_diagonal(A[:, c0:c0 + 2000], 0.0) if False else None
        for k in range(2, N + 1, 2):
            out[k // 2] = out.get(k // 2, 0.0) + float(A[d == k].sum())
    tot = sum(out.values()); return {k: v / tot for k, v in out.items() if v > 0}
res['scan'] = []
for a in [-1.5, -1.0, -0.7, -0.5, -0.35, -0.25, -0.15, -0.08, 0.0, 0.08, 0.15, 0.25, 0.5, 1.0]:
    if abs(1 + a * J2 / 4) < 1e-3 or abs(1 - 3 * a * J2 / 4) < 1e-3: continue
    Ht, K = Ht_of(a)
    phi = np.linalg.solve(K, psi0)
    resid = float(np.linalg.norm(Ht @ phi - E0 * phi) / np.linalg.norm(phi))
    rec = dict(a=a, g=g_of(Ht), W_M_phi=WM(phi), eig_resid=resid, condK=float(np.linalg.cond(K)) if D < 20000 else None)
    if a in (-0.5, -0.25, 0.0, 0.25): rec['hamming_profile'] = ham_profile(Ht)
    res['scan'].append(rec); log(rec)
    json.dump(res, open(OUT, 'w'), indent=1)
best = min(res['scan'], key=lambda r: r['g']); res['best'] = best; res['ratio'] = best['g'] / res['g_H']
res['verdict'] = 'PASS' if res['ratio'] <= 0.1 else ('KILL' if res['ratio'] >= 0.5 else 'INCONCLUSIVE')
res['sec'] = time.time() - T0; json.dump(res, open(OUT, 'w'), indent=1); log('VERDICT', res['verdict'], res['ratio'])
