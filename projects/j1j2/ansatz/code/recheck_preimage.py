"""Re-check (2026-10-26, PI asked to look again): is the J2 pre-image's learnable tail real or Lieb-Mattis dominance?
  python3 recheck_preimage.py LX LY OUT_JSON
For Phi_a = (1 + a H_J2)^{-1} psi0 at cond 3, 5, 10, 30 (a < 0): overlap with the Lieb-Mattis Neel singlet |LM>
(H_J2's top state in the singlet sector: two ferromagnetic J2 sublattices), and a training-free learnability proxy:
rms over antiparallel J1 bonds of d = log|f(y)| - log|f(x)|, by probability decade of x, for f = psi0 and Phi_a.
"""
import sys, json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from math import comb
LX, LY, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
N = LX * LY; J2 = 0.5; T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.1f}s]', *a, flush=True)
def site(x, y): return (x % LX) + LX * (y % LY)
NN = [(site(x, y), site(x + 1, y)) for y in range(LY) for x in range(LX)] + [(site(x, y), site(x, y + 1)) for y in range(LY) for x in range(LX)]
NNN = [(site(x, y), site(x + 1, y + 1)) for y in range(LY) for x in range(LX)] + [(site(x, y), site(x + 1, y - 1)) for y in range(LY) for x in range(LX)]
allx = np.arange(1 << N, dtype=np.int64); pc = np.zeros(1 << N, np.int8)
for i in range(N): pc += ((allx >> i) & 1).astype(np.int8)
basis = allx[pc == N // 2]; del allx, pc; D = basis.size; log('D', D)
def bond(i, j):
    anti = ((basis >> i) & 1) != ((basis >> j) & 1)
    return anti, np.minimum(np.searchsorted(basis, basis ^ ((1 << i) | (1 << j))), D - 1)
def ham(bonds, J):
    rows, cols, dg = [], [], np.zeros(D)
    for (i, j) in bonds:
        anti, nbr = bond(i, j); dg += J * np.where(anti, -0.25, 0.25); r = np.nonzero(anti)[0]; rows.append(r); cols.append(nbr[r])
    r = np.concatenate(rows); c = np.concatenate(cols)
    return (sp.csr_matrix((np.full(r.size, 0.5 * J), (r, c)), shape=(D, D)) + sp.diags(dg)).tocsr()
H1 = ham(NN, 1.0); HJ2 = ham(NNN, J2); H = (H1 + HJ2).tocsr()
E, V = sla.eigsh(H, k=1, which='SA', tol=1e-12); E0 = E[0]; psi0 = V[:, 0]
Aset = [site(x, y) for y in range(LY) for x in range(LX) if (x + y) % 2 == 0]
nA = np.zeros(D, np.int64)
for i in Aset: nA += (basis >> i) & 1
sN = (1 - 2 * (nA & 1)).astype(float)
if psi0 @ (np.abs(psi0) * sN) < 0: psi0 = -psi0
log('E0/N', E0 / N)
# Lieb-Mattis Neel singlet: two ferromagnetic sublattices (spin S = N_A/2 each) coupled to S_tot = 0
NA = len(Aset); S = NA // 2; m = nA - S
LM = sN / np.sqrt((2 * S + 1) * np.array([comb(NA, S + mm) * comb(NA, S - mm) for mm in m], float))
LM /= np.linalg.norm(LM)
res = dict(LX=LX, LY=LY, E0_site=E0 / N, ov_LM_psi0=float((LM @ psi0) ** 2),
           check_LM_is_top_HJ2=float(LM @ (HJ2 @ LM)), E_LM_site_above_E0=float((LM @ (H @ LM) - E0) / N))
lmax = sla.eigsh(HJ2, k=1, which='LA', tol=1e-10)[0][0]; lmin = sla.eigsh(HJ2, k=1, which='SA', tol=1e-10)[0][0]
res['HJ2_spectrum'] = [float(lmin), float(lmax)]; log('LM overlap with psi0', res['ov_LM_psi0'], '<LM|HJ2|LM>', res['check_LM_is_top_HJ2'], 'lmax', lmax)
BD = [bond(i, j) for (i, j) in NN]
def smooth(f, ref):
    """rms of log|f(y)| - log|f(x)| over antiparallel J1 bonds, by decade of ref's per-config probability, weighted by ref^2."""
    lf = np.log(np.maximum(np.abs(f / np.linalg.norm(f)), 1e-300)); p = (ref / np.linalg.norm(ref)) ** 2
    dec = np.floor(np.log10(np.maximum(p, 1e-300))).astype(int); s2 = np.zeros(D); cnt = np.zeros(D)
    for anti, nbr in BD:
        d = np.where(anti, lf[nbr] - lf, 0.0); s2 += d * d; cnt += anti
    msd = s2 / np.maximum(cnt, 1)
    return {str(k): float(np.sqrt(np.sum(p[dec == k] * msd[dec == k]) / np.sum(p[dec == k]))) for k in range(dec.max(), max(dec.min(), -16) - 1, -1) if (dec == k).any() and p[dec == k].sum() > 1e-12}
def WM(f):
    f = f / np.linalg.norm(f); return float(np.sum(f ** 2 * (np.sign(f) * np.sign(np.sum(f * np.abs(f) * sN)) != sN)))
res['smooth_psi0'] = smooth(psi0, psi0); res['scan'] = []
for cond in (3.0, 5.0, 10.0, 30.0):
    a = -(1 - 1 / cond) / (lmax - lmin / cond)             # 1 + a lmax = (1 + a lmin)/cond, a < 0
    phi = sla.minres(sp.identity(D, format='csr') + a * HJ2, psi0, tol=1e-12, maxiter=20000)[0]
    phi /= np.linalg.norm(phi)
    rec = dict(cond=cond, a=float(a), W_M=WM(phi), ov_LM_phi=float((LM @ phi) ** 2), ov_psi0_phi=float((psi0 @ phi) ** 2),
               smooth_phi_by_psi0_decade=smooth(phi, psi0), smooth_phi_by_own_decade=smooth(phi, phi))
    # Phi with its LM component removed: is the remainder still smooth?
    r = phi - (LM @ phi) * LM; rec['W_M_remainder'] = WM(r); rec['smooth_remainder_by_psi0_decade'] = smooth(r, psi0)
    res['scan'].append(rec); log(cond, a, 'W_M', rec['W_M'], 'ov_LM', rec['ov_LM_phi'], 'ov_psi0', rec['ov_psi0_phi'])
    json.dump(res, open(OUT, 'w'), indent=1)
res['sec'] = time.time() - T0; json.dump(res, open(OUT, 'w'), indent=1); log('DONE')
