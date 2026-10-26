"""Local vs additive undoing of J2 (2026-10-26 re-check): does the strength needed to make the pre-image sign-free
grow with N?   python3 local_undo.py LX LY OUT_JSON
Forms: additive Phi = (1 + a H_J2)^{-1} psi0 (a < 0); exponential Phi = exp(t H_J2) psi0 (t > 0, i.e. psi0 =
exp(-t H_J2) Phi: Phi evolved in imaginary time under J2); gate product Phi = prod_b (1 + a h_b)^{-1} psi0 (fixed order).
Per form and strength: W_M(Phi)/W_M(psi0), overlap with the Lieb-Mattis Neel singlet, overlap with psi0.
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
def SS(i, j, J):
    anti = ((basis >> i) & 1) != ((basis >> j) & 1)
    nbr = np.minimum(np.searchsorted(basis, basis ^ ((1 << i) | (1 << j))), D - 1); r = np.nonzero(anti)[0]
    return (sp.csr_matrix((np.full(r.size, 0.5 * J), (r, nbr[r])), shape=(D, D)) + sp.diags(J * np.where(anti, -0.25, 0.25))).tocsr()
HJ1 = sum(SS(i, j, 1.0) for i, j in NN).tocsr(); hb = [SS(i, j, J2) for i, j in NNN]; HJ2 = sum(hb).tocsr(); H = (HJ1 + HJ2).tocsr()
E, V = sla.eigsh(H, k=1, which='SA', tol=1e-12); E0 = E[0]; psi0 = V[:, 0]
Aset = [site(x, y) for y in range(LY) for x in range(LX) if (x + y) % 2 == 0]; nA = np.zeros(D, np.int64)
for i in Aset: nA += (basis >> i) & 1
sN = (1 - 2 * (nA & 1)).astype(float)
if psi0 @ (np.abs(psi0) * sN) < 0: psi0 = -psi0
NA = len(Aset); S = NA // 2
LM = sN / np.sqrt(np.array([comb(NA, S + mm) * comb(NA, S - mm) for mm in (nA - S)], float)); LM /= np.linalg.norm(LM)
def WM(f):
    f = f / np.linalg.norm(f); return float(np.sum(f ** 2 * (np.sign(f) * np.sign(np.sum(f * np.abs(f) * sN)) != sN)))
W0 = WM(psi0); log('E0/N', E0 / N, 'W_M psi0', W0, 'ov_LM', float((LM @ psi0) ** 2))
res = dict(LX=LX, LY=LY, E0_site=E0 / N, W_M_psi0=W0, ov_LM_psi0=float((LM @ psi0) ** 2), forms={})
def rec(phi, **kw):
    phi = phi / np.linalg.norm(phi)
    return dict(kw, ratio=WM(phi) / W0, ov_LM=float((LM @ phi) ** 2), ov_psi0=float((psi0 @ phi) ** 2))
lmax = sla.eigsh(HJ2, k=1, which='LA', tol=1e-10)[0][0]; lmin = sla.eigsh(HJ2, k=1, which='SA', tol=1e-10)[0][0]
out = []
for cond in (1.5, 2, 3, 5, 10, 30):
    a = -(1 - 1 / cond) / (lmax - lmin / cond)
    out.append(rec(sla.minres(sp.identity(D, format='csr') + a * HJ2, psi0, tol=1e-12, maxiter=20000)[0], cond=cond, a=float(a)))
    log('add', out[-1])
res['forms']['additive'] = out; out = []
for t in (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5):
    out.append(rec(sla.expm_multiply(t * HJ2, psi0), t=t)); log('exp', out[-1])
res['forms']['exponential'] = out; out = []
I = sp.identity(D, format='csr')
for a in (-0.1, -0.25, -0.5, -0.75, -1.0, -1.5):
    ct, cs = 1.0 / (1 + a * J2 / 4), 1.0 / (1 - 3 * a * J2 / 4); phi = psi0.copy()
    for h in reversed(hb):                                  # K = G_last ... G_1, K^{-1} psi0 = G_1^-1 ... G_last^-1 psi0
        S_ = h / J2; phi = (ct * (0.75 * I + S_) + cs * (0.25 * I - S_)) @ phi
    out.append(rec(phi, a=a, gate_cond=float(max(abs(1 / ct), abs(1 / cs)) / min(abs(1 / ct), abs(1 / cs))))); log('gates', out[-1])
res['forms']['gates'] = out
res['sec'] = time.time() - T0; json.dump(res, open(OUT, 'w'), indent=1); log('DONE')
