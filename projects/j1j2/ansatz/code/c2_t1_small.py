"""C2-T1 preview on small exact lattices (PLAN_ansatz.md, C2-T1): layers on sign-free (Marshall-signed) bases.
  python3 c2_t1_small.py LX LY OUT_JSON
Bases (all Marshall-signed, positive):  B1 |psi0| (oracle amplitude);  B2 Marshall lattice-FN ground state, guide B1;
  B3 ground state of H_M = H_J1 + diag(H_J2) (J2 hops dropped: stoquastic, no oracle).
Layers: additive (1 + a K) phi, K in {H, H_J2, H_J1};  product phi(x) prod_b (1 + a e_b(x)) over J2 bonds,
  e_b(x) = J2 [S^z S^z]_b + (J2/2) phi(flip_b x)/phi(x) [antiparallel] (in the phi sign convention);  per-plaquette product.
Measured: E_var(a*) - E0 per site, E_FN (lattice FN, sign and guide of the layered state) - E0 per site, W_M, overlap.
"""
import sys, json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import minimize_scalar
LX, LY, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
N = LX * LY; J2 = 0.5; T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
def site(x, y): return (x % LX) + LX * (y % LY)
NN = [(site(x, y), site(x + 1, y)) for y in range(LY) for x in range(LX)] + \
     [(site(x, y), site(x, y + 1)) for y in range(LY) for x in range(LX)]
NNN = [(site(x, y), site(x + 1, y + 1)) for y in range(LY) for x in range(LX)] + \
      [(site(x, y), site(x + 1, y - 1)) for y in range(LY) for x in range(LX)]
PLAQ = [[(site(x, y), site(x + 1, y + 1)), (site(x, y + 1), site(x + 1, y))] for y in range(LY) for x in range(LX)]
allx = np.arange(1 << N, dtype=np.int64); pc = np.zeros(1 << N, np.int8)
for i in range(N): pc += ((allx >> i) & 1).astype(np.int8)
basis = allx[pc == N // 2]; del allx, pc; D = basis.size; log('D', D)
def bond_data(i, j):
    anti = ((basis >> i) & 1) != ((basis >> j) & 1)
    nbr = np.minimum(np.searchsorted(basis, basis ^ ((1 << i) | (1 << j))), D - 1)   # parallel bonds: masked by anti
    return anti, nbr
def ham(bonds, J, diag=True, off=True):
    rows, cols, vals = [], [], []; dg = np.zeros(D)
    for (i, j) in bonds:
        anti, nbr = bond_data(i, j); dg += J * np.where(anti, -0.25, 0.25)
        if off:
            r = np.nonzero(anti)[0]; rows.append(r); cols.append(nbr[r]); vals.append(np.full(r.size, 0.5 * J))
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(D, D)) if off else sp.csr_matrix((D, D))
    return (M + sp.diags(dg)).tocsr() if diag else M
H1 = ham(NN, 1.0); HJ2 = ham(NNN, J2); H = (H1 + HJ2).tocsr(); DJ2 = HJ2.diagonal()
E, V = sla.eigsh(H, k=1, which='SA', tol=1e-12); E0 = E[0]; psi0 = V[:, 0]
MA = sum(1 << site(x, y) for y in range(LY) for x in range(LX) if (x + y) % 2 == 0)
cnt = np.zeros(D, np.int64)
for i in range(N):
    if (MA >> i) & 1: cnt += (basis >> i) & 1
sN = (1 - 2 * (cnt & 1)).astype(float)
if psi0 @ (np.abs(psi0) * sN) < 0: psi0 = -psi0
log('E0/N', E0 / N)
def en(v): return float(v @ (H @ v) / (v @ v))
def WM(v):
    v = v / np.linalg.norm(v); return float(np.sum(v ** 2 * ((np.sign(v) * np.sign(np.sum(v * np.abs(v) * sN))) != sN)))
def fn_energy(v):
    """lattice FN (ten Haaf): sign and guide from v."""
    s = np.where(v >= 0, 1.0, -1.0); g = np.maximum(np.abs(v), 1e-300)
    C = H.tocoo(); off = C.row != C.col
    r, c, h = C.row[off], C.col[off], C.data[off]
    viol = h * s[r] * s[c] > 0
    keep = sp.csr_matrix((h[~viol], (r[~viol], c[~viol])), shape=(D, D))
    vsf = np.bincount(r[viol], weights=h[viol] * s[r[viol]] * s[c[viol]] * g[c[viol]] / g[r[viol]], minlength=D)
    HF = keep + sp.diags(H.diagonal() + vsf)
    return float(sla.eigsh(HF, k=1, which='SA', tol=1e-10)[0][0])
res = dict(LX=LX, LY=LY, D=int(D), E0_site=E0 / N, W_M_psi0=WM(psi0))
# ------------------------------------------------ bases
B1 = np.abs(psi0) * sN
B2 = np.abs(sla.eigsh(sp.csr_matrix(H.shape), k=1)[1][:, 0]) if False else None
# Marshall FN ground state with guide B1
s_ = sN; g = np.abs(B1); C = H.tocoo(); off = C.row != C.col
r, c, h = C.row[off], C.col[off], C.data[off]; viol = h * s_[r] * s_[c] > 0
HF = sp.csr_matrix((h[~viol], (r[~viol], c[~viol])), shape=(D, D)) + sp.diags(H.diagonal() + np.bincount(
    r[viol], weights=h[viol] * s_[r[viol]] * s_[c[viol]] * g[c[viol]] / np.maximum(g[r[viol]], 1e-300), minlength=D))
ef, uf = sla.eigsh(HF, k=1, which='SA', tol=1e-12); B2 = np.abs(uf[:, 0]) * sN
HM = (H1 + sp.diags(DJ2)).tocsr()
em, um = sla.eigsh(HM, k=1, which='SA', tol=1e-12); B3 = np.abs(um[:, 0]) * sN
bases = {'B1_abs_psi0': B1, 'B2_marshall_FN': B2, 'B3_noJ2hop_GS': B3}
res['E_FN_marshall_guideB1'] = (ef[0] - E0) / N
# ------------------------------------------------ J2 bond data for the product layers
BD = [bond_data(i, j) for (i, j) in NNN]
def e_bonds(phi):
    """per-J2-bond local energy e_b(x) = J2 zz + (J2/2) phi(y)/phi(x) on antiparallel bonds."""
    out = []
    for (anti, nbr), (i, j) in zip(BD, NNN):
        zz = np.where(anti, -0.25, 0.25) * J2
        out.append(zz + np.where(anti, 0.5 * J2 * phi[nbr] / np.where(phi == 0, 1e-300, phi), 0.0))
    return np.array(out)                                        # (n_bonds, D)
bidx = {b: k for k, b in enumerate(NNN)}
def layered(phi, kind, a, EB=None):
    if kind == 'add_H': return phi + a * (H @ phi)
    if kind == 'add_HJ2': return phi + a * (HJ2 @ phi)
    if kind == 'add_HJ1': return phi + a * (H1 @ phi)
    if kind == 'prod_bond': return phi * np.prod(1.0 + a * EB, axis=0)
    if kind == 'prod_plaq':
        f = np.ones(D)
        for pq in PLAQ: f *= 1.0 + a * (EB[bidx[pq[0]]] + EB[bidx[pq[1]]])
        return phi * f
res['bases'] = {}
for bn, phi in bases.items():
    phi = phi / np.linalg.norm(phi); EB = e_bonds(phi); rb = dict(E_var=(en(phi) - E0) / N, E_FN=(fn_energy(phi) - E0) / N,
                                                                 overlap=float(abs(phi @ psi0)), W_M=WM(phi))
    log(bn, 'base', rb)
    for kind in ('add_H', 'add_HJ2', 'add_HJ1', 'prod_bond', 'prod_plaq'):
        f = lambda a: en(layered(phi, kind, a, EB))
        # coarse grid then bounded refinement (E(a) is not convex for the product forms)
        grid = np.concatenate([-np.logspace(-3, 0.5, 30)[::-1], [0.0], np.logspace(-3, 0.5, 30)])
        vals = [f(a) for a in grid]; k = int(np.argmin(vals)); lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
        r_ = minimize_scalar(f, bounds=(lo, hi), method='bounded', options=dict(xatol=1e-6))
        a = float(r_.x) if r_.fun <= vals[k] else float(grid[k]); psi = layered(phi, kind, a, EB)
        rb[kind] = dict(a=a, E_var=(en(psi) - E0) / N, E_FN=(fn_energy(psi) - E0) / N,
                        overlap=float(abs(psi @ psi0) / np.linalg.norm(psi)), W_M=WM(psi))
        log(bn, kind, rb[kind])
    res['bases'][bn] = rb
    json.dump(res, open(OUT, 'w'), indent=1)
res['sec'] = time.time() - T0; json.dump(res, open(OUT, 'w'), indent=1); log('DONE')
