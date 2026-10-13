"""Gate V2 (PREREG E3): the three-hop recursion of dc_p2.threehop and the psi_2 local-energy formula, on 4x4 (Sz = 0),
against exact sparse-matrix vectors.  Guide: |psi0| exp(0.3 xi) with 10% sign flips (not an eigenstate).
  python dc_p2_gate4x4.py OUT.json
"""
import sys, json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import dc_fn as F
import dc_p2 as P

lat = F.Lattice(4); N = 16
st = np.array([s for s in range(1 << N) if bin(s).count('1') == N // 2], np.uint64); D = len(st)
NBs, V = lat.neighbors(st)
H = sp.csr_matrix((0.5 * np.broadcast_to(lat.JB, (D, lat.NB))[V], (np.repeat(np.arange(D), lat.NB)[V.reshape(-1)],
                   np.searchsorted(st, NBs[V]))), shape=(D, D)) + sp.diags(lat.diag(V))
w, v = sla.eigsh(H, k=1, which='SA'); psi0 = v[:, 0]
rg = np.random.default_rng(5)
la = np.log(np.abs(psi0)) + 0.3 * rg.standard_normal(D); s = np.sign(psi0); s[rg.random(D) < 0.1] *= -1
psi = s * np.exp(la); Er = -8.3
A = H - Er * sp.identity(D)
v1 = A @ psi; v2 = A @ v1; v3 = A @ v2
ex = [v1 / psi, v2 / psi, v3 / psi]
guide = lambda S: (la[np.searchsorted(st, S)], s[np.searchsorted(st, S)])
idx = rg.choice(D, 200, replace=False); err = []
for i in idx:
    d = P.threehop(lat, guide, st[i], Er)
    err.append(max(abs(d['a%d' % (k + 1)] - ex[k][i]) / max(1.0, abs(ex[k][i])) for k in range(3)))
b1, b2 = -0.2, 0.015
g = 1 + b1 * ex[0] + b2 * ex[1]; psi2 = psi * g
EL2 = Er + (ex[0] + b1 * ex[1] + b2 * ex[2]) / g
E_formula = np.sum(psi2 ** 2 * EL2) / np.sum(psi2 ** 2); E_exact = psi2 @ (H @ psi2) / (psi2 @ psi2)
# Ritz check: ritz3 with exact psi^2 weights over all states vs dense 3x3 Krylov Ritz
ev, c1, c2, mu = P.ritz3(ex[0], ex[1], ex[2], psi ** 2, 'balanced')
K = np.stack([psi, v1, v2], 1); Sm = K.T @ K; Hm = K.T @ (H @ K)
from scipy.linalg import eigh
evx = eigh(Hm, Sm, eigvals_only=True)[0]
res = dict(max_rel_err_a=float(max(err)), E_formula=E_formula, E_exact=E_exact, rel=abs(E_formula - E_exact) / abs(E_exact),
           ritz_shift=ev, ritz_exact=evx - Er, ritz_rel=abs(ev - (evx - Er)) / abs(evx),
           PASS=bool(max(err) < 1e-9 and abs(E_formula - E_exact) / abs(E_exact) < 1e-9 and abs(ev - (evx - Er)) / abs(evx) < 1e-9))
print(json.dumps(res, indent=1)); json.dump(res, open(sys.argv[1], 'w'), indent=1)
