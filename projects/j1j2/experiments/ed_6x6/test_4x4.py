"""Validation of the symmetric-ED pipeline and of the psi0 reconstruction on the
4x4 torus (J2/J1 = 0.5), where full S^z=0 ED (dim 12870) is cheap.

Checks:
  1. lowest energy per symmetry sector (lattice_symmetries) vs full ED spectrum;
  2. psi(x) = v_r / sqrt(|orbit r|) reconstructed from the A1,+ vector equals the
     full-ED ground state (|overlap| = 1) and the Psi0 evaluator reproduces it;
  3. orbit sizes sum to the full S^z=0 dimension.
"""
import itertools, json, os, sys, tempfile
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ed_common as ec
from psi0_6x6 import Psi0

L = 4
n = L * L
J2 = 0.5

# --- full ED --------------------------------------------------------------
states = np.array([sum(1 << i for i in c) for c in itertools.combinations(range(n), n // 2)],
                  dtype=np.uint64)
states.sort()
diag, nbr, coeff = ec.local_connections(states, L, 1.0, J2)
rows = np.repeat(np.arange(len(states)), nbr.shape[1])
cols = np.searchsorted(states, nbr.ravel())
H = sp.csr_matrix((coeff.ravel(), (rows, cols)), shape=(len(states),) * 2) + sp.diags(diag)
assert abs(H - H.T).max() < 1e-14
w, V = sla.eigsh(H, k=12, which="SA", tol=1e-13)
o = np.argsort(w); w, V = w[o], V[:, o]
print("full ED lowest:", np.round(w, 8))
psi_full = V[:, 0]

# --- symmetric ED --------------------------------------------------------
res = {}
for k in [(0, 0), (2, 2)]:
    for point in ["A1", "A2", "B1", "B2"] + (["E"] if k == (0, 0) else []):
        for flip in (1, -1):
            basis, G = ec.make_basis(L, k=k, point=point, flip=flip)
            basis.build()
            dim = basis.number_states
            if dim == 0:
                continue
            Hs = ec.make_operator(basis, L, 1.0, J2).to_csr()
            Hs = Hs.real if np.iscomplexobj(Hs.data) else Hs
            if dim < 400:
                e = np.linalg.eigvalsh(Hs.toarray())[:2]
            else:
                e = sla.eigsh(Hs, k=2, which="SA", tol=1e-13)[0]
            key = f"k={k} {point} flip={flip:+d}"
            res[key] = (dim, sorted(e.tolist()))
            print(f"{key:28s} dim={dim:6d} E={np.round(sorted(e), 8)}")
            # every sector eigenvalue must be in the full spectrum
            full_all = np.linalg.eigvalsh(H.toarray()) if 'full_all' not in globals() else full_all
            globals()['full_all'] = full_all
            for ee in e:
                assert np.min(np.abs(full_all - ee)) < 1e-9, (key, ee)

emin = min(v[1][0] for v in res.values())
print("min over sectors", emin, "full", w[0])
assert abs(emin - w[0]) < 1e-10

# --- reconstruction of psi0 from A1,+ ----------------------------------------
basis, G = ec.make_basis(L, k=(0, 0), point="A1", flip=1)
basis.build()
Hs = ec.make_operator(basis, L, 1.0, J2).to_csr().real
e, v = sla.eigsh(Hs, k=1, which="SA", tol=1e-14)
v = v[:, 0]
reps, amp, orb = ec.amplitude_table(basis.states, v, L)
assert orb.sum() == len(states), (orb.sum(), len(states))
with tempfile.TemporaryDirectory() as td:
    fn = os.path.join(td, "t.npz")
    np.savez(fn, L=L, reps=reps, amp=amp, E0=e[0])
    P = Psi0(fn)
    psi_rec = P(states)
ov = float(psi_rec @ psi_full)
nrm = float(psi_rec @ psi_rec)
print(f"E0(A1+)={e[0]:.12f}  norm(rec)={nrm:.14f}  |<rec|full>|={abs(ov):.14f}")
assert abs(nrm - 1) < 1e-10 and abs(abs(ov) - 1) < 1e-10
# local energy with independent full-basis H
le = (diag * psi_rec + (coeff * P(nbr.ravel()).reshape(nbr.shape)).sum(1)) / psi_rec
print("local energy max dev", np.max(np.abs(le - e[0])))
# Marshall
M = P.marshall(states)
s = np.sign(np.sum(psi_rec * M))
ws = float(np.sum(psi_rec ** 2 * (np.sign(psi_rec) != s * M)))
print("Marshall wrong-sign weight 4x4:", ws)
print("ALL 4x4 CHECKS PASSED")
