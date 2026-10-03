"""Exact 4x4 FN/Krylov loop with the walker-learning half step.

Same as closed_fn_krylov_exact4x4.py, but the new amplitude is the maximum-likelihood
target of a model trained on FN walkers, a_new^2 ∝ a_old * phi_FN (paper Eq. 7),
instead of the full step a_new = phi_FN. The Krylov sign step uses a_new.
Start: exact J2=0 amplitude, Marshall signs; target J2/J1=0.5. ED used only for scoring.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_fn_krylov_exact4x4 as L

MAXITER = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = Path(__file__).resolve().parents[1] / "results" / "closed_fn_krylov_4x4_J2p5_J2zero_init_halfstep.json"

H, diag = L.build_H(0.5)
E0, psi0 = L.ground(H)
sM = L.canonical(L.marshall_signs())
if np.dot(psi0, sM) < 0: psi0 = -psi0
atrue = np.abs(psi0); atrue /= np.linalg.norm(atrue)
strue = L.canonical(np.where(psi0 >= 0, 1, -1).astype(np.int8)); ptrue = atrue * atrue
Hi, _ = L.build_H(0.0)
_, pi = L.ground(Hi)
a = np.abs(pi); a /= np.linalg.norm(a); s = sM.copy()

def rec(it):
    return dict(it=it, E_error=float(L.physical_energy(H, a, s) - E0),
                O_sign=float(abs(np.sum(ptrue * s * strue))), F_amp=float(np.dot(a, atrue) ** 2))

hist = [rec(0)]
for it in range(1, MAXITER + 1):
    _, afn = L.fixed_node_solve(H, diag, a, s)
    afn = np.abs(afn); afn /= np.linalg.norm(afn)
    a = np.sqrt(a * afn); a /= np.linalg.norm(a)          # half step
    s, *_ = L.projected_krylov_update(H, diag, a, s)
    hist.append(rec(it))
    if it % 10 == 0:
        print(it, hist[-1], flush=True)
json.dump(dict(E0=E0, update="half step a_new^2 ∝ a phi_FN", history=hist), open(OUT, "w"), indent=1)
print("wrote", OUT)
