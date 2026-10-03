"""Recompute the 4x4 one-step Krylov benchmark at J2/J1=0.5 and store r(x) for the threshold panel.

Guide: psi_0(x) = s_Marshall(x) |psi_ED(x)|  (exact amplitude, Marshall signs).
Update: s_1(x) = s_0(x) sgn[T - r(x)],  r(x) = (H psi_0)(x) / psi_0(x),
with T chosen by minimizing the fixed-amplitude energy (no sign labels used).
Same construction as krylov_sign_structure/experiments/square_exact_energyopt.py.
"""
from pathlib import Path
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

L = 4; N = L * L; J2 = 0.5
OUT = Path(__file__).resolve().parents[1] / "data" / "k1_threshold_4x4_J2p5.npz"

site = lambda x, y: (x % L) + L * (y % L)
NN, NNN = [], []
for y in range(L):
    for x in range(L):
        i = site(x, y)
        NN += [(i, site(x + 1, y)), (i, site(x, y + 1))]
        NNN += [(i, site(x + 1, y + 1)), (i, site(x + 1, y - 1))]
basis = np.array([s for s in range(1 << N) if s.bit_count() == N // 2], np.uint32)
D = len(basis); idx = {int(s): i for i, s in enumerate(basis)}

rr, cc, vv = [], [], []; diag = np.zeros(D)
for bi, s0 in enumerate(basis):
    s = int(s0); e = 0.
    for bonds, J in ((NN, 1.0), (NNN, J2)):
        for u, v in bonds:
            if ((s >> u) & 1) == ((s >> v) & 1):
                e += .25 * J
            else:
                e -= .25 * J
                rr.append(bi); cc.append(idx[s ^ (1 << u) ^ (1 << v)]); vv.append(.5 * J)
    diag[bi] = e
H = sp.coo_matrix((vv + list(diag), (rr + list(range(D)), cc + list(range(D)))), shape=(D, D)).tocsr()

ev, V = sla.eigsh(H, k=1, which="SA", tol=1e-12)
E0 = float(ev[0]); vec = V[:, 0]
vec = vec if vec[np.argmax(abs(vec))] > 0 else -vec
a = np.abs(vec); truth = np.where(vec >= 0, 1, -1)

mask = sum(1 << site(x, y) for y in range(L) for x in range(L) if (x + y) % 2 == 0)
s0 = np.array([1 if (int(s) & mask).bit_count() % 2 == 0 else -1 for s in basis])
if np.sum(a * a * s0 * truth) < 0:
    s0 = -s0  # fix the irrelevant global sign
psi0 = a * s0
r = (H @ psi0) / psi0

# exact energy-optimal threshold: sweep T upward through sorted r, flipping one sign at a time
W = H.copy().tolil(); W.setdiag(0); W = sp.diags(a) @ W.tocsr() @ sp.diags(a)
s = -s0.copy(); E = float(np.sum(diag * a * a)) + float(s @ (W @ s)); h = np.asarray(W @ s).ravel()
best = (E, -np.inf)
for ii in np.argsort(r):
    old = s[ii]; E -= 4. * old * h[ii]; s[ii] = -old
    st, en = W.indptr[ii], W.indptr[ii + 1]; h[W.indices[st:en]] -= 2. * old * W.data[st:en]
    if E < best[0] - 1e-13:
        best = (E, r[ii])
T = float(best[1])
s1 = s0 * np.where(r <= T, 1, -1)

p = a * a / np.sum(a * a)
res = dict(E0=E0, T=T,
           wrong0=float(p[s0 != truth].sum()), wrong1=float(p[s1 != truth].sum()),
           eps0=float((psi0 @ H @ psi0 - E0) / abs(E0)),
           eps1=float(((a * s1) @ H @ (a * s1) - E0) / abs(E0)))
print(json.dumps(res, indent=1))
np.savez_compressed(OUT, r=r, p=p, is_wrong0=(s0 != truth), is_wrong1=(s1 != truth), **res)
print("wrote", OUT)
