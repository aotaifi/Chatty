"""Exact 4x4 Krylov sign benchmark with the exact ED amplitude held fixed.

Guide: psi_0(x) = s_ref(x) |psi_ED(x)|, with s_ref = Marshall (or stripe for J2/J1 >= 0.8).
Krylov sign step: s_{k+1}(x) = s_k(x) sgn[T_k - r_k(x)],  r_k = (H a s_k) / (a s_k),
where T_k minimizes the energy of a s_{k+1} over all thresholds (no sign labels used).

Writes
  data/k1_threshold_4x4_J2p5.npz : r(x), |psi_0|^2 and wrong-sign flags at J2/J1=0.5 (Fig. 1a)
  data/k1_iterated_4x4.json      : errors versus number of Krylov steps (Fig. 1b,c)
  data/k1_approx_amp_4x4.json    : same at J2/J1=0.5 with the amplitude of another J2 (Fig. 2)
Same construction as krylov_sign_structure/experiments/square_exact_energyopt.py and
iterated_krylov_exact.py.
"""
from pathlib import Path
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

L = 4; N = L * L
DATA = Path(__file__).resolve().parents[1] / "data"
STARTS = [(0.4, "marshall"), (0.5, "marshall"), (0.6, "marshall"), (0.8, "stripe_x"), (1.0, "stripe_x")]
NSTEPS = 6

site = lambda x, y: (x % L) + L * (y % L)
NN, NNN = [], []
for y in range(L):
    for x in range(L):
        i = site(x, y)
        NN += [(i, site(x + 1, y)), (i, site(x, y + 1))]
        NNN += [(i, site(x + 1, y + 1)), (i, site(x + 1, y - 1))]
basis = np.array([s for s in range(1 << N) if s.bit_count() == N // 2], np.uint32)
D = len(basis); idx = {int(s): i for i, s in enumerate(basis)}


def hamiltonian(J2):
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
    return H, diag


def reference_sign(kind):
    on = {"marshall": lambda x, y: (x + y) % 2 == 0, "stripe_x": lambda x, y: x % 2 == 0}[kind]
    mask = sum(1 << site(x, y) for y in range(L) for x in range(L) if on(x, y))
    return np.array([1 if (int(s) & mask).bit_count() % 2 == 0 else -1 for s in basis])


def krylov_step(H, diag, a, s):
    """One label-free Krylov sign step; returns new signs, r(x) and the threshold T."""
    psi = a * s; r = (H @ psi) / psi
    W = H.copy().tolil(); W.setdiag(0); W = sp.diags(a) @ W.tocsr() @ sp.diags(a)
    # sweep T upward through sorted r: flip one sign at a time, track the exact energy
    t = -s.copy(); E = float(np.sum(diag * a * a)) + float(t @ (W @ t)); h = np.asarray(W @ t).ravel()
    best = (E, -np.inf)
    for ii in np.argsort(r):
        old = t[ii]; E -= 4. * old * h[ii]; t[ii] = -old
        st, en = W.indptr[ii], W.indptr[ii + 1]; h[W.indices[st:en]] -= 2. * old * W.data[st:en]
        if E < best[0] - 1e-13:
            best = (E, r[ii])
    T = float(best[1])
    return s * np.where(r <= T, 1, -1), r, T


rows = []
for J2, kind in STARTS:
    H, diag = hamiltonian(J2)
    ev, V = sla.eigsh(H, k=1, which="SA", tol=1e-12)
    E0 = float(ev[0]); vec = V[:, 0]
    a = np.abs(vec); truth = np.where(vec >= 0, 1, -1); p = a * a / np.sum(a * a)
    s = reference_sign(kind)
    if np.sum(p * s * truth) < 0:
        s = -s  # fix the irrelevant global sign
    hist = []
    for k in range(NSTEPS + 1):
        if np.sum(p * s * truth) < 0:
            s = -s  # a global sign flip is the same state; fix the gauge before scoring
        psi = a * s
        hist.append(dict(step=k, wrong=float(p[s != truth].sum()),
                         eps=float((psi @ H @ psi - E0) / abs(E0))))
        if k == NSTEPS:
            break
        s_new, r, T = krylov_step(H, diag, a, s)
        if J2 == 0.5 and k == 0:
            np.savez_compressed(DATA / "k1_threshold_4x4_J2p5.npz", r=r, p=p, T=T, E0=E0,
                                is_wrong0=(s != truth), is_wrong1=(s_new != truth))
        s = s_new
    rows.append(dict(J2=J2, start=kind, E0=E0, history=hist))
    print(J2, kind, " ".join(f"{h['wrong']:.2e}" for h in hist))

json.dump(dict(system="4x4 periodic, Sz=0, exact ED amplitude held fixed", rows=rows),
          open(DATA / "k1_iterated_4x4.json", "w"), indent=1)
print("wrote", DATA / "k1_iterated_4x4.json")

# Approximate amplitude: target J2/J1=0.5, amplitude = exact |psi| of a different J2 (structured error)
TARGET = 0.5; SOURCES = [0.0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6]; ASTEPS = 8
H, diag = hamiltonian(TARGET)
ev, V = sla.eigsh(H, k=1, which="SA", tol=1e-12)
E0 = float(ev[0]); vec = V[:, 0]; a0 = np.abs(vec); truth = np.where(vec >= 0, 1, -1)
amp_rows = []
for J2s in SOURCES:
    Hs, _ = hamiltonian(J2s)
    a = np.abs(sla.eigsh(Hs, k=1, which="SA", tol=1e-12)[1][:, 0]); a /= np.linalg.norm(a)
    fid = float((a @ a0) ** 2 / (a0 @ a0))
    p = a * a
    s = reference_sign("marshall")
    best_signs = float(((a * truth) @ H @ (a * truth) - E0) / abs(E0))  # this amplitude with exact signs
    hist = []
    for k in range(ASTEPS + 1):
        if np.sum(p * s * truth) < 0:
            s = -s
        psi = a * s
        hist.append(dict(step=k, wrong=float((a0 ** 2)[s != truth].sum() / (a0 @ a0)),
                         eps=float((psi @ H @ psi - E0) / abs(E0))))
        if k < ASTEPS:
            s = krylov_step(H, diag, a, s)[0]
    amp_rows.append(dict(amp_source_J2=J2s, amp_fidelity=fid, eps_exact_signs=best_signs, history=hist))
    print(f"amp from {J2s}: F={fid:.4f}  eps {hist[0]['eps']:.2e} -> {hist[-1]['eps']:.2e}  "
          f"(exact signs {best_signs:.2e})  wrong {hist[0]['wrong']:.2e} -> {hist[-1]['wrong']:.2e}")
json.dump(dict(target_J2=TARGET, E0=E0, start="marshall", rows=amp_rows),
          open(DATA / "k1_approx_amp_4x4.json", "w"), indent=1)
print("wrote", DATA / "k1_approx_amp_4x4.json")
