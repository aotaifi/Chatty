#!/usr/bin/env python3
"""Validate the symmetric-sector FN/Krylov machinery (closed_fn_krylov_sym6x6.py) against the
full-basis 4x4 reference (closed_fn_krylov_anderson.py): orbit-invariance of r, H_FN matrix action,
FN ground state, Krylov update (groups, threshold energy, signs)."""
import os, sys, json
import numpy as np
import scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import closed_fn_krylov_anderson as ref
import closed_fn_krylov_sym6x6 as sym
ec = sym.ec

S = sym.Sym(4, 0.5)
# python 3.9 (ls_env) has no int.bit_count: build the reference Model by hand (same data as ref.Model("4x4"))
M = ref.Model.__new__(ref.Model)
M.N, M.NN, M.NNN, M.mask, M.geom = ref.make_lattice("4x4")
M.basis = np.array([s for s in range(1 << M.N) if bin(s).count("1") == M.N // 2], dtype=np.uint32)
M.D = len(M.basis)
M.state_to_idx = np.full(1 << M.N, -1, dtype=np.int32); M.state_to_idx[M.basis] = np.arange(M.D, dtype=np.int32)
H, diag, ei, ej, hij = M.build_H(0.5)
basis64 = M.basis.astype(np.uint64)
reps, orb = ec.rep_and_orbit(basis64, S.T, 16)
ridx = np.searchsorted(S.states, reps)
assert np.array_equal(S.states[ridx], reps)
assert np.allclose(orb, S.n_orb[ridx])
print("full dim", M.D, "sym dim", S.D, "orbit sizes sum", int(S.n_orb.sum()))
# Marshall consistency with reference full basis signs
sMfull = np.array([1 if bin(int(x) & M.mask).count("1") % 2 == 0 else -1 for x in M.basis], dtype=np.int8)
assert np.array_equal(sMfull, S.marshall[ridx]) or np.array_equal(sMfull, -S.marshall[ridx]), "Marshall mismatch"
print("Marshall signs consistent with full-basis reference")

def lift(vec):
    return vec[ridx]

def full_FN_matrix(a, s):
    D = len(a)
    af = np.maximum(a, 1e-15)
    rr = np.r_[ei, ej]; cc = np.r_[ej, ei]; hh = np.r_[hij, hij]
    kij = s[rr].astype(float) * hh * s[cc].astype(float)
    keep = kij < 0; bad = ~keep
    corr = np.bincount(rr[bad], weights=kij[bad] * af[cc[bad]] / af[rr[bad]], minlength=D)
    d = diag + corr
    rows = np.r_[rr[keep], np.arange(D)]; cols = np.r_[cc[keep], np.arange(D)]
    vals = np.r_[kij[keep], d]
    return sp.coo_matrix((vals, (rows, cols)), shape=(D, D)).tocsr()

rng = np.random.default_rng(0)
vinit, Einit = sym.init_amplitude(S, 0.0, 1e-13)
vinit /= np.linalg.norm(vinit)
tests = []
tests.append(("init Marshall", vinit.copy(), S.marshall.copy()))
vn = vinit * np.exp(0.3 * rng.standard_normal(S.D)); vn /= np.linalg.norm(vn)
tests.append(("noisy amp Marshall", vn, S.marshall.copy()))
# a few Krylov-evolved sign patterns
v, s = vinit.copy(), S.marshall.copy()
for k in range(4):
    e, af, _ = S.fn_solve(v, s, 1e-13)
    s, _, _, _ = S.krylov(af, s); v = af
    tests.append((f"after {k+1} plain steps", v.copy(), s.copy()))
ok_all = True
for name, v, s in tests:
    v = v / np.linalg.norm(v)
    a_full = lift(v) / np.sqrt(S.n_orb[ridx]); s_full = lift(s)
    # (1) r orbit-invariance / equality with the full-basis local energy
    psi_full = a_full * s_full
    r_full = (H @ psi_full) / psi_full
    psi_sym = v * s
    r_sym = S.H(psi_sym) / psi_sym
    d_r = np.max(np.abs(r_full - lift(r_sym)))
    # (2) H_FN action on random symmetric vector (comps u_r -> per-config u_r/sqrt(n_r))
    F = full_FN_matrix(a_full, s_full)
    u = rng.standard_normal(S.D)
    yfull = F @ (lift(u) / np.sqrt(S.n_orb[ridx]))
    # sym-basis action
    af_ = np.maximum(v / S.sqrt_n, 1e-15); w = af_ * S.sqrt_n
    Dg = np.empty(S.D); sym.k_fn_diag(S.indptr, S.indices, S.data, s, w, Dg)
    ys = np.empty(S.D); sym.k_fn_matvec(S.indptr, S.indices, S.data, s, Dg, u, ys)
    d_F = np.max(np.abs(yfull - lift(ys) / np.sqrt(S.n_orb[ridx])))
    # also check yfull is orbit constant
    # (3) FN ground state
    efr, afr = ref.fixed_node_solve(H, diag, ei, ej, hij, a_full, s_full, 1e-13)
    efs, vfs, nm = S.fn_solve(v, s, 1e-13)
    d_E = abs(efr - efs); d_a = np.max(np.abs(afr - lift(vfs) / np.sqrt(S.n_orb[ridx])))
    # (4) Krylov
    snr, ethr, Gr = ref.projected_krylov_update(H, diag, ei, ej, hij, a_full, s_full)
    sns, eths, Gs, _ = S.krylov(v, s)
    snl = lift(sns); snl = snl if snl[0] == snr[0] else -snl
    nflip = int(np.sum(snl != snr))
    print(f"{name:24s} |dr|={d_r:.2e} |dF y|={d_F:.2e} |dE_FN|={d_E:.2e} |da_FN|={d_a:.2e} "
          f"G {Gr}/{Gs} dEth={abs(ethr-eths):.2e} sign mismatches={nflip}")
    ok = d_r < 1e-9 and d_F < 1e-11 and d_E < 1e-10 and d_a < 1e-8 and Gr == Gs and abs(ethr - eths) < 1e-10 and nflip == 0
    ok_all &= ok
print("ALL ONE-STEP CHECKS PASSED" if ok_all else "ONE-STEP CHECKS FAILED")
sys.exit(0 if ok_all else 1)
