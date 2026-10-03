#!/usr/bin/env python3
"""Sampled version of the exact 4x4 FN/Krylov loop (paper Fig. 4), level (A): i.i.d. samples.

Exact loop (krylov_sign_structure/experiments/closed_fn_krylov_exact4x4.py):
  guide (a, s) -> ground state phi_FN of lattice fixed-node H_FN[a, s]
  -> s' = s * sgn[T - r], r = (H phi s)/(phi s), T exact energy-optimal (grouped)
  -> new guide (phi_FN, s').

Sampled loop: the ONLY change is that phi_FN is replaced by a histogram estimate from
N i.i.d. samples of the exact mixed distribution f(x) = a(x) phi_FN(x) / sum(a phi_FN):
  counts ~ Multinomial(N, f),  phi_hat(x) = counts(x) / (N a(x))     (visited x)
Unvisited-state rule ("cap", default): keep the old guide amplitude shape with the global
ratio C = sum_vis phi_hat a / sum_vis a^2, but never let it exceed the half-count bound
(the Jeffreys posterior mean of f for zero counts is ~0.5/N):
  phi_hat(x) = min( C a(x), 0.5 / (N a(x)) )                          (unvisited x)
Rule "old" drops the cap (phi_hat = C a on unvisited states).
Rule "kt" (Krichevsky-Trofimov / Jeffreys pseudocount, all states): phi_hat(x) = (count(x)+1/2)/(N a(x)).
H_FN, its exact ground state, and the Krylov step with the exact energy-optimal threshold
(computed on the noisy amplitude phi_hat) are all exact. N=0 means no sampling (exact loop).
Target ED (E0, a0, true signs) is used for diagnostics only.

Observables per iteration (trial state a*s after the FN->amplitude->Krylov update):
  w_s = sum_{x: s(x) != s_true(x)} a0(x)^2 (global sign fixed: min(w, 1-w)),
  eps = (E[a s] - E0)/|E0|.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, "/Users/aliotaifi/Chatty-organize/projects/j1j2/krylov_sign_structure/experiments")
import closed_fn_krylov_exact4x4 as ref  # noqa: E402  (reference building blocks; unmodified)

D = ref.D


def prep(J2):
    H, diag = ref.build_H(J2)
    off = (H - sp.diags(diag)).tocoo()
    m = off.row != off.col
    return H, diag, off.row[m].astype(np.int64), off.col[m].astype(np.int64), off.data[m].astype(float)


def build_fn_vec(diag, rows, cols, h, a, s):
    """Vectorized copy of ref.build_fixed_node (verified equal below)."""
    af = np.maximum(np.asarray(a, float), 1e-15)
    k = s[rows].astype(float) * h * s[cols].astype(float)
    keep = k < 0
    d = diag.astype(float) + np.bincount(rows[~keep], weights=k[~keep] * af[cols[~keep]] / af[rows[~keep]],
                                         minlength=D)
    r = np.r_[rows[keep], np.arange(D)]
    c = np.r_[cols[keep], np.arange(D)]
    v = np.r_[k[keep], d]
    return sp.coo_matrix((v, (r, c)), shape=(D, D)).tocsr()


def fn_solve(diag, rows, cols, h, a, s, tol):
    F = build_fn_vec(diag, rows, cols, h, a, s)
    e, v = ref.ground(F, v0=np.asarray(a, float), tol=tol)
    v = np.abs(v)
    return e, v / np.linalg.norm(v)


def histogram_amplitude(a, phi, N, rng, rule):
    f = a * phi
    f = f / f.sum()
    cnt = rng.multinomial(int(N), f)
    vis = cnt > 0
    ph = np.zeros(D)
    ph[vis] = cnt[vis] / (N * a[vis])
    C = float(np.sum(ph[vis] * a[vis]) / np.sum(a[vis] ** 2))
    if rule == "cap":
        ph[~vis] = np.minimum(C * a[~vis], 0.5 / (N * a[~vis]))
    elif rule == "old":
        ph[~vis] = C * a[~vis]
    elif rule == "kt":
        ph = (cnt + 0.5) / (N * a)
    else:
        raise ValueError(rule)
    ph /= np.linalg.norm(ph)
    info = dict(n_unvisited=int(np.sum(~vis)), f_mass_unvisited=float(np.sum(f[~vis])))
    return ph, info


def run(N, seed, maxiter, rule, J2=0.5, J2init=0.0, tol=1e-10, verify=False):
    H, diag, rows, cols, h = prep(J2)
    E0, psi0 = ref.ground(H)
    sM = ref.canonical(ref.marshall_signs())
    if np.dot(psi0, sM) < 0:
        psi0 = -psi0
    a0 = np.abs(psi0); a0 /= np.linalg.norm(a0)
    strue = ref.canonical(np.where(psi0 >= 0, 1, -1).astype(np.int8))
    p0 = a0 * a0
    Hi, _ = ref.build_H(J2init)
    _, pin = ref.ground(Hi)
    a = np.abs(pin); a /= np.linalg.norm(a)
    s = sM.copy()

    def ws(sg):
        w = float(np.sum(p0[sg != strue]))
        return min(w, 1.0 - w)

    def eps(aa, sg):
        return (ref.physical_energy(H, aa, sg) - E0) / abs(E0)

    if verify:
        F1 = build_fn_vec(diag, rows, cols, h, a, s)
        F2 = ref.build_fixed_node(H, diag, a, s)
        assert abs(F1 - F2).max() < 1e-12, "vectorized FN builder mismatch"

    rng = np.random.default_rng(seed)
    hist = [dict(it=0, w_s=ws(s), eps=eps(a, s))]
    t0 = time.time()
    for it in range(1, maxiter + 1):
        efn, phi = fn_solve(diag, rows, cols, h, a, s, tol)
        if N > 0:
            amp, info = histogram_amplitude(a, phi, N, rng, rule)
        else:
            amp, info = phi, {}
        snew, T, eth, r, G = ref.projected_krylov_update(H, diag, amp, s)
        rec = dict(it=it, w_s=ws(snew), eps=eps(amp, snew), E_FN_of_guide=float(efn), T=float(T),
                   amp_fidelity=float(np.dot(amp, phi) ** 2), sign_hash=ref.sign_hash(snew), **info)
        if N > 0:
            # diagnostics: noise-free Krylov step from the same guide signs, and the
            # sampled signs dressed with the exact FN amplitude
            sx, *_ = ref.projected_krylov_update(H, diag, phi, s)
            rec["w_s_noisefree_step"] = ws(sx)
            rec["eps_sampled_signs_exact_phi"] = eps(phi, snew)
        hist.append(rec)
        a, s = amp, snew
        if it % 10 == 0 or it <= 3:
            print(f"N={N:g} seed={seed} it={it} w_s={rec['w_s']:.3e} eps={rec['eps']:.3e} "
                  f"Efn={efn:.8f} unvis={info.get('n_unvisited', 0)} t={time.time()-t0:.0f}s", flush=True)
    return dict(N=N, seed=seed, rule=rule, E0=float(E0), history=hist)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=float, required=True, help="i.i.d. samples per iteration; 0 = exact loop")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--maxiter", type=int, default=50)
    ap.add_argument("--rule", default="cap", choices=["cap", "old", "kt"])
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--outdir", default="/Users/aliotaifi/Chatty-organize/projects/j1j2/results/fn_krylov_loop_sampled_4x4")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    N = int(args.N)
    for sd in args.seeds:
        res = run(N, sd, args.maxiter, args.rule, tol=args.tol, verify=args.verify)
        res["method"] = __doc__
        tag = "exact" if N == 0 else f"N{N:.0e}".replace("+", "")
        fn = out / f"iid_{tag}_{args.rule}_seed{sd}.json"
        fn.write_text(json.dumps(res, indent=1))
        print("WROTE", fn, flush=True)
