#!/usr/bin/env python3
"""(c) 4x4: one Krylov step from Marshall with noisy amplitudes a = |psi0| exp(sigma xi), xi~N(0,1) iid per configuration.
Decision variables z(x) (s' = s sgn[T - z], T energy-optimal, label-free):
  r      : plain local energy (H a s)/(a s)
  delta  : r - r_FN, r_FN = (H_FN a s)/(a s) with H_FN[a,s] built as specified (K<0 kept, K>0 -> diagonal * a_y/a_x)
           (algebraically r_FN == r; max|delta| is recorded to verify)
  dV     : r - r_kept = V_sf(x) = sum_{y:K_xy>0} K_xy a_y/a_x   (kept hops only, no sign-flip potential)
  dA     : r - r_abs  = 2 V_sf   (absolute-value/stoquastic baseline; same ordering as dV, recorded as check)
Metrics vs ED: w_s, eps=(E(a,s')-E0)/|E0|, eps_sign = eps - eps(a, exact signs) (sign-attributable part),
and w_s_bestT = best w_s over ALL thresholds T (ranking quality, uses labels: diagnostic only).
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.environ.get("SSL_DIR", str(Path(__file__).resolve().parents[1] / "stored_signs")))
import stored_sign_loop_4x4 as S

ap = argparse.ArgumentParser()
ap.add_argument('--sigmas', default='0,0.02,0.05,0.1,0.2,0.3')
ap.add_argument('--nseed', type=int, default=20)
ap.add_argument('--noise', choices=['iid', 'smooth'], default='iid')
ap.add_argument('--out', required=True)
args = ap.parse_args()
H, diag, ei, ej, hij = S.build_H(0.5); D = S.D
E0, psi0 = S.ground(H, tol=1e-12)
strue = np.where(psi0 >= 0, 1, -1).astype(np.int8); ptrue = psi0 ** 2; a0 = np.abs(psi0)
M = S.marshall()
if np.sum(ptrue * M * strue) < 0: strue = -strue


def thr_step(a, s, z):
    """energy-optimal grouped-threshold step with decision variable z (copy of stored_sign_loop_4x4.krylov, r -> z)."""
    gid, G = S.group_ids(z)
    wij = hij * a[ei] * a[ej]
    c0 = -s.astype(np.int8); val = 2.0 * wij * c0[ei] * c0[ej]
    e0 = float(np.sum(diag * a * a)) + float(np.sum(val))
    gi = gid[ei]; gj = gid[ej]; lo = np.minimum(gi, gj); hi = np.maximum(gi, gj); m = lo < hi
    delta = (np.bincount(lo[m], weights=-2.0 * val[m], minlength=G) + np.bincount(hi[m], weights=+2.0 * val[m], minlength=G))
    cand = np.r_[e0, e0 + np.cumsum(delta)]; kb = int(np.argmin(cand)) - 1
    sn = c0.copy()
    if kb >= 0: sn[gid <= kb] *= -1
    if np.sum(a * a * sn * s) < 0: sn = -sn
    return sn.astype(np.int8)


def best_ws_over_T(s, z):
    b = (s * strue).astype(float)   # +1 agree
    order = np.argsort(z, kind='mergesort')
    # T below all z: s' = -s -> agree weight = -b ; T raising flips... s' = s*sgn(T-z): z<T keeps s, z>T flips
    # wrong(T) = sum_{z<T}[b=-1] p + sum_{z>T}[b=+1] p
    p = ptrue[order]; bb = b[order]
    wr_keep = np.r_[0.0, np.cumsum(p * (bb < 0))]
    wr_flip = np.r_[np.cumsum((p * (bb > 0))[::-1])[::-1], 0.0]
    W = wr_keep + wr_flip
    return float(np.min(np.minimum(W, 1 - W)))


def variables(a, s):
    kij = s[np.r_[ei, ej]].astype(float) * np.r_[hij, hij] * s[np.r_[ej, ei]].astype(float)
    rr = np.r_[ei, ej]; cc = np.r_[ej, ei]
    psi = a * s; r = np.asarray((H @ psi) / psi, float)
    pos = kij > 0
    Vsf = np.bincount(rr[pos], weights=kij[pos] * a[cc[pos]] / a[rr[pos]], minlength=D)
    # r_FN = (H_FN a)/a built explicitly
    d = diag + Vsf
    keep = ~pos
    HFa = d * a + np.bincount(rr[keep], weights=kij[keep] * a[cc[keep]], minlength=D)
    rFN = HFa / a
    rabs = diag - np.bincount(rr, weights=np.abs(kij) * a[cc] / a[rr], minlength=D)
    return dict(r=r, delta=r - rFN, dV=r - (r - Vsf), dA=r - rabs), float(np.max(np.abs(r - rFN)))


def score(a, s):
    E = S.physical_energy(H, a, s); O = abs(float(np.sum(ptrue * s * strue)))
    return (E - E0) / abs(E0), max(0.0, (1 - O) / 2)


spins = 2 * ((S.basis[:, None] >> np.arange(S.N)[None, :]) & 1).astype(float) - 1
iu = np.triu_indices(S.N, 1); PAIR = spins[:, iu[0]] * spins[:, iu[1]]   # (D, 120) pair correlators


def noise(rng):
    if args.noise == 'iid':
        return rng.standard_normal(D)
    xi = PAIR @ rng.standard_normal(PAIR.shape[1])                        # smooth random pair-correlation field
    xi -= np.sum(ptrue * xi); return xi / np.sqrt(np.sum(ptrue * xi ** 2))   # unit std under |psi0|^2


res = []
for sig in [float(x) for x in args.sigmas.split(',')]:
    for sd in range(args.nseed if sig > 0 else 1):
        rng = np.random.default_rng(1000 + sd)
        a = a0 * np.exp(sig * noise(rng)); a /= np.linalg.norm(a)
        e_ex, _ = score(a, strue)
        e_M, w_M = score(a, M)
        vs, maxdelta = variables(a, M)
        rec = dict(sigma=sig, seed=sd, eps_exact_signs=e_ex, eps_marshall=e_M, w_s_marshall=w_M, max_abs_delta_spec=maxdelta,
                   r_std=float(np.std(vs['r'])), dV_std=float(np.std(vs['dV'])))
        for k, z in vs.items():
            sn = thr_step(a, M, z); e, w = score(a, sn)
            rec[k] = dict(eps=e, eps_sign=e - e_ex, w_s=w, w_s_bestT=best_ws_over_T(M, z), nflip=int(np.sum(sn != M)))
        res.append(rec)
    print(sig, 'done', {k: np.mean([r[k]['w_s'] for r in res if r['sigma'] == sig]) for k in ('r', 'delta', 'dV', 'dA')}, flush=True)
Path(args.out).write_text(json.dumps(dict(E0=E0, rows=res), indent=1))
