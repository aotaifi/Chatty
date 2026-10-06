#!/usr/bin/env python3
"""Rung 2 loop (2026-10-03) started from a TRAINED-NETWORK guide (net_start_4x4, 2026-10-06): --start npz gives (a, s).

Exact 4x4 periodic J1-J2 (J2/J1=0.5), S^z=0, D=12870.  Closed FN/Krylov loop in which the
amplitude refresh is done by a NETWORK trained ONLY on finite i.i.d. samples:

  loop k: guide (a_k, s_k)
    1. F_k = lattice fixed-node H_FN[a_k, s_k] (exact sparse matrix; only its LOCAL rows at
       sampled configurations enter training).  E_FN_k, phi_k by exact diag = diagnostics only.
    2. --mode vmc (default, production): at EVERY SR step draw N fresh i.i.d. configurations from
       the exact |b_theta|^2 (perfect sampler), split train/val.  --mode reweight: one fixed set of
       N i.i.d. samples of the exact guide a_k^2 per loop iteration, reweighted by exp(2r) (pilot showed
       this is exploitable: the network raises b on unsampled neighbours, the exact frozen energy rises
       while train AND val estimates fall).
    3. train b_theta = a_k exp(r_theta), r_theta(x) = f_theta(x) - f_theta_start(x)
       (residual translational 2D ViT, 154,780 params, warm-started across loop iterations)
       to minimise the frozen-H_FN Rayleigh quotient  E[b] = <b|F_k|b>/<b|b>  estimated by
       self-normalised importance reweighting of the a_k^2 samples:
          E_hat = sum_i w_i E_loc(x_i) / sum_i w_i,   w_i = exp(2 r(x_i)),
          E_loc(x) = sum_y F_k[x,y] b(y)/b(x)   (network evaluated on x and its H-neighbours)
       SR: gradient g = 2 <(E_loc - E_hat)(O - <O>)>_w, Fisher S = <(O-<O>)(O-<O>)^T>_w with
       O = d r/d theta on the unique train configurations; (S + shift) d = g by CG;
       step length set by trust RMS_w(delta r) = trust; backtracking line search accepted only
       if the VALIDATION estimate E_hat_val decreases and the train ESS fraction stays above
       --ess-min.  Training stops at the first rejected step (early stopping) or --sr-steps.
    4. a_{k+1} = callable chain a_k exp(r_theta) (stored as its values on the basis).
    5. s_{k+1} = one current-sign Krylov step with the exact energy-optimal grouped threshold,
       evaluated with the LEARNED amplitude a_{k+1} (as closed_fn_krylov_exact4x4.py).
  Target ED (E0, true signs) is used ONLY for scoring.

Scores per loop iteration (guide after the update):
  w_s = sum_{x: s != s_true} psi0(x)^2 (global sign fixed), eps = (E[a s] - E0)/|E0|,
  plus E_FN_k (diagnostic), frozen gain fraction and phi fidelity of the learned refresh, and
  the counterfactual score if the exact phi_k had been used for this single step.
"""
import argparse, json, time, hashlib, os
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

L = 4; N = 16; B = 2


def red(x, y):
    return (x % L) + L * (y % L)


NN = []; NNN = []
for yy in range(L):
    for xx in range(L):
        i = red(xx, yy)
        NN += [(i, red(xx + 1, yy)), (i, red(xx, yy + 1))]
        NNN += [(i, red(xx + 1, yy + 1)), (i, red(xx + 1, yy - 1))]
MASK = 0
for yy in range(L):
    for xx in range(L):
        if (xx + yy) % 2 == 0:
            MASK |= 1 << red(xx, yy)

BASIS = np.array([s for s in range(1 << N) if s.bit_count() == N // 2], dtype=np.uint32)
D = len(BASIS)
S2I = np.full(1 << N, -1, dtype=np.int32); S2I[BASIS] = np.arange(D, dtype=np.int32)


def build_H(J2):
    diag = np.zeros(D); rr = []; cc = []; vv = []; rows = np.arange(D, dtype=np.int32)
    for bonds, J in ((NN, 1.0), (NNN, J2)):
        for u, v in bonds:
            bu = (BASIS >> u) & 1; bv = (BASIS >> v) & 1; same = bu == bv
            diag += J * np.where(same, .25, -.25)
            sel = ~same; r = rows[sel]
            t = BASIS[sel] ^ (np.uint32(1 << u) | np.uint32(1 << v))
            rr.append(r); cc.append(S2I[t]); vv.append(np.full(r.size, .5 * J))
    rr = np.concatenate(rr); cc = np.concatenate(cc); vv = np.concatenate(vv)
    H = sp.coo_matrix((np.r_[vv, diag], (np.r_[rr, rows], np.r_[cc, rows])), shape=(D, D)).tocsr()
    up = sp.triu(H - sp.diags(diag), k=1).tocoo()
    return H, diag, up.row.astype(np.int32), up.col.astype(np.int32), up.data.astype(float)


def marshall():
    return np.array([1 if (int(s) & MASK).bit_count() % 2 == 0 else -1 for s in BASIS], dtype=np.int8)


def ground(A, v0=None, tol=2e-11):
    ew, ev = sla.eigsh(A, k=1, which='SA', v0=v0, tol=tol, maxiter=300000)
    v = np.asarray(ev[:, 0], float); v /= np.linalg.norm(v)
    return float(ew[0]), v


def canonical(s):
    s = np.asarray(s, dtype=np.int8).copy()
    if s[0] < 0: s = -s
    return s


def sign_hash(s):
    return hashlib.sha1(np.packbits((s > 0).astype(np.uint8)).tobytes()).hexdigest()[:12]


def group_ids(r, atol=1e-11, rtol=1e-11):
    order = np.argsort(r, kind='mergesort'); rs = r[order]
    scale = np.maximum(np.maximum(np.abs(rs[1:]), np.abs(rs[:-1])), 1.0)
    br = np.empty(D, dtype=bool); br[0] = True; br[1:] = np.abs(rs[1:] - rs[:-1]) > (atol + rtol * scale)
    gs = np.cumsum(br, dtype=np.int32) - 1; gid = np.empty(D, dtype=np.int32); gid[order] = gs
    return gid, int(gs[-1]) + 1


def krylov_update(H, diag, ei, ej, hij, a, s):
    # current-sign projected Krylov, exact energy-optimal grouped threshold (no target data)
    psi = a * s; r = np.asarray((H @ psi) / np.where(np.abs(psi) > 1e-300, psi, 1.0), float)
    gid, G = group_ids(r)
    wij = hij * a[ei] * a[ej]
    c0 = -s.copy(); val = 2.0 * wij * c0[ei] * c0[ej]
    e0 = float(np.sum(diag * a * a)) + float(np.sum(val))
    gi = gid[ei]; gj = gid[ej]; lo = np.minimum(gi, gj); hi = np.maximum(gi, gj); m = lo < hi
    delta = (np.bincount(lo[m], weights=-2.0 * val[m], minlength=G) +
             np.bincount(hi[m], weights=+2.0 * val[m], minlength=G))
    cand = np.r_[e0, e0 + np.cumsum(delta)]; kb = int(np.argmin(cand)) - 1
    sn = c0.copy()
    if kb >= 0: sn[gid <= kb] *= -1
    return canonical(sn), G


def build_fn(diag, ei, ej, hij, a, s):
    af = np.maximum(a, 1e-15)
    rr = np.r_[ei, ej]; cc = np.r_[ej, ei]; hh = np.r_[hij, hij]
    kij = s[rr].astype(float) * hh * s[cc].astype(float)
    keep = kij < 0; bad = ~keep
    corr = np.bincount(rr[bad], weights=kij[bad] * af[cc[bad]] / af[rr[bad]], minlength=D)
    rows = np.r_[rr[keep], np.arange(D, dtype=np.int32)]
    cols = np.r_[cc[keep], np.arange(D, dtype=np.int32)]
    vals = np.r_[kij[keep], diag + corr]
    return sp.coo_matrix((vals, (rows, cols)), shape=(D, D)).tocsr()


def bits2x(ss):
    a = np.asarray(ss, np.uint32).reshape(-1, 1)
    return 2.0 * (((a >> np.arange(N, dtype=np.uint32)) & 1).astype(np.float64)) - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, required=True,
                    help='reweight: i.i.d. samples of a_k^2 per loop iteration; vmc: i.i.d. samples of b_theta^2 per SR step (train+val)')
    ap.add_argument('--mode', choices=['reweight', 'vmc'], default='vmc')
    ap.add_argument('--val-frac', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--loop-iters', type=int, default=30)
    ap.add_argument('--sr-steps', type=int, default=30)
    ap.add_argument('--shift', type=float, default=0.1)
    ap.add_argument('--trust', type=float, default=0.01)
    ap.add_argument('--ess-min', type=float, default=0.5)
    ap.add_argument('--cg-tol', type=float, default=3e-5)
    ap.add_argument('--cg-maxiter', type=int, default=200)
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--jac-chunk', type=int, default=128)
    ap.add_argument('--net-seed', type=int, default=None)
    ap.add_argument('--jac-dtype', choices=['float32', 'float64'], default='float32')
    ap.add_argument('--trust-adapt', type=int, default=1, help='1: after a step accepted at factor f<1, trust*=f for the rest of this loop iteration')
    ap.add_argument('--start', required=True, help='npz with a (amplitude on basis) and s (sign on basis) of the trained-network guide')
    ap.add_argument('--cpus', type=int, default=16)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    t_start = time.time()
    net_seed = args.net_seed if args.net_seed is not None else 20261004 + args.seed

    H, diag, ei, ej, hij = build_H(0.5)
    H0, _, _, _, _ = build_H(0.0)
    sM = canonical(marshall())
    E0, psi0 = ground(H, tol=1e-12)                   # scoring only
    if np.dot(psi0, sM) < 0: psi0 = -psi0
    strue = canonical(np.where(psi0 >= 0, 1, -1).astype(np.int8)); ptrue = psi0 ** 2
    atrue = np.abs(psi0)
    st = np.load(args.start)
    a = np.asarray(st['a'], float); a /= np.linalg.norm(a); s = canonical(st['s'])   # trained-network guide

    def score(a, s):
        E = float((a * s) @ (H @ (a * s)) / (a @ a)); O = abs(float(np.sum(ptrue * s * strue)))
        return dict(E=E, eps=(E - E0) / abs(E0), w_s=max(0.0, (1 - O) / 2),
                    F_amp=float(np.dot(a, atrue) ** 2 / (a @ a)), sign_hash=sign_hash(s))

    X = bits2x(BASIS)
    def walltag(): return (time.time() - t_start) * args.cpus / 3600
    model = ViT(num_layers=4, d_model=60, heads=10, L_eff=4, b=2, transl_invariant=True, two_dimensional=True)
    p0 = model.init(jax.random.PRNGKey(net_seed), jnp.asarray(X[:2]))["params"]
    flat, unravel = ravel_pytree(p0)
    npar = int(flat.size)

    def raw_f(z, x):
        return jnp.real(_logpsi_transl_2d(model.apply, B, {"params": unravel(z)}, x))

    f_chunk = jax.jit(raw_f)
    grad_chunk = jax.jit(jax.vmap(jax.grad(lambda z, x: raw_f(z, x[None, :])[0]), in_axes=(None, 0)))

    def padded(idx, c):
        # fixed chunk shapes (pad with index 0) -> no jit recompilation when sample sets change
        idx = np.asarray(idx); n = len(idx); npad = (-n) % c
        return np.r_[idx, np.zeros(npad, dtype=idx.dtype)] if npad else idx, n

    def eval_f(z, idx):
        pidx, n = padded(idx, args.chunk)
        out = [np.asarray(f_chunk(z, jnp.asarray(X[pidx[lo:lo + args.chunk]]))) for lo in range(0, len(pidx), args.chunk)]
        return np.concatenate(out)[:n] if out else np.zeros(0)

    def jac(z, idx):
        pidx, n = padded(idx, args.jac_chunk)
        O = np.empty((n, npar), dtype=np.dtype(args.jac_dtype))
        for lo in range(0, len(pidx), args.jac_chunk):
            g = np.asarray(grad_chunk(z, jnp.asarray(X[pidx[lo:lo + args.jac_chunk]])))
            hi = min(lo + args.jac_chunk, n)
            O[lo:hi] = g[:hi - lo]
        return O

    ALL = np.arange(D)
    F0 = build_fn(diag, ei, ej, hij, a, s); efn0, _ = ground(F0, v0=np.maximum(a, 1e-15), tol=2e-11)
    hist = [dict(it=0, cpu_h=0.0, **score(a, s), E_FN=efn0, eps_FN=(efn0 - E0) / abs(E0))]
    print('SETUP', json.dumps(dict(D=D, npar=npar, E0=E0, budget=args.budget, seed=args.seed,
                                   net_seed=net_seed)), flush=True)
    print('ITER 0', json.dumps(hist[-1]), flush=True)

    meta = dict(script=os.path.basename(__file__), args=vars(args), npar=npar, E0=E0, D=D,
                objective='frozen-H_FN Rayleigh quotient of b=a_k exp(r_theta)',
                estimator=('vmc: every SR step draws --budget fresh i.i.d. samples from the exact |b_theta|^2 '
                           '(exact sampler = perfectly mixed VMC), half train / half val; E_hat = mean E_loc; '
                           'line-search candidates scored on the val half reweighted by exp(2 delta r)'
                           if args.mode == 'vmc' else
                           'reweight: one fixed set of --budget i.i.d. samples of the exact a_k^2 per loop iteration; '
                           'E_hat = sum w E_loc / sum w, w = exp(2 r)') +
                          '; E_loc(x)=sum_y F[x,y] b(y)/b(x) from local rows of H_FN[a_k,s_k]; SR gradient '
                          '2<(E_loc-E)(O-<O>)>_w, Fisher <(O-<O>)(O-<O>)^T>_w on unique train configs (CG); '
                          'step accepted only if the held-out val estimate decreases (early stop at first reject)',
                network='residual translational 2D ViT (4 layers, d=60, 10 heads, b=2), warm-started across loop',
                oracle_policy='ED used only to draw samples from a_k^2 (exact), for diagnostics E_FN/phi, and scoring')

    def dump(terminal):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(meta=meta, terminal=terminal, elapsed_sec=time.time() - t_start,
                                                  history=hist), indent=1))

    for k in range(args.loop_iters):
        tt = time.time()
        sold = s.copy()
        F = build_fn(diag, ei, ej, hij, a, s)
        efn, phi = ground(F, v0=np.maximum(a, 1e-15), tol=2e-11)   # diagnostic only
        phi = np.abs(phi); phi /= np.linalg.norm(phi)
        Ef_base = float(a @ (F @ a) / (a @ a))
        s_cf, _ = krylov_update(H, diag, ei, ej, hij, phi, sold)
        cf = score(phi, s_cf)

        loga = np.log(np.maximum(a, 1e-300))
        f0_all = eval_f(flat, ALL)                     # reference of the residual (r=0 at start)
        nsamp_used = 0

        def draw(logamp, n, tag):
            # n i.i.d. configurations from the exact distribution ∝ exp(2 logamp)
            rng = np.random.default_rng([args.seed, k, tag, 777])
            pp = np.exp(2.0 * (logamp - logamp.max())); pp /= pp.sum()
            smp = rng.choice(D, size=n, p=pp)
            nva = int(round(args.val_frac * n)); ntr = n - nva
            utr, ctr = np.unique(smp[:ntr], return_counts=True)
            uva, cva = np.unique(smp[ntr:], return_counts=True)
            return utr, ctr, F[utr], uva, cva, F[uva]

        def est_set(U, C, Fr, lb_c, lb_s):
            # self-normalised IS estimate of <b|F|b>/<b|b> for b=exp(lb_c) from samples of exp(2 lb_s)
            bvec = np.exp(lb_c - np.max(lb_c[np.isfinite(lb_c)]))   # -inf (never evaluated) -> 0
            el = (Fr @ bvec) / bvec[U]
            lw = 2.0 * (lb_c[U] - lb_s[U]); e = np.exp(lw - lw.max())
            w = C * e; w = w / w.sum()
            ess = (np.sum(C * e) ** 2) / np.sum(C * e * e) / C.sum()
            return float(np.sum(w * el)), w, el, float(ess)

        if args.mode == 'reweight':
            # one fixed sample set from the exact guide distribution a_k^2 per loop iteration
            utr, ctr, Ftr, uva, cva, Fva = draw(loga, args.budget, 0)
            nsamp_used = args.budget
            evalset = np.unique(np.r_[utr, uva, Ftr.indices, Fva.indices])
            lb_samp = loga
        else:
            evalset = ALL

        def lb_of(f_eval):
            lb = np.full(D, -np.inf); lb[evalset] = loga[evalset] + f_eval - f0_all[evalset]
            return lb

        lb_cur = lb_of(f0_all[evalset])
        E_tr0 = E_va0 = None
        rows = []; nacc = 0; stop = 'max_steps'; trust = args.trust
        for st in range(1, args.sr_steps + 1):
            t0 = time.time()
            if args.mode == 'vmc':
                # fresh i.i.d. samples from the exact distribution of the CURRENT network amplitude b_theta^2
                utr, ctr, Ftr, uva, cva, Fva = draw(lb_cur, args.budget, st)
                nsamp_used += args.budget
                lb_samp = lb_cur
            Etr, w, el, ess_tr = est_set(utr, ctr, Ftr, lb_cur, lb_samp)
            Eva = est_set(uva, cva, Fva, lb_cur, lb_samp)[0]
            if E_tr0 is None: E_tr0, E_va0 = Etr, Eva
            O = jac(flat, utr)
            t1 = time.time()
            Ob = (w.astype(O.dtype)) @ O
            O -= Ob[None, :]
            O *= np.sqrt(w).astype(O.dtype)[:, None]
            epsv = 2.0 * np.sqrt(w) * (el - Etr)
            g = (O.T @ epsv.astype(O.dtype)).astype(np.float64)
            def mv(v):
                vv = np.asarray(v).astype(O.dtype)
                return (O.T @ (O @ vv)).astype(np.float64) + args.shift * np.asarray(v)
            Aop = sla.LinearOperator((npar, npar), matvec=mv, dtype=np.float64)
            sol, info = sla.cg(Aop, g, rtol=args.cg_tol, atol=0.0, maxiter=args.cg_maxiter)
            d = -sol
            rms = float(np.linalg.norm((O @ d.astype(O.dtype)).astype(np.float64)))
            del O
            t2 = time.time()
            eta = trust / max(rms, 1e-300)
            acc = None
            for fac in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
                cand = flat + jnp.asarray(eta * fac * d)
                lb_c = lb_of(eval_f(cand, evalset))
                eva_c = est_set(uva, cva, Fva, lb_c, lb_samp)[0]
                etr_c = est_set(utr, ctr, Ftr, lb_c, lb_samp)
                if np.isfinite(eva_c) and eva_c < Eva and etr_c[3] >= args.ess_min:
                    acc = (cand, lb_c, eva_c, etr_c, fac); break
            row = dict(step=st, E_tr=Etr, E_va=Eva, ess_tr=ess_tr, n_unique_tr=int(len(utr)), cg_info=int(info),
                       rms_per_eta=rms, trust=trust, t_jac=t1 - t0, t_cg=t2 - t1, t_ls=time.time() - t2)
            if acc is None:
                row['accepted'] = False; rows.append(row); stop = 'val_reject'; break
            flat, lb_cur, eva_c, etr_c, fac = acc
            nacc += 1
            if args.trust_adapt and fac < 1.0:
                trust *= fac
            row.update(accepted=True, factor=fac, E_tr_new=etr_c[0], E_va_new=eva_c, ess_tr_new=etr_c[3])
            rows.append(row)
        last = rows[-1]
        E_tr_end = last.get('E_tr_new', last['E_tr']); E_va_end = last.get('E_va_new', last['E_va'])
        ess_end = last.get('ess_tr_new', last['ess_tr']); needed = evalset

        # ---- new callable amplitude (chain a_k exp r) on the basis ----
        r_all = eval_f(flat, ALL) - f0_all
        lb = loga + r_all; an = np.exp(lb - lb.max()); an /= np.linalg.norm(an)
        Ef_new = float(an @ (F @ an))
        gainfrac = (Ef_base - Ef_new) / max(Ef_base - efn, 1e-300)
        wphi = phi * phi
        lr = np.log(np.maximum(an, 1e-300)) - np.log(np.maximum(phi, 1e-300)); lr -= np.sum(wphi * lr)
        lr0 = loga - np.log(np.maximum(phi, 1e-300)); lr0 -= np.sum(wphi * lr0)
        snew, G = krylov_update(H, diag, ei, ej, hij, an, sold)
        a = an; s = snew
        rec = dict(it=k + 1, cpu_h=walltag(), **score(a, s), E_FN=efn, eps_FN=(efn - E0) / abs(E0),
                   frozen_E_base=Ef_base, frozen_E_learned=Ef_new, frozen_gain_frac=float(gainfrac),
                   frozen_E_tr_start=E_tr0, frozen_E_va_start=E_va0,
                   frozen_E_tr_end=E_tr_end, frozen_E_va_end=E_va_end, ess_tr_end=ess_end,
                   samples_used=int(nsamp_used), mode=args.mode,
                   phi_fidelity=float((an @ phi) ** 2), phi_logratio_rms=float(np.sqrt(np.sum(wphi * lr * lr))),
                   phi_logratio_rms_before=float(np.sqrt(np.sum(wphi * lr0 * lr0))),
                   n_unique_tr=int(len(utr)), n_unique_va=int(len(uva)), n_needed=int(len(needed)),
                   sr_accepted=nacc, sr_stop=stop, n_sign_flips=int(np.sum(s != sold)), r_groups=G,
                   cf_exactphi_w_s=cf['w_s'], cf_exactphi_eps=cf['eps'], sec=time.time() - tt,
                   sr_rows=rows)
        hist.append(rec)
        print('ITER', k + 1, json.dumps({kk: v for kk, v in rec.items() if kk != 'sr_rows'}), flush=True)
        dump('running')
    dump('maxiter')
    print('DONE', args.out, 'elapsed', time.time() - t_start, flush=True)


if __name__ == '__main__':
    main()
