#!/usr/bin/env python3
"""T1/T2 (4x4, J2/J1=0.5, exact scoring): second-order amplitude optimisers on the trained ViT (net H).

Wave function: complex translational ViT (nqsmagic, log psi = amplitude + i phase), warm-started from net H.
Modes
  --mode amp     : b = exp(Re log psi_theta) with an EXTERNAL sign s (fixed-sign VMC energy E_H[b] = <bs|H|bs>/<b|b>).
                   --signs net    : s = sign of net H (fixed-net-sign control)
                   --signs krylov : s <- exact energy-optimal current-sign Krylov step on the current b, applied at the
                                    start of every outer iteration (the loop; FN only as referee)
  --mode complex : the full complex psi_theta (sign AND amplitude learned jointly) -- standard VMC with the same optimiser
Optimisers (fresh tempered samples x ~ |psi|^{2 beta}, weights |psi|^{2-2beta}, exact i.i.d. sampler on the basis):
  rgntr  : RGN [Webber-Lindsey] (M + lam*Dm) d = -g by MINRES, M = 2(Hbar - E S) (edge form for amp, symmetrised
           non-symmetric estimator for complex); Dm = kept-edge stoquastic Laplacian A_+ (amp) or S (complex);
           Levenberg-Marquardt trust region verified on an independent fresh tempered sample (accept if measured
           paired decrease >= 0.25 * predicted; lam /= 3 if > 0.75; else lam *= 4, up to 6 tries).
  hybrid : (A_+ + eps) d = -g by CG, same verification (backtracking on step factor)
  sr     : (S + shift) d = -g by CG, step length from the quadratic model, same verification
  c1     : (A_F + eps) d = -g_F, frozen F = H_FN[b_k, s_k] at the start of the iteration, step factor 1, no gate
Jacobians are computed on the full basis (on 4x4 the one-hop neighbourhood of N >= 1e4 samples is the full basis).
CPU-h = (wall time excluding exact scoring) x cpus. Scoring (ED, E_FN of the guide) is diagnostics only.
"""
import argparse, json, time
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
from common import build_H, ground, canonical, marshall, guide_from_psi, BASIS, D, L, B, bits2x


def group_ids(r, atol=1e-11, rtol=1e-11):
    order = np.argsort(r, kind='mergesort'); rs = r[order]
    scale = np.maximum(np.maximum(np.abs(rs[1:]), np.abs(rs[:-1])), 1.0)
    br = np.empty(D, dtype=bool); br[0] = True; br[1:] = np.abs(rs[1:] - rs[:-1]) > (atol + rtol * scale)
    gs = np.cumsum(br, dtype=np.int32) - 1; gid = np.empty(D, dtype=np.int32); gid[order] = gs
    return gid, int(gs[-1]) + 1


def krylov_update(H, diag, ei, ej, hij, a, s):
    psi = a * s; r = np.asarray((H @ psi) / np.where(np.abs(psi) > 1e-300, psi, 1.0), float)
    gid, G = group_ids(r)
    wij = hij * a[ei] * a[ej]
    c0 = -s.copy(); val = 2.0 * wij * c0[ei] * c0[ej]
    e0 = float(np.sum(diag * a * a)) + float(np.sum(val))
    gi = gid[ei]; gj = gid[ej]; lo = np.minimum(gi, gj); hi = np.maximum(gi, gj); m = lo < hi
    delta = (np.bincount(lo[m], weights=-2.0 * val[m], minlength=G) + np.bincount(hi[m], weights=+2.0 * val[m], minlength=G))
    cand = np.r_[e0, e0 + np.cumsum(delta)]; kb = int(np.argmin(cand)) - 1
    sn = c0.copy()
    if kb >= 0: sn[gid <= kb] *= -1
    return canonical(sn)


def build_fn(diag, ei, ej, hij, a, s):
    af = np.maximum(a, 1e-15)
    rr = np.r_[ei, ej]; cc = np.r_[ej, ei]; hh = np.r_[hij, hij]
    kij = s[rr].astype(float) * hh * s[cc].astype(float)
    keep = kij < 0; bad = ~keep
    corr = np.bincount(rr[bad], weights=kij[bad] * af[cc[bad]] / af[rr[bad]], minlength=D)
    rows = np.r_[rr[keep], np.arange(D, dtype=np.int32)]; cols = np.r_[cc[keep], np.arange(D, dtype=np.int32)]
    vals = np.r_[kij[keep], diag + corr]
    return sp.coo_matrix((vals, (rows, cols)), shape=(D, D)).tocsr()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['amp', 'complex'], default='amp')
    ap.add_argument('--signs', choices=['net', 'krylov'], default='krylov')
    ap.add_argument('--opt', choices=['rgntr', 'hybrid', 'sr', 'c1'], default='rgntr')
    ap.add_argument('--N', type=int, default=10000)
    ap.add_argument('--beta', type=float, default=0.25)
    ap.add_argument('--steps-per-iter', type=int, default=6)
    ap.add_argument('--iters', type=int, default=1)
    ap.add_argument('--lam0', type=float, default=0.03)
    ap.add_argument('--eps', type=float, default=1e-6, help='relative identity shift')
    ap.add_argument('--maxiter', type=int, default=150)
    ap.add_argument('--max-cpu-h', type=float, default=1e9)
    ap.add_argument('--adapt', type=int, default=0, help='1: lam reset per outer iteration and capped at 30*lam0 per step; '
                    'N doubled (up to --Nmax) after a step whose 6 tries all fail verification (noise floor)')
    ap.add_argument('--Nmax', type=int, default=160000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--init', required=True)
    ap.add_argument('--layers', type=int, default=2); ap.add_argument('--d-model', type=int, default=32); ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--cpus', type=int, default=12)
    ap.add_argument('--chunk', type=int, default=1024); ap.add_argument('--jac-chunk', type=int, default=256)
    ap.add_argument('--tag', required=True); ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    t_start = time.time(); t_score = 0.0
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([args.seed, args.N, 977])

    H, diag, ei, ej, hij = build_H(0.5)
    E0, psi0 = ground(H, tol=1e-12)
    sM = canonical(marshall())
    if np.dot(psi0, sM) < 0: psi0 = -psi0
    strue = canonical(np.where(psi0 >= 0, 1, -1).astype(np.int8)); ptrue = psi0 ** 2
    Hoff = (H - sp.diags(H.diagonal())).tocsr(); Hd = H.diagonal()

    X = bits2x(BASIS)
    model = ViT(num_layers=args.layers, d_model=args.d_model, heads=args.heads, L_eff=4, b=B, transl_invariant=True, two_dimensional=True)
    p0 = model.init(jax.random.PRNGKey(0), jnp.asarray(X[:2]))["params"]
    _, unravel = ravel_pytree(p0)
    flat = jnp.asarray(np.load(args.init)['flat']); P = int(flat.size)

    def lp(z, x): return _logpsi_transl_2d(model.apply, B, {"params": unravel(z)}, x)
    f_chunk = jax.jit(lp)
    if args.mode == 'amp':
        jac_chunk = jax.jit(jax.vmap(jax.grad(lambda z, x: jnp.real(lp(z, x[None, :])[0])), in_axes=(None, 0)))
    else:
        def f2(z, x):
            o = lp(z, x[None, :])[0]; return jnp.stack([o.real, o.imag])
        jac_chunk = jax.jit(jax.vmap(jax.jacrev(f2), in_axes=(None, 0)))

    def logpsi_all(z):
        o = []
        for lo in range(0, D, args.chunk):
            xb = X[lo:lo + args.chunk]; n = len(xb)
            if n < args.chunk: xb = np.r_[xb, np.repeat(xb[:1], args.chunk - n, 0)]
            o.append(np.asarray(f_chunk(z, jnp.asarray(xb)))[:n])
        return np.concatenate(o)

    def jac_all(z):
        c = args.jac_chunk; outs = []
        for lo in range(0, D, c):
            xb = X[lo:lo + c]; n = len(xb)
            if n < c: xb = np.r_[xb, np.repeat(xb[:1], c - n, 0)]
            g = np.asarray(jac_chunk(z, jnp.asarray(xb)))[:n]
            outs.append(g.astype(np.float32) if args.mode == 'amp' else (g[:, 0] + 1j * g[:, 1]).astype(np.complex64))
        return np.concatenate(outs)

    # ---------------- state ----------------
    st0 = guide_from_psi(np.exp(logpsi_all(flat) - logpsi_all(flat).real.max()))
    s_net = st0[1]
    s = s_net.copy()

    def amp_of(lpsi):
        u = lpsi.real; b = np.exp(u - u.max()); return b / np.linalg.norm(b)

    def score(lpsi, s):
        t = time.time()
        if args.mode == 'amp':
            b = amp_of(lpsi); sg = s
        else:
            psi = np.exp(lpsi - lpsi.real.max()); b, sg, _ = guide_from_psi(psi)
        p = b * sg; Eg = float(p @ (H @ p))
        F = build_fn(diag, ei, ej, hij, b, sg); efn, _ = ground(F, v0=np.maximum(b, 1e-15), tol=2e-11)
        O = abs(float(np.sum(ptrue * sg * strue)))
        rec = dict(eps_guide=(Eg - E0) / abs(E0), eps_FN=(efn - E0) / abs(E0), w_s=max(0.0, (1 - O) / 2))
        if args.mode == 'complex':
            psi = np.exp(lpsi - lpsi.real.max()); E = float(np.real(np.vdot(psi, H @ psi)) / np.vdot(psi, psi).real)
            rec['eps_psi'] = (E - E0) / abs(E0)
        nonlocal t_score
        t_score += time.time() - t
        return rec

    def cpu_h(): return (time.time() - t_start - t_score) * args.cpus / 3600

    hist = []
    lpsi = logpsi_all(flat)
    rec0 = dict(iter=0, step=0, cpu_h=0.0, **score(lpsi, s)); hist.append(rec0)
    print('SETUP', json.dumps(dict(P=P, E0=E0, **vars(args))), flush=True)
    print('REC', json.dumps(rec0), flush=True)

    def draw(psi_abs2, n):
        q = psi_abs2 ** args.beta; p = q / q.sum()
        smp = rng.choice(D, size=n, p=p)
        U, C = np.unique(smp, return_counts=True)
        w = C * psi_abs2[U] / p[U]; return U, w / w.sum(), p

    def edge_lap(U, w, ratio, coef):
        """sparse D x D Laplacian sum_x w_x sum_y c_xy ratio_xy (e_x - e_y)(e_x - e_y)^T over H-neighbours y of x."""
        R = Hoff[U].tocoo(); xi = U[R.row]; yi = R.col
        ww = w[R.row] * coef(xi, yi, R.data) * ratio(xi, yi)
        m = ww != 0
        xi, yi, ww = xi[m], yi[m], ww[m]
        Lm = sp.coo_matrix((np.r_[ww, ww, -ww, -ww], (np.r_[xi, yi, xi, yi], np.r_[xi, yi, yi, xi])), shape=(D, D)).tocsr()
        return Lm

    lam = args.lam0; Ncur = args.N
    for it in range(1, args.iters + 1):
        if args.adapt: lam = args.lam0
        if args.mode == 'amp' and args.signs == 'krylov':
            s = krylov_update(H, diag, ei, ej, hij, amp_of(lpsi), s)
        sf = s.astype(float)
        Hs = sp.diags(sf) @ H @ sp.diags(sf); Hs = Hs.tocsr()
        if args.opt == 'c1':
            Ffro = build_fn(diag, ei, ej, hij, amp_of(lpsi), s)
        for stp in range(1, args.steps_per_iter + 1):
            if cpu_h() > args.max_cpu_h: break
            t0 = time.time()
            if args.adapt: lam = min(lam, 30 * args.lam0)
            J = jac_all(flat)
            if args.mode == 'amp':
                b = amp_of(lpsi); psi = b; absq = b * b; Hop = Hs
            else:
                psi = np.exp(lpsi - lpsi.real.max()); psi /= np.linalg.norm(psi); absq = np.abs(psi) ** 2; Hop = H
            Eloc = (Hop @ psi) / psi
            U, w, qp = draw(absq, Ncur)
            Uv, wv, _ = draw(absq, Ncur)
            Ebar = complex(np.sum(w * Eloc[U]))
            Ob = (w.astype(J.dtype) @ J[U])
            JU = J[U] - Ob[None, :]
            if args.mode == 'amp':
                g = 2.0 * (JU.T @ (w * (Eloc[U].real - Ebar.real)).astype(np.float32)).astype(np.float64)
            else:
                g = 2.0 * np.real(JU.conj().T @ (w * (Eloc[U] - Ebar)).astype(np.complex64)).astype(np.float64)

            def Smv(v):
                t = JU @ v.astype(np.float32)
                return np.real(JU.conj().T @ (w * t).astype(t.dtype)).astype(np.float64)

            if args.mode == 'amp':
                kept = lambda xi, yi, h: np.where(sf[xi] * h * sf[yi] < 0, np.abs(h), 0.0)
                signed = lambda xi, yi, h: -(sf[xi] * h * sf[yi])
                ratio = lambda xi, yi: b[yi] / b[xi]
                Lp = edge_lap(U, w, ratio, kept)
                LH = edge_lap(U, w, ratio, signed)
                cvec = w * (Eloc[U].real - Ebar.real)

                def Apmv(v):
                    t = J @ v.astype(np.float32); return (J.T @ (Lp @ t).astype(np.float32)).astype(np.float64)

                def Mmv(v):
                    t = J @ v.astype(np.float32); r1 = (J.T @ (LH @ t).astype(np.float32)).astype(np.float64)
                    tu = JU @ v.astype(np.float32)
                    return r1 + 2.0 * (JU.T @ (cvec * tu).astype(np.float32)).astype(np.float64)
                Dmv = Apmv
            else:
                Er = Ebar.real; wfull = np.zeros(D); wfull[U] = w; psic = psi.conj()

                def Mmv(v):
                    u = (J @ v.astype(np.float32)) - (Ob @ v.astype(np.float32))
                    Tu = (H @ (psi * u)) / psi - Er * u
                    m1 = 2.0 * np.real(JU.conj().T @ (w * Tu[U]).astype(np.complex64))
                    z = wfull * u
                    Thz = psic * (H @ (z / psic)) - Er * z
                    Jc = J.conj().T @ Thz.astype(np.complex64) - np.conj(Ob) * Thz.sum()
                    m2 = 2.0 * np.real(Jc)
                    return 0.5 * (m1 + m2).astype(np.float64)
                Dmv = Smv

            # scales for relative shifts (Hutchinson, 4 probes)
            pr = np.random.default_rng(stp).choice([-1.0, 1.0], size=(4, P))
            trM = abs(np.mean([z @ Mmv(z) for z in pr])); trD = max(np.mean([z @ Dmv(z) for z in pr]), 1e-300)
            trS = max(np.mean([z @ Smv(z) for z in pr[:2]]), 1e-300)
            sc = trM / P

            def energy_est(lpsi_new):
                """paired self-normalised energy of the candidate on the verification samples Uv (drawn for current psi)."""
                if args.mode == 'amp':
                    bn = amp_of(lpsi_new); pn = bn; el = (Hs @ bn)[Uv] / bn[Uv]
                    lw = 2 * (np.log(bn[Uv]) - np.log(b[Uv]))
                else:
                    pn = np.exp(lpsi_new - lpsi_new.real.max()); pn /= np.linalg.norm(pn)
                    el = ((H @ pn)[Uv] / pn[Uv]).real
                    lw = 2 * (np.log(np.abs(pn[Uv])) - np.log(np.abs(psi[Uv])))
                e = np.exp(lw - lw.max()) * wv
                return float(e @ el / e.sum())

            Ecur = energy_est(lpsi)
            info = dict(tries=0, accepted=False)
            if args.opt == 'c1':
                bq = amp_of(lpsi); elF = (Ffro @ bq) / bq; EF = float(np.sum(w * elF[U]))
                gF = 2.0 * (JU.T @ (w * (elF[U] - EF)).astype(np.float32)).astype(np.float64)
                A = sla.LinearOperator((P, P), matvec=lambda v: Apmv(v) + args.eps * (trD / P) * v, dtype=np.float64)
                d, _ = sla.cg(A, -gF, rtol=1e-6, atol=0.0, maxiter=args.maxiter)
                flat = flat + jnp.asarray(d); lpsi = logpsi_all(flat); info.update(accepted=True, tries=1)
            else:
                if args.opt == 'rgntr':
                    def solve(lam_):
                        A = sla.LinearOperator((P, P), matvec=lambda v: Mmv(v) + lam_ * (trM / trD) * Dmv(v) + args.eps * sc * v, dtype=np.float64)
                        d, _ = sla.minres(A, -g, rtol=1e-6, maxiter=args.maxiter); return d
                elif args.opt == 'hybrid':
                    A = sla.LinearOperator((P, P), matvec=lambda v: Apmv(v) + args.eps * (trD / P) * v, dtype=np.float64)
                    d0, _ = sla.cg(A, -g, rtol=1e-6, atol=0.0, maxiter=args.maxiter)
                    solve = lambda lam_: d0 * min(1.0, args.lam0 / lam_)
                else:  # sr
                    A = sla.LinearOperator((P, P), matvec=lambda v: Smv(v) + 1e-3 * (trS / P) * v, dtype=np.float64)
                    d0, _ = sla.cg(A, -g, rtol=1e-6, atol=0.0, maxiter=args.maxiter)
                    curv = float(d0 @ Mmv(d0)); alpha = -float(g @ d0) / curv if curv > 0 else 0.05
                    d0 = d0 * alpha
                    solve = lambda lam_: d0 * min(1.0, args.lam0 / lam_)
                for tr in range(6):
                    d = solve(lam); info['tries'] = tr + 1
                    pred = float(g @ d + 0.5 * d @ Mmv(d))
                    cand = flat + jnp.asarray(d); lc = logpsi_all(cand)
                    if not np.all(np.isfinite(lc)):
                        lam *= 4; continue
                    meas = energy_est(lc) - Ecur
                    if pred < 0 and meas < 0 and meas / pred > 0.25:
                        flat, lpsi = cand, lc; info.update(accepted=True, pred=pred, meas=meas)
                        if meas / pred > 0.75 and args.opt == 'rgntr': lam /= 3
                        break
                    lam *= 4
                if args.opt != 'rgntr':
                    lam = args.lam0
                if args.adapt and not info['accepted']:
                    Ncur = min(2 * Ncur, args.Nmax)
            rec = dict(iter=it, step=stp, cpu_h=cpu_h(), t_step=time.time() - t0, lam=lam, N=Ncur, n_unique=int(len(U)), **info)
            rec.update(score(lpsi, s if args.mode == 'amp' else None))
            hist.append(rec)
            print('REC', json.dumps(rec), flush=True)
            np.savez(out / f'{args.tag}_flat.npz', flat=np.asarray(flat), s=s)
            (out / f'{args.tag}.json').write_text(json.dumps(dict(args=vars(args), P=P, E0=E0, history=hist), indent=1))
        if cpu_h() > args.max_cpu_h: break
    print('DONE', json.dumps(hist[-1]), flush=True)


if __name__ == '__main__':
    main()
