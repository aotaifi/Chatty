#!/usr/bin/env python3
"""Accelerated amplitude fixed point for the closed exact FN/Krylov loop.

Plain loop (closed_fn_krylov_exact4x4.py / closed_fn_krylov_exact20.py):
    a_FN = Perron GS of H_FN[a_k, s_k]
    s_{k+1} = energy-optimal one-step current-sign Krylov update(a_FN, s_k)
    a_{k+1} = a_FN
Here the amplitude update is accelerated in log space, x = log a (normalized,
||exp x||_2 = 1), g(x) = log a_FN[exp x, s]:
    plain        : x_{k+1} = g_k
    overrelax w  : x_{k+1} = x_k + w (g_k - x_k)
    anderson m,b : type-II Anderson (Walker-Ni) with memory m, mixing b
Safeguard (accelerated methods): if E_FN(a_k,s_k) > E_FN(a_{k-1},s_{k-1}) (+tiny tol),
or the proposal is non-finite / makes a jump > --max-jump in log space, the history
is cleared and the plain step x_{k+1} = g_k is taken from the current point.
Sign update: s_{k+1} = Krylov(a_{k+1}, s_k) on the accelerated amplitude
(--krylov-amp accel, default) or on a_FN as in the plain loop (--krylov-amp fn).
Target ED ground state is used ONLY for scoring (w_s, eps, F_amp); never in updates.
"""
import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def make_lattice(name):
    if name == "4x4":
        LX, LY = 4, 4
        def red(x, y):
            return (x % LX) + LX * (y % LY)
        geom = "4x4 periodic"
    elif name == "20":
        LX, LY = 4, 5
        def red(x, y):
            q = y // LY; y -= q * LY; x -= q
            return (x % LX) + LX * y
        geom = "20-site tilted torus T1=(4,0), T2=(1,5)"
    else:
        raise ValueError(name)
    N = LX * LY
    NN = []; NNN = []
    for y in range(LY):
        for x in range(LX):
            i = red(x, y)
            NN += [(i, red(x + 1, y)), (i, red(x, y + 1))]
            NNN += [(i, red(x + 1, y + 1)), (i, red(x + 1, y - 1))]
    mask = 0
    for y in range(LY):
        for x in range(LX):
            if (x + y) % 2 == 0:
                mask |= 1 << red(x, y)
    return N, NN, NNN, mask, geom


class Model:
    def __init__(self, lattice):
        self.N, self.NN, self.NNN, self.mask, self.geom = make_lattice(lattice)
        N = self.N
        self.basis = np.array([s for s in range(1 << N) if s.bit_count() == N // 2], dtype=np.uint32)
        self.D = len(self.basis)
        self.state_to_idx = np.full(1 << N, -1, dtype=np.int32)
        self.state_to_idx[self.basis] = np.arange(self.D, dtype=np.int32)

    def build_H(self, J2):
        D = self.D; basis = self.basis
        diag = np.zeros(D, float); rr = []; cc = []; vv = []; rows = np.arange(D, dtype=np.int32)
        for bonds, J in ((self.NN, 1.0), (self.NNN, J2)):
            for u, v in bonds:
                bu = (basis >> u) & 1; bv = (basis >> v) & 1; same = bu == bv
                diag += J * np.where(same, .25, -.25)
                sel = ~same; r = rows[sel]
                t = basis[sel] ^ (np.uint32(1 << u) | np.uint32(1 << v))
                rr.append(r); cc.append(self.state_to_idx[t]); vv.append(np.full(r.size, .5 * J))
        rr = np.concatenate(rr); cc = np.concatenate(cc); vv = np.concatenate(vv)
        H = sp.coo_matrix((np.r_[vv, diag], (np.r_[rr, rows], np.r_[cc, rows])), shape=(D, D)).tocsr()
        up = sp.triu(H - sp.diags(diag), k=1).tocoo()
        return H, diag, up.row.astype(np.int32), up.col.astype(np.int32), up.data.astype(float)

    def marshall_signs(self):
        m = self.mask
        return np.array([1 if (int(s) & m).bit_count() % 2 == 0 else -1 for s in self.basis], dtype=np.int8)


def ground(A, v0=None, tol=2e-11):
    ew, ev = sla.eigsh(A, k=1, which='SA', v0=v0, tol=tol, maxiter=300000)
    v = np.asarray(ev[:, 0], float); v /= np.linalg.norm(v)
    return float(ew[0]), v


def canonical(s):
    s = np.asarray(s, dtype=np.int8).copy()
    if s[0] < 0: s = -s
    return s


def physical_energy(H, a, s):
    psi = a * s
    return float(psi @ (H @ psi) / (psi @ psi))


def sign_hash(s):
    return hashlib.sha1(np.packbits((s > 0).astype(np.uint8)).tobytes()).hexdigest()[:12]


def group_ids(r, atol=1e-11, rtol=1e-11):
    D = len(r)
    order = np.argsort(r, kind='mergesort'); rs = r[order]
    scale = np.maximum(np.maximum(np.abs(rs[1:]), np.abs(rs[:-1])), 1.0)
    br = np.empty(D, dtype=bool); br[0] = True; br[1:] = np.abs(rs[1:] - rs[:-1]) > (atol + rtol * scale)
    gs = np.cumsum(br, dtype=np.int32) - 1; gid = np.empty(D, dtype=np.int32); gid[order] = gs
    return gid, int(gs[-1]) + 1


def projected_krylov_update(H, diag, ei, ej, hij, a, s):
    # IMPORTANT: no target ground-state data enters this function.
    psi = a * s; r = np.asarray((H @ psi) / np.where(np.abs(psi) > 1e-300, psi, 1.0), float)
    gid, G = group_ids(r)
    wij = hij * a[ei] * a[ej]
    c0 = -s.copy(); val = 2.0 * wij * c0[ei] * c0[ej]
    diagE = float(np.sum(diag * a * a)); e0 = diagE + float(np.sum(val))
    gi = gid[ei]; gj = gid[ej]; lo = np.minimum(gi, gj); hi = np.maximum(gi, gj); mask = lo < hi
    delta = (np.bincount(lo[mask], weights=-2.0 * val[mask], minlength=G) +
             np.bincount(hi[mask], weights=+2.0 * val[mask], minlength=G))
    candidates = np.r_[e0, e0 + np.cumsum(delta)]; kbest = int(np.argmin(candidates)) - 1
    sn = c0.copy()
    if kbest >= 0: sn[gid <= kbest] *= -1
    return canonical(sn), float(np.min(candidates)), G


def fixed_node_solve(H, diag, ei, ej, hij, a, s, tol):
    # IMPORTANT: no target ground-state data enters this function.
    D = len(a)
    af = np.maximum(a, 1e-15)
    rr = np.r_[ei, ej]; cc = np.r_[ej, ei]; hh = np.r_[hij, hij]
    kij = s[rr].astype(float) * hh * s[cc].astype(float)
    keep = kij < 0; bad = ~keep
    corr = np.bincount(rr[bad], weights=kij[bad] * af[cc[bad]] / af[rr[bad]], minlength=D)
    d = diag + corr
    rows = np.r_[rr[keep], np.arange(D, dtype=np.int32)]
    cols = np.r_[cc[keep], np.arange(D, dtype=np.int32)]
    vals = np.r_[kij[keep], d]
    F = sp.coo_matrix((vals, (rows, cols)), shape=(D, D)).tocsr()
    e, v = ground(F, v0=af, tol=tol)
    v = np.abs(v); v /= np.linalg.norm(v)
    return e, v


def lognorm(x):
    # shift x so that ||exp(x)||_2 = 1
    m = np.max(x)
    return x - (m + 0.5 * np.log(np.sum(np.exp(2.0 * (x - m)))))


class Anderson:
    def __init__(self, m, beta, weights=None):
        self.m = m; self.beta = beta; self.X = []; self.F = []; self.w = weights

    def reset(self):
        self.X = []; self.F = []

    def step(self, x, f, sqrtw=None):
        self.X.append(x.copy()); self.F.append(f.copy())
        if len(self.X) > self.m + 1:
            self.X.pop(0); self.F.pop(0)
        if len(self.X) == 1:
            return x + self.beta * f, 0, None
        dX = np.stack([self.X[i + 1] - self.X[i] for i in range(len(self.X) - 1)], axis=1)
        dF = np.stack([self.F[i + 1] - self.F[i] for i in range(len(self.F) - 1)], axis=1)
        if sqrtw is not None:
            A = dF * sqrtw[:, None]; b = f * sqrtw
        else:
            A = dF; b = f
        gam, *_ = np.linalg.lstsq(A, b, rcond=1e-12)
        xn = x + self.beta * f - (dX + self.beta * dF) @ gam
        return xn, dX.shape[1], gam


def run(args):
    t0 = time.time()
    M = Model(args.lattice)
    D = M.D
    print('lattice', M.geom, 'D', D, flush=True)
    H, diag, ei, ej, hij = M.build_H(args.target_j2)
    Hinit, _, _, _, _ = M.build_H(args.init_source_j2)
    sM = canonical(M.marshall_signs())
    # target ED: diagnostics only
    E0, psi0 = ground(H, tol=args.ed_tol)
    if np.dot(psi0, sM) < 0: psi0 = -psi0
    atrue = np.abs(psi0); atrue /= np.linalg.norm(atrue)
    strue = canonical(np.where(psi0 >= 0, 1, -1).astype(np.int8)); ptrue = atrue ** 2
    print('E0', E0, 'sec', time.time() - t0, flush=True)
    # oracle-free init: exact modulus of J2=init_source ground state, Marshall signs
    _, psiinit = ground(Hinit, tol=args.ed_tol)
    if np.dot(psiinit, sM) < 0: psiinit = -psiinit
    a = np.abs(psiinit); a /= np.linalg.norm(a); s = sM.copy()
    x = lognorm(np.log(np.maximum(a, 1e-300)))

    method = args.method
    acc = Anderson(args.m, args.beta) if method == 'anderson' else None

    def score(a, s):
        E = physical_energy(H, a, s); O = abs(float(np.sum(ptrue * s * strue)))
        return dict(E_trial=E, eps=(E - E0) / abs(E0), w_s=max(0.0, (1 - O) / 2),
                    F_amp=float(abs(np.dot(a, atrue)) ** 2), sign_hash=sign_hash(s))

    hist = [dict(it=0, **score(a, s))]
    print('ITER 0', json.dumps(hist[-1]), flush=True)
    efn_prev = np.inf
    nreset = 0
    terminal = 'maxiter'
    for it in range(1, args.maxiter + 1):
        tt = time.time()
        sold = s.copy()
        efn, afn = fixed_node_solve(H, diag, ei, ej, hij, a, s, args.fn_tol)
        g = lognorm(np.log(np.maximum(afn, 1e-300)))
        f = g - x
        info = dict(E_FN=efn, eps_FN=(efn - E0) / abs(E0), res_log_l2=float(np.linalg.norm(f)),
                    res_log_wl2=float(np.sqrt(np.sum(afn ** 2 * f ** 2))), res_log_inf=float(np.max(np.abs(f))))
        reset = False; reason = None
        if method != 'plain' and efn > efn_prev + args.e_tol * abs(efn):
            reset = True; reason = 'E_FN_up'
        if method == 'plain' or reset:
            xn = g; nhist = 0
            if acc is not None: acc.reset()
        elif method == 'overrelax':
            xn = x + args.w * f; nhist = 0
        else:
            sqrtw = np.sqrt(afn ** 2) if args.weight == 'prob' else None
            xn, nhist, gam = acc.step(x, f, sqrtw)
        if method != 'plain' and not reset:
            jump = float(np.max(np.abs(xn - g))) if np.all(np.isfinite(xn)) else np.inf
            if not np.isfinite(jump) or jump > args.max_jump:
                reset = True; reason = f'jump>{args.max_jump}' if np.isfinite(jump) else 'nonfinite'
                xn = g; nhist = 0
                if acc is not None: acc.reset()
        if reset: nreset += 1
        xn = lognorm(xn)
        an = np.exp(xn); an /= np.linalg.norm(an)
        kamp = an if args.krylov_amp == 'accel' else afn
        snew, eth, G = projected_krylov_update(H, diag, ei, ej, hij, kamp, sold)
        efn_prev = efn
        amp_delta = float(np.linalg.norm(an - a))
        x = xn; a = an; s = snew
        rec = dict(it=it, **score(a, s), **info, n_hist=int(nhist), reset=reason,
                   amp_delta_l2=amp_delta, n_sign_flips=int(np.sum(s != sold)), r_groups=G,
                   min_amp=float(np.min(a)), sec=time.time() - tt)
        hist.append(rec)
        print('ITER', it, json.dumps(rec), flush=True)
        if rec['w_s'] < 1e-14 and rec['eps'] < args.stop_eps:
            terminal = f'converged_eps<{args.stop_eps}'; break
        if np.array_equal(s, sold) and amp_delta < 1e-12:
            terminal = 'fixed_pair'; break

    def first(pred):
        for r in hist:
            if pred(r): return r['it']
        return None

    def settled(pred):
        # first iteration from which pred holds for all later recorded iterations
        k = None
        for r in reversed(hist):
            if pred(r): k = r['it']
            else: break
        return k

    summary = dict(
        it_exact_signs_first=first(lambda r: r['w_s'] < 1e-14),
        it_exact_signs_settled=settled(lambda r: r['w_s'] < 1e-14),
        it_eps_1e5=first(lambda r: r['eps'] < 1e-5),
        it_eps_1e6=first(lambda r: r['eps'] < 1e-6),
        it_eps_1e7=first(lambda r: r['eps'] < 1e-7),
        n_resets=nreset, n_iter=hist[-1]['it'], final_eps=hist[-1]['eps'], final_w_s=hist[-1]['w_s'],
        terminal=terminal,
    )
    out = dict(method=dict(lattice=M.geom, D=D, target_J2=args.target_j2, init_amplitude_source_J2=args.init_source_j2,
                           initial_signs='Marshall', accel=method, m=args.m, beta=args.beta, w=args.w,
                           weight=args.weight, krylov_amp=args.krylov_amp, e_tol=args.e_tol,
                           max_jump=args.max_jump, fn_tol=args.fn_tol,
                           log_space='x=log a normalized ||exp x||=1; g=log a_FN[exp x, s]',
                           safeguard='E_FN increase or jump/nonfinite -> clear history, plain step',
                           oracle_policy='target ED only for scoring (w_s, eps, F_amp)'),
               E0=E0, summary=summary, elapsed_sec=time.time() - t0, history=hist)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)
    print('WROTE', args.out, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lattice', choices=['4x4', '20'], default='4x4')
    ap.add_argument('--method', choices=['plain', 'overrelax', 'anderson'], default='plain')
    ap.add_argument('--m', type=int, default=5)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--w', type=float, default=1.5)
    ap.add_argument('--weight', choices=['uniform', 'prob'], default='uniform')
    ap.add_argument('--krylov-amp', choices=['accel', 'fn'], default='accel')
    ap.add_argument('--e-tol', type=float, default=1e-10)
    ap.add_argument('--max-jump', type=float, default=5.0)
    ap.add_argument('--fn-tol', type=float, default=None)
    ap.add_argument('--ed-tol', type=float, default=1e-12)
    ap.add_argument('--target-j2', type=float, default=0.5)
    ap.add_argument('--init-source-j2', type=float, default=0.0)
    ap.add_argument('--maxiter', type=int, default=150)
    ap.add_argument('--stop-eps', type=float, default=1e-8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.fn_tol is None:
        args.fn_tol = 2e-11 if args.lattice == '4x4' else 2e-10
    run(args)
