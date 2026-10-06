#!/usr/bin/env python3
"""Test A (4x4, exact scoring): does the FN surrogate + edge Gauss-Newton (C1) beat a GOOD second-order
optimizer applied directly to the fixed-sign VMC energy E_H[b] = <bs|H|bs>/<b|b>?

Same model (tab or rf), same tempered samples x ~ |b|^{2 beta} with weights b^{2-2beta}, same N, same number
of steps (4), fresh samples each step, then the same exact energy-optimal Krylov sign step.
  C1      : d = -(A_F + lam)^-1 g_F,  A_F = E[sum_y |F_xy| b_y/b_x (O_x-O_y)(O_x-O_y)^T]   (frozen F = H_FN[a_k,s_k])
  RGN     : d = -(M_H + mu (trM/trS) S + 1e-6)^-1 g_H, M_H = 2E[(E_loc-E) dO dO^T] + E[sum_y (-H^s_xy) b_y/b_x ..]
            (= 2(Hbar - E S) of Webber-Lindsey; indefinite because violating edges have H^s_xy > 0)
  LM      : linear method, non-symmetric estimator (Nightingale/Umrigar-Toulouse), level shift a_diag
  hybrid  : d = -(A_F + lam)^-1 g_H  (VMC gradient, C1's PSD preconditioner)
Scores: eps(<H>) of the new guide (b, s_new) and next-iteration eps(E_FN[b, s_new]); also the converged
fixed-sign VMC optimum vs phi_FN in full capacity (tab, exact).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sl
sys.path.insert(0, str(Path(__file__).resolve().parent))
import amp_design_4x4 as M

ap = argparse.ArgumentParser()
ap.add_argument('--state', default='k0')
ap.add_argument('--model', default='rf')
ap.add_argument('--Ns', default='10000,100000')
ap.add_argument('--beta', type=float, default=0.25)
ap.add_argument('--seeds', type=int, default=2)
ap.add_argument('--steps', type=int, default=4)
ap.add_argument('--grid', default='full')
ap.add_argument('--no-exact', action='store_true')
ap.add_argument('--tag', default='')
args = ap.parse_args()

H, diag, ei, ej, hij = M.build_H(0.5)
E0, _ = M.ground(H, tol=1e-12)
st = np.load(M.ROOT / 'results/net_start_4x4/startH.npz')
a = np.asarray(st['a'], float); a /= np.linalg.norm(a); s = M.canonical(st['s'])
if args.state == 'k1':
    F0 = M.build_fn(diag, ei, ej, hij, a, s); _, phi = M.ground(F0, v0=a, tol=1e-11); phi = np.abs(phi) / np.linalg.norm(phi)
    s = M.krylov_update(H, diag, ei, ej, hij, phi, s); a = phi
fz = M.Frozen(M.build_fn(diag, ei, ej, hij, a, s), a)
sf = s.astype(float)
Hs = sp.diags(sf) @ H @ sp.diags(sf); Hs = Hs.tocsr()
Hs_off = (Hs - sp.diags(Hs.diagonal())).tocsr(); Hdiag = Hs.diagonal()
oid = M.orbit_ids(); no = int(oid.max()) + 1
model = M.tab_model(oid) if args.model == 'tab' else M.rf_model(oid, 1000, 7)
J = model.J


def EH(u):
    b = fz.b(u); return float(b @ (Hs @ b))


def eloc_H(u):
    b = fz.b(u); return (Hs @ b) / b


def next_scores(u):
    b = fz.b(u); sn = M.krylov_update(H, diag, ei, ej, hij, b, s)
    p = b * sn; eH = float((p @ (H @ p) - E0) / abs(E0))
    e, _ = M.ground(M.build_fn(diag, ei, ej, hij, b, sn), v0=np.maximum(b, 1e-15), tol=1e-11)
    return dict(eps_H_next=eH, eps_FN_next=float((e - E0) / abs(E0)), eps_H_fixed=float((EH(u) - E0) / abs(E0)),
                frozen_gain=fz.gain(u))


def draw(u, n, rng, beta):
    b = fz.b(u); q = b ** (2 * beta); p = q / q.sum()
    smp = rng.choice(M.D, size=n, p=p)
    U, C = np.unique(smp, return_counts=True)
    w = C * b[U] ** 2 / p[U]; return U, w / w.sum()


def edge_mat(U, w, b, Off, sign):
    """sum_x w_x sum_y c (b_y/b_x) (e_ox - e_oy)(...)^T with c = sign * Off_xy (orbit space)."""
    R = Off[U].tocoo(); xi = U[R.row]; yi = R.col
    ww = w[R.row] * sign * R.data * b[yi] / b[xi]
    ox = oid[xi]; oy = oid[yi]; m = ox != oy
    A = np.zeros((no, no))
    np.add.at(A, (ox[m], ox[m]), ww[m]); np.add.at(A, (oy[m], oy[m]), ww[m])
    np.add.at(A, (ox[m], oy[m]), -ww[m]); np.add.at(A, (oy[m], ox[m]), -ww[m])
    return A


def stats(u, U, w):
    b = fz.b(u)
    elF = fz.eloc(u)[U]; EF = float(w @ elF)
    elH = eloc_H(u)[U]; EHh = float(w @ elH)
    gF = 2 * np.bincount(oid[U], weights=w * (elF - EF), minlength=no)
    hH = np.bincount(oid[U], weights=w * (elH - EHh), minlength=no); gH = 2 * hH
    Ob = np.bincount(oid[U], weights=w, minlength=no)
    S = np.diag(Ob) - np.outer(Ob, Ob)
    AF = edge_mat(U, w, b, fz.Foff, -1.0)                   # |F_xy| = -F_xy
    EdH = edge_mat(U, w, b, Hs_off, -1.0)                   # (-H^s_xy): negative on violating edges
    Dd = np.diag(hH) - np.outer(Ob, hH) - np.outer(hH, Ob)  # E[(E_loc-E) dO dO^T]
    MH = 2 * Dd + EdH
    # linear method, non-symmetric estimator
    R = Hs_off[U].tocoo(); xi = U[R.row]; yi = R.col; c = R.data * b[yi] / b[xi]
    K0 = np.zeros((no, no))
    np.add.at(K0, (oid[U], oid[U]), w * Hdiag[U])
    np.add.at(K0, (oid[xi], oid[yi]), w[R.row] * c)
    r = np.bincount(oid[U], weights=w * Hdiag[U], minlength=no) + np.bincount(oid[yi], weights=w[R.row] * c, minlength=no)
    Hb = K0 - np.outer(Ob, r) - np.outer(hH, Ob)
    H0j = r - Ob * EHh
    return dict(gF=gF, gH=gH, S=S, AF=AF, MH=MH, Hb=Hb, H0j=H0j, hH=hH, EH=EHh)


def lift(Mt): return J.T @ Mt @ J if model.name == 'rf' else Mt
def lv(v): return J.T @ v if model.name == 'rf' else v
def down(d): return (J @ d if model.name == 'rf' else d)[oid]


def solve_shift(Mx, g, lam):
    P = Mx.shape[0]; sh = lam * max(abs(np.trace(Mx)) / P, 1e-300)
    return -sl.solve(Mx + sh * np.eye(P), g, assume_a='sym')


def step(method, par, T):
    if method == 'C1':
        return down(solve_shift(lift(T['AF']), lv(T['gF']), par))
    if method == 'hybrid':
        return down(solve_shift(lift(T['AF']), lv(T['gH']), par))
    if method == 'RGN':      # par = (mu_S, lam_I, trust): RGN damping, identity shift, RMS trust cap (on samples)
        mu, lamI, trust = par
        Mx = lift(T['MH']); Sx = lift(T['S'])
        Mx = Mx + mu * (abs(np.trace(Mx)) / max(np.trace(Sx), 1e-300)) * Sx
        d = solve_shift(Mx, lv(T['gH']), lamI)
        if trust is not None:
            dS = float(np.sqrt(max(d @ Sx @ d, 0.0)))
            if dS > trust: d = d * (trust / dS)
        return down(d)
    if method == 'RGNA':     # RGN on E_H damped by the PSD stoquastic-edge Laplacian A_F: (M_H + mu A_F) d = -g_H
        Mx = lift(T['MH']) + par * lift(T['AF'])
        return down(solve_shift(Mx, lv(T['gH']), 1e-6))
    if method == 'LM':
        Hb = lift(T['Hb']); Sx = lift(T['S']); P = Sx.shape[0]
        sc = np.trace(Sx) / P
        Hm = np.zeros((P + 1, P + 1)); Sm = np.zeros((P + 1, P + 1))
        Hm[0, 0] = T['EH']; Hm[1:, 0] = lv(T['hH']); Hm[0, 1:] = lv(T['H0j'])
        Hm[1:, 1:] = Hb + T['EH'] * Sx + par * sc * np.eye(P)
        Sm[0, 0] = 1.0; Sm[1:, 1:] = Sx + 1e-8 * sc * np.eye(P)
        ev, V = sl.eig(Hm, Sm)
        order = np.argsort(ev.real)
        for k in order[:50]:
            v = V[:, k].real
            nrm = v[0] ** 2 + v[1:] @ Sm[1:, 1:] @ v[1:]
            if nrm > 0 and v[0] ** 2 / nrm > 0.5:
                return down(v[1:] / v[0])
        return down(np.zeros(P))
    raise ValueError(method)


def est_H_pair(u_cur, u_new, U, w):
    """paired self-normalised E_H of b_new on samples drawn for b_cur (weights w already -> b_cur^2)."""
    lw = 2.0 * (u_new[U] - u_cur[U]); e = np.exp(lw - lw.max()) * w
    return float(e @ eloc_H(u_new)[U] / e.sum())


def run_rgntr(par, N, rng, beta, steps):
    """RGN on E_H, damping (M + lam*Dmat), Levenberg-Marquardt trust region verified on FRESH samples:
    accept if measured (paired, fresh tempered samples) decrease >= 0.25 * predicted decrease."""
    kind, lam = par
    u = np.zeros(M.D); hist = []; nrej = 0
    for k in range(steps):
        U, w = draw(u, N, rng, beta); T = stats(u, U, w)
        Uv, wv = draw(u, N, rng, beta)
        Mx = lift(T['MH']); g = lv(T['gH'])
        Dm = lift(T['AF'] if kind == 'A' else T['S'])
        Dm = Dm * (abs(np.trace(Mx)) / max(np.trace(Dm), 1e-300))
        P = Mx.shape[0]; Ecur = est_H_pair(u, u, Uv, wv)
        for tries in range(6):
            try:
                d = -sl.solve(Mx + lam * Dm + 1e-8 * abs(np.trace(Mx)) / P * np.eye(P), g, assume_a='sym')
            except (ValueError, np.linalg.LinAlgError, sl.LinAlgError):
                lam *= 4; continue
            pred = float(g @ d + 0.5 * d @ Mx @ d)
            uc = u + down(d)
            meas = est_H_pair(u, uc, Uv, wv) - Ecur if np.all(np.isfinite(uc)) and np.max(np.abs(down(d))) < 5 else np.inf
            if pred < 0 and meas < 0 and meas / pred > 0.25:
                u = uc
                if meas / pred > 0.75: lam /= 3
                break
            lam *= 4; nrej += 1
        hist.append(dict(eps_H_fixed=float((EH(u) - E0) / abs(E0)), frozen_gain=fz.gain(u), lam=float(lam)))
    return u, hist


def run(method, par, N, rng, beta, steps):
    if method == 'RGNTR':
        return run_rgntr(par, N, rng, beta, steps)
    u = np.zeros(M.D); hist = []
    for k in range(steps):
        if N is None:
            b = fz.b(u); U = np.arange(M.D); w = b * b
        else:
            U, w = draw(u, N, rng, beta)
        try:
            d = step(method, par, stats(u, U, w))
        except (ValueError, np.linalg.LinAlgError, sl.LinAlgError):
            d = np.zeros(M.D)
        if not np.all(np.isfinite(d)) or np.max(np.abs(d)) > 5.0: d = np.zeros(M.D)   # reject blow-ups
        u = u + d
        hist.append(dict(eps_H_fixed=float((EH(u) - E0) / abs(E0)), frozen_gain=fz.gain(u)))
    return u, hist


GRID = {'C1': [1e-6, 1e-5], 'hybrid': [1e-6],
        'RGN': [(1e-2, 1e-6, None), (1e-2, 1e-4, None), (1e-2, 1e-3, None), (1e-1, 1e-3, None),
                (1e-2, 1e-6, 0.02), (1e-2, 1e-4, 0.02), (1e-2, 1e-6, 0.05)]}
if args.grid == 'proto':
    GRID = {'C1': [1e-6], 'hybrid': [1e-6], 'RGNTR': [('A', 0.3), ('A', 0.03), ('S', 0.01)]}
if args.grid == 'rgna':
    GRID = {'C1': [1e-6], 'hybrid': [1e-6], 'RGNA': [0.1, 0.3, 1.0], 'RGN': [(1e-2, 1e-4, None)]}
if args.grid == 'small':
    GRID = {'C1': [1e-6], 'hybrid': [1e-6], 'RGN': [1e-2, 1e-1], 'LM': [1e-2, 1e-1]}
def pk(p): return '_'.join(('%g' % x if not isinstance(x, str) else x) if x is not None else 'none' for x in p) if isinstance(p, tuple) else '%g' % p


t0 = time.time()
res = dict(state=args.state, model=args.model, beta=args.beta, base=next_scores(np.zeros(M.D)),
           ideal=next_scores(fz.ustar), runs={})
print(args.state, args.model, 'base', res['base'], '\n ideal', res['ideal'], flush=True)

if args.no_exact:
    GRID_EXACT = {}
else:
    GRID_EXACT = GRID
# exact-weight versions (infinite N): 4 steps, plus converged fixed-sign VMC optimum (30 LM/RGN steps)
for meth, pars in GRID_EXACT.items():
    for par in pars:
        u, hist = run(meth, par, None, None, 1.0, args.steps)
        key = f'exact_{meth}_{pk(par)}'; res['runs'][key] = dict(final=next_scores(u), hist=hist)
        print(f'{key:22s}', {k: float('%.4g' % v) for k, v in res['runs'][key]['final'].items()}, f'{time.time()-t0:.0f}s', flush=True)
for meth, par in ([] if args.no_exact else (('RGN', (1e-2, 1e-6, 0.02)), ('C1', 1e-6))):
    u, hist = run(meth, par, None, None, 1.0, 30)
    key = f'exact_{meth}_{pk(par)}_30steps'; res['runs'][key] = dict(final=next_scores(u), hist=hist[-1:])
    print(f'{key:22s}', {k: float('%.4g' % v) for k, v in res['runs'][key]['final'].items()}, flush=True)

for N in [int(x) for x in args.Ns.split(',')]:
    for meth, pars in GRID.items():
        for par in pars:
            rows = []
            for sd in range(args.seeds):
                u, hist = run(meth, par, N, np.random.default_rng([sd, N, 41]), args.beta, args.steps)
                rows.append(dict(final=next_scores(u), hist=hist))
            key = f'N{N}_{meth}_{pk(par)}'
            mean = {k: float(np.mean([r['final'][k] for r in rows])) for k in rows[0]['final']}
            sd_ = {k: float(np.std([r['final'][k] for r in rows])) for k in rows[0]['final']}
            res['runs'][key] = dict(mean=mean, std=sd_, rows=rows)
            print(f'{key:24s}', {k: float('%.4g' % v) for k, v in mean.items()}, f'{time.time()-t0:.0f}s', flush=True)
out = M.ROOT / f'results/amp_design/vmc_vs_fn_{args.state}_{args.model}_b{args.beta:g}{args.tag}.json'
Path(out).write_text(json.dumps(res, indent=1))
print('DONE', time.time() - t0)
