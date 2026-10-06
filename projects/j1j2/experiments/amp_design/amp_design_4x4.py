#!/usr/bin/env python3
"""Design test for the amplitude-update step of the FN/Krylov loop (4x4, J2/J1=0.5, exact, CPU, minutes).

Question: in the tiny-correction regime of the net-start loop (log-ratio RMS ~ 4e-3), is the bottleneck
(a) the first-order Fisher/SR geometry (ITE-like, many steps) or (b) sampling noise, and does a
Gauss-Newton step whose metric is the Dirichlet ("edge-Fisher") form of the stoquastic H_FN fix it?

Frozen problem: minimise E[b] = <b|F|b>/<b|b>, F = H_FN[a_k, s_k] (stoquastic), b = a_k exp(u).
Identity used: for stoquastic F with ground state phi, E[phi e^v] - E_FN
   = 1/2 E_{x~phi^2}[ sum_y |F_xy| (phi_y/phi_x) (v_x - v_y)^2 ] + O(v^3),
so the Hessian of the Rayleigh quotient in log-amplitude space is the edge-weighted graph Laplacian.

Models (both linear in log space around a_k, so SR/GN are exact quadratic problems):
  - tab : one parameter per translation orbit (full capacity for translation-invariant amplitudes;
          no generalisation to unsampled orbits other than through Hamiltonian edges).
  - rf  : K translation-symmetrised random tanh features (generalising model, P = K).
Optimisers (from N i.i.d. samples of b^2 per step; E_loc exact given b on neighbours):
  - SR protocol  : Fisher metric, trust RMS 0.02, backtracking, paired val gate, first-reject stop,
                   <= 20 steps, trust shrink (replica of loop_from_net_4x4.py).
  - SR oracle-tau: same SR direction, step length by exact line search (isolates direction quality).
  - GN           : (A_hat + mu) d = -g_hat, A_hat = sampled edge-Fisher, step factor 1, no gate.
ED is used only for sampling (exact sampler) and for scoring the frozen gain.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # this Mac's openblas is 20x slower multi-threaded
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.linalg as sla_dense

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'net_start_4x4'))
from common import build_H, ground, canonical, marshall, BASIS, D, N as NS, L, bits2x  # noqa

ROOT = Path(__file__).resolve().parents[2]


# ---------------- FN and Krylov sign step (copied from loop_from_net_4x4.py, no jax) ----------------
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
    delta = (np.bincount(lo[m], weights=-2.0 * val[m], minlength=G) +
             np.bincount(hi[m], weights=+2.0 * val[m], minlength=G))
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
    rows = np.r_[rr[keep], np.arange(D, dtype=np.int32)]
    cols = np.r_[cc[keep], np.arange(D, dtype=np.int32)]
    vals = np.r_[kij[keep], diag + corr]
    return sp.coo_matrix((vals, (rows, cols)), shape=(D, D)).tocsr()


# ---------------- translation orbits ----------------
def orbit_ids():
    bits = ((BASIS[:, None] >> np.arange(NS, dtype=np.uint32)) & 1).astype(np.uint32)
    reps = np.full(D, np.iinfo(np.uint32).max, dtype=np.uint32)
    for dx in range(L):
        for dy in range(L):
            perm = np.array([((i % L + dx) % L) + L * ((i // L + dy) % L) for i in range(NS)])
            t = np.zeros(D, dtype=np.uint32)
            for i in range(NS):
                t |= bits[:, i] << np.uint32(perm[i])
            reps = np.minimum(reps, t)
    _, oid = np.unique(reps, return_inverse=True)
    return oid.astype(np.int64)


# ---------------- frozen problem ----------------
class Frozen:
    def __init__(self, F, a):
        self.F = F.tocsr(); self.a = a / np.linalg.norm(a)
        self.loga = np.log(np.maximum(self.a, 1e-300))
        self.E_FN, phi = ground(self.F, v0=np.maximum(self.a, 1e-15), tol=1e-11)
        self.phi = np.abs(phi) / np.linalg.norm(phi)
        self.E_base = self.energy(np.zeros(D))
        self.G = self.E_base - self.E_FN
        Foff = self.F - sp.diags(self.F.diagonal())
        self.Foff = Foff.tocsr()
        assert self.Foff.data.max() <= 1e-14, 'F not stoquastic'
        Fc = self.Foff.tocoo(); self.er, self.ec, self.ev = Fc.row, Fc.col, -Fc.data  # |F_xy|
        self.ustar = np.log(self.phi) - self.loga

    def b(self, u):
        lb = self.loga + u; v = np.exp(lb - lb.max()); return v / np.linalg.norm(v)

    def energy(self, u):
        if not np.all(np.isfinite(u)): return np.inf
        b = self.b(u); e = float(b @ (self.F @ b))
        return e if np.isfinite(e) else np.inf

    def gain(self, u):
        return (self.E_base - self.energy(u)) / self.G

    def eloc(self, u):
        b = self.b(u); return (self.F @ b) / b


def ite_curve(fz, nmax=200):
    """function-space ITE / steepest descent: b <- b - tau (F - E) b, exact line search on tau each step
    (= SR with full capacity and exact data, in its linear-amplitude form)."""
    b = fz.a.copy(); out = {}
    def en(v): return float(v @ (fz.F @ v) / (v @ v))
    for n in range(1, nmax + 1):
        E = en(b); r = fz.F @ b - E * b
        ts = np.geomspace(1e-5, 10.0, 80); es = [en(b - t * r) for t in ts]
        t0 = ts[int(np.argmin(es))]
        ts2 = np.linspace(0.5 * t0, 1.5 * t0, 41); es2 = [en(b - t * r) for t in ts2]
        t = ts2[int(np.argmin(es2))]; b = b - t * r; b /= np.linalg.norm(b)
        if n in (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000):
            out[n] = dict(gain=(fz.E_base - en(b)) / fz.G, tau=float(t))
    return out


def krylov_rr(fz, nmax=5):
    out = {}; V = [fz.a.copy()]
    for n in range(1, nmax + 1):
        w = fz.F @ V[-1]
        for v in V: w -= (v @ w) * v
        for v in V: w -= (v @ w) * v
        w /= np.linalg.norm(w); V.append(w)
        Q = np.array(V).T; Hs = Q.T @ (fz.F @ Q)
        ew, ev = np.linalg.eigh(Hs); c = Q @ ev[:, 0]
        out[n] = float((fz.E_base - ew[0]) / fz.G)
    return out


# ---------------- linear log-models: u = (J_orb theta)[oid] ----------------
# Both models are translation invariant, so everything is assembled in orbit space (P_tab x P_tab).
class Model:
    def __init__(self, oid, Jorb, name):
        self.oid = oid; self.J = Jorb; self.P = Jorb.shape[1]; self.name = name; self.no = Jorb.shape[0]

    def u(self, th): return (self.J @ th)[self.oid]

    def lift(self, M_tab):   # J^T M J
        return M_tab if self.name == 'tab' else self.J.T @ M_tab @ self.J


def tab_model(oid):
    no = int(oid.max()) + 1; return Model(oid, np.eye(no), 'tab')


def rf_model(oid, K, seed, scale=0.6):
    rng = np.random.default_rng(seed); X = bits2x(BASIS)
    W = rng.normal(0, scale / np.sqrt(NS), (K, NS)); c = rng.normal(0, 0.3, K)
    Jf = np.zeros((D, K))
    for dx in range(L):
        for dy in range(L):
            perm = np.array([((i % L + dx) % L) + L * ((i // L + dy) % L) for i in range(NS)])
            Xt = np.empty_like(X); Xt[:, perm] = X
            Jf += np.tanh(Xt @ W.T + c)
    Jf -= Jf.mean(0); Jf /= Jf.std(0) + 1e-12
    no = int(oid.max()) + 1; Jorb = np.zeros((no, K)); Jorb[oid] = Jf
    assert np.allclose(Jorb[oid], Jf)
    return Model(oid, Jorb, 'rf')


def sample(fz, u, n, rng):
    b = fz.b(u); p = b * b; p /= p.sum()
    smp = rng.choice(D, size=n, p=p)
    U, C = np.unique(smp, return_counts=True)
    return U, C / n


def tab_stats(fz, oid, u, U, w, edges=True):
    """orbit-space g_tab, S_tab, A_tab from weighted configurations U (weights w, sum 1)."""
    no = int(oid.max()) + 1
    el = fz.eloc(u)[U]; E = float(w @ el)
    g = 2.0 * np.bincount(oid[U], weights=w * (el - E), minlength=no)
    wo = np.bincount(oid[U], weights=w, minlength=no)
    S = np.diag(wo) - np.outer(wo, wo)
    A = None
    if edges:
        b = fz.b(u); R = fz.Foff[U].tocoo()
        xi = U[R.row]; yi = R.col; ww = w[R.row] * (-R.data) * b[yi] / b[xi]
        ox = oid[xi]; oy = oid[yi]; m = ox != oy
        A = np.zeros((no, no))
        np.add.at(A, (ox[m], ox[m]), ww[m]); np.add.at(A, (oy[m], oy[m]), ww[m])
        np.add.at(A, (ox[m], oy[m]), -ww[m]); np.add.at(A, (oy[m], ox[m]), -ww[m])
    return g, S, A, E


def solve(M, g, lam):
    P = M.shape[0]; sh = lam * max(np.trace(M) / P, 1e-300)
    return -sla_dense.solve(M + sh * np.eye(P), g, assume_a='sym')


def paired_val(fz, u_cur, u_new, U, w):
    """self-normalised paired estimate of E[b_new] on val samples drawn from b_cur^2."""
    lw = 2.0 * (u_new[U] - u_cur[U]); e = np.exp(lw - lw.max()) * w
    return float(e @ fz.eloc(u_new)[U] / e.sum())


def sr_dir(fz, model, u, U, w, lam):
    g, S, _, _ = tab_stats(fz, model.oid, u, U, w, edges=False)
    return solve(model.lift(S), model.J.T @ g, lam)


def run_sr_protocol(fz, model, N, rng, lam, trust0=0.02, steps=20):
    """replica of loop_from_net_4x4.py: trust RMS 0.02, backtracking, paired val gate, first-reject stop."""
    th = np.zeros(model.P); u = model.u(th); trust = trust0; nacc = 0; nst = 0
    for st in range(steps):
        Ut, wt = sample(fz, u, N // 2, rng); Uv, wv = sample(fz, u, N - N // 2, rng); nst += 1
        d = sr_dir(fz, model, u, Ut, wt, lam)
        du = model.u(d); m = wt @ du[Ut]; rms = float(np.sqrt(wt @ (du[Ut] - m) ** 2))
        eta = trust / max(rms, 1e-300)
        Eva = paired_val(fz, u, u, Uv, wv); acc = False
        for fac in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            thc = th + eta * fac * d; uc = model.u(thc)
            if paired_val(fz, u, uc, Uv, wv) < Eva:
                th, u = thc, uc; acc = True
                if fac < 1.0: trust *= fac
                break
        if not acc: break
        nacc += 1
    return dict(gain=fz.gain(u), nacc=nacc, samples=N * nst, u=u)


def run_sr_oracle(fz, model, N, rng, lam, steps=10):
    """same SR direction from N samples per step; step length by exact line search (oracle)."""
    th = np.zeros(model.P); u = model.u(th)
    for st in range(steps):
        U, w = sample(fz, u, N, rng)
        d = sr_dir(fz, model, u, U, w, lam); du = model.u(d)
        ts = np.geomspace(1e-8, 1e3, 120); es = [fz.energy(u + t * du) for t in ts]
        if min(es) < fz.energy(u): th = th + ts[int(np.argmin(es))] * d; u = model.u(th)
    return dict(gain=fz.gain(u), samples=N * steps, u=u)


def run_gn(fz, model, N, rng, lam, steps=2):
    """Gauss-Newton with the sampled edge-Fisher (Dirichlet) metric; step factor 1, no gate."""
    th = np.zeros(model.P); u = model.u(th); gains = []
    for st in range(steps):
        U, w = sample(fz, u, N, rng)
        g, _, A, _ = tab_stats(fz, model.oid, u, U, w)
        d = solve(model.lift(A), model.J.T @ g, lam)
        th = th + d; u = model.u(th); gains.append(fz.gain(u))
    return dict(gain=fz.gain(u), gains=gains, samples=N * steps, u=u)


def run_exact(fz, model, lam, steps=2, sr_steps=100):
    """exact weights (all configurations, w=b^2): tangent-space ceilings of GN and of SR (oracle tau)."""
    out = dict(gn=[], sr=[])
    U = np.arange(D)
    th = np.zeros(model.P); u = model.u(th)
    for st in range(steps):
        b = fz.b(u); g, _, A, _ = tab_stats(fz, model.oid, u, U, b * b)
        d = solve(model.lift(A), model.J.T @ g, lam); du = model.u(d)
        ts = np.linspace(0.0, 2.0, 81); es = [fz.energy(u + t * du) for t in ts]
        th = th + d; u = model.u(th)
        out['gn'].append(dict(gain=fz.gain(u), best_fac=float(ts[int(np.argmin(es))])))
    th = np.zeros(model.P); u = model.u(th)
    for st in range(1, sr_steps + 1):
        b = fz.b(u); d = sr_dir(fz, model, u, U, b * b, lam); du = model.u(d)
        ts = np.geomspace(1e-8, 1e3, 120); es = [fz.energy(u + t * du) for t in ts]
        if min(es) < fz.energy(u): th = th + ts[int(np.argmin(es))] * d; u = model.u(th)
        if st in (1, 3, 10, 30, 100): out['sr'].append(dict(steps=st, gain=fz.gain(u)))
    # best possible in the span: exact projection of u* in the Dirichlet norm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--Ns', default='1000,10000,100000,1000000')
    ap.add_argument('--rfK', type=int, default=1000)
    ap.add_argument('--rf-states', default='k0,k1')
    ap.add_argument('--Ns-rf', default='1000,10000,100000')
    ap.add_argument('--states', default='k0,k1,k1stale,k5')
    ap.add_argument('--skip-rf', action='store_true')
    ap.add_argument('--lam-sr', type=float, default=1e-3)
    ap.add_argument('--lam-gn', type=float, default=1e-3)
    ap.add_argument('--out', default=str(ROOT / 'results/amp_design/amp_design_4x4.json'))
    args = ap.parse_args()
    t0 = time.time()
    H, diag, ei, ej, hij = build_H(0.5)
    E0, psi0 = ground(H, tol=1e-12)
    st = np.load(ROOT / 'results/net_start_4x4/startH.npz')
    aH = np.asarray(st['a'], float); aH /= np.linalg.norm(aH); sH = canonical(st['s'])

    # ideal-loop states
    states = {}; a, s = aH.copy(), sH.copy()
    for k in range(6):
        F = build_fn(diag, ei, ej, hij, a, s)
        states[f'k{k}'] = (F, a.copy(), s.copy())
        _, phi = ground(F, v0=np.maximum(a, 1e-15), tol=1e-11); phi = np.abs(phi) / np.linalg.norm(phi)
        sn = krylov_update(H, diag, ei, ej, hij, phi, s)
        if k == 0:   # stale amplitude: net amplitude with the first Krylov signs (learner refreshed nothing)
            states['k1stale'] = (build_fn(diag, ei, ej, hij, aH, sn), aH.copy(), sn.copy())
        a, s = phi, sn
    oid = orbit_ids(); tab = tab_model(oid)
    print(f'orbits P_tab={tab.P}', flush=True)
    rf = None if args.skip_rf else rf_model(oid, args.rfK, 7)
    Ns = [int(x) for x in args.Ns.split(',')]; Ns_rf = [int(x) for x in args.Ns_rf.split(',')]
    res = dict(E0=E0, P_tab=tab.P, P_rf=(rf.P if rf else None), states={})

    def eps_fn_next(fz, u, s):
        b = fz.b(u); sn = krylov_update(H, diag, ei, ej, hij, b, s)
        e, _ = ground(build_fn(diag, ei, ej, hij, b, sn), v0=np.maximum(b, 1e-15), tol=1e-11)
        return float((e - E0) / abs(E0))

    def eps_after(fz, u, s):
        sn = krylov_update(H, diag, ei, ej, hij, fz.b(u), s)
        p = fz.b(u) * sn; return float((p @ (H @ p) - E0) / abs(E0))

    for name in args.states.split(','):
        F, a, s = states[name]; fz = Frozen(F, a)
        us = fz.ustar - np.sum(fz.phi ** 2 * fz.ustar)
        orb_var = float(np.sum(fz.phi ** 2 * (us - np.bincount(oid, weights=fz.phi**2 * us)[oid] /
                                               np.bincount(oid, weights=fz.phi**2)[oid]) ** 2))
        R = dict(G=fz.G, E_base=fz.E_base, E_FN=fz.E_FN, ustar_rms=float(np.sqrt(np.sum(fz.phi ** 2 * us ** 2))),
                 ustar_offorbit_var=orb_var, eps_guide=float((fz.E_base - E0) / abs(E0)),
                 eps_after_ideal=eps_after(fz, fz.ustar, s))
        R['ite'] = ite_curve(fz, 100)
        R['eps_FN_next_noupdate'] = eps_fn_next(fz, np.zeros(D), s)
        R['eps_FN_next_ideal'] = eps_fn_next(fz, fz.ustar, s)
        R['krylov_rr'] = krylov_rr(fz, 5)
        R['exact_tab'] = run_exact(fz, tab, 1e-10, 2)
        print(name, json.dumps({k: v for k, v in R.items()}), flush=True)
        use_rf = rf is not None and name in args.rf_states.split(',')
        models = [('tab', tab)] + ([('rf', rf)] if use_rf else [])
        if use_rf:
            R['exact_rf'] = run_exact(fz, rf, 1e-8, 2, sr_steps=30)
            print(name, 'exact_rf', R['exact_rf'], flush=True)
        for mname, model in models:
            for N in (Ns if mname == 'tab' else Ns_rf):
                rows = dict(sr_protocol=[], sr_oracle=[], gn1=[], gn2=[], sr_protocol_nacc=[], eps_gn2=[], eps_srp=[],
                            epsFN_gn2=[], epsFN_srp=[])
                for sd in range(args.seeds):
                    rng = np.random.default_rng([sd, N, 11])
                    shift_sr = args.lam_sr
                    p = run_sr_protocol(fz, model, N, rng, shift_sr)
                    rows['sr_protocol'].append(p['gain']); rows['sr_protocol_nacc'].append(p['nacc'])
                    rows['eps_srp'].append(eps_after(fz, p['u'], s)); rows['epsFN_srp'].append(eps_fn_next(fz, p['u'], s))
                    rows['sr_oracle'].append(run_sr_oracle(fz, model, N, rng, shift_sr, steps=10)['gain'])
                    gq = run_gn(fz, model, N, rng, args.lam_gn, steps=2)
                    rows['gn1'].append(gq['gains'][0]); rows['gn2'].append(gq['gains'][1])
                    rows['eps_gn2'].append(eps_after(fz, gq['u'], s)); rows['epsFN_gn2'].append(eps_fn_next(fz, gq['u'], s))
                summ = {k: [float(np.mean(v)), float(np.std(v))] for k, v in rows.items()}
                R[f'{mname}_N{N}'] = summ
                print(name, mname, N, json.dumps(summ), f'{time.time()-t0:.0f}s', flush=True)
        res['states'][name] = R
        Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    print('DONE', time.time() - t0)


if __name__ == '__main__':
    main()
