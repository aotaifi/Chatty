#!/usr/bin/env python3
"""Tail concentration of one frozen-FN iteration vs system size (exact, symmetric sector k=0, A1, flip+, J2/J1 = 0.5).

For a guide (a, s) and F = H_FN[a, s] with Perron vector phi (E_FN), delta* = log phi - log a:
  G  = (E_F[a] - E_FN)/N                       exact frozen-FN gain per site (Rayleigh quotient of F at a)
  Q  = 1/N sum_xy<ordered> 1/2 |F_xy| phi_x phi_y (delta_x - delta_y)^2        quadratic form of writeback_tail
  Gx = same sum with (rho_x - rho_y)^2, rho = a/phi, divided by <a|a>/N       EXACT identity: Gx == G
Node decomposition: c_x = 1/2 sum_y |F_xy| phi_x phi_y (...)^2 (each edge split equally between its endpoints);
edge decomposition: each unordered edge attributed to its lowest-phi^2 endpoint ('min') or highest ('max').
Everything is accumulated on 8 bins per decade of the per-configuration phi^2 = u_r^2/n_r (u = sector vector).
Truncation curves (exact nonlinear): E_F of the vector equal to phi on a set S of configurations and to a elsewhere.

Reuses (unmodified) krylov_sign_structure/experiments/closed_fn_krylov_sym6x6.py (module C): symmetric-sector H, FN solve,
current-sign Krylov step with energy-optimal threshold.  Loop = plain (no acceleration): (v,s) -> FN solve -> v := phi,
s := Krylov(phi, s), from the J2=0 ground-state amplitude with Marshall signs.

  python tvn.py --cluster N16 --kmax 3 --out OUT.json [--vit]
"""
import argparse, json, os, sys, time, types
import numpy as np
import numba
from numba import prange

sys.path.insert(0, os.environ.get("FN_CODE", "/home/a/A.Otaifi/chatty_fn6x6/code"))
sys.path.insert(0, os.environ.get("ED_COMMON_DIR", "/home/a/A.Otaifi/chatty_fn6x6/code"))
import closed_fn_krylov_sym6x6 as C

BPD = 8            # bins per decade of per-configuration phi^2
NB = 40 * BPD      # covers phi^2 down to 1e-40
NCH = 11           # orbits, configs, mass, c_quad, c_exact, eq_min, ex_min, eq_max, ex_max, ncfg_by_gaindens, gain_by_gaindens
CLUSTERS = {'N16': (4, None), 'N20': (0, '4,0,1,5'), 'N24': (0, '4,0,0,6'), 'N28': (0, '4,0,1,7'),
            'N32': (0, '4,4,-4,4'), 'N36': (6, None)}
log = lambda *a: print(time.strftime('[%H:%M:%S]'), *a, flush=True)


@numba.njit(parallel=True, cache=False)
def k_hist(indptr, indices, data, s, u, dl, rho, bn, nrep, gnorm, hist):
    n = indptr.shape[0] - 1
    nb = hist.shape[0]
    nb_bpd = 8.0
    chunk = (n + nb - 1) // nb
    for b in prange(nb):
        lo = b * chunk
        hi = min(n, lo + chunk)
        for i in range(lo, hi):
            bi = bn[i]
            hist[b, 0, bi] += 1.0
            hist[b, 1, bi] += nrep[i]
            hist[b, 2, bi] += u[i] * u[i]
            cq = 0.0
            cx = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                if j == i or s[j] == s[i]:
                    continue
                wuu = data[p] * u[i] * u[j]
                dq = wuu * (dl[i] - dl[j]) ** 2
                dx = wuu * (rho[i] - rho[j]) ** 2
                cq += dq
                cx += dx
                if j > i:
                    bj = bn[j]
                    bmin = max(bi, bj)
                    bmax = min(bi, bj)
                    hist[b, 5, bmin] += dq
                    hist[b, 6, bmin] += dx
                    hist[b, 7, bmax] += dq
                    hist[b, 8, bmax] += dx
            hist[b, 3, bi] += 0.5 * cq
            hist[b, 4, bi] += 0.5 * cx
            # Lorenz channels: bin configurations by their gain density relative to the uniform density
            rel = 0.5 * cx * gnorm / nrep[i]
            if rel > 1e-300:
                bg = int(np.floor((20.0 - np.log10(rel)) * (nb_bpd)))
            else:
                bg = hist.shape[2] - 1
            bg = min(max(bg, 0), hist.shape[2] - 1)
            hist[b, 9, bg] += nrep[i]
            hist[b, 10, bg] += 0.5 * cx * gnorm / nrep[i] * nrep[i] / 1.0


@numba.njit(parallel=True, cache=False)
def k_jacobi(indptr, indices, data, s, Dg, u, E, thr, unew, rel):
    n = indptr.shape[0] - 1
    for i in prange(n):
        den = Dg[i] - E
        if den > thr:
            acc = 0.0
            si = s[i]
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                if s[j] != si:
                    acc += data[p] * u[j]
            val = acc / den
            unew[i] = val
            rel[i] = abs(val - u[i]) / max(u[i], 1e-300)
        else:
            unew[i] = u[i]
            rel[i] = 0.0


class Diag:
    def __init__(self, S, E0):
        self.S = S; self.E0 = E0; self.N = S.N; self.trunc_step = 1
        self.nrep = S.n_orb

    def Fmv(self, Dg, s, x):
        y = np.empty(self.S.D)
        C.k_fn_matvec(self.S.indptr, self.S.indices, self.S.data, s, Dg, np.ascontiguousarray(x, dtype=np.float64), y)
        return y

    def rayleigh(self, Dg, s, x):
        return float(x @ self.Fmv(Dg, s, x) / (x @ x))

    def polish(self, Dg, s, u, E, maxsweep, tol=1e-9, thr=1.0):
        """Masked Jacobi sweeps on rows with D_FN - E > thr: refines the tail components to relative accuracy
        (the eigensolver's tolerance is absolute and says nothing about components of 1e-8)."""
        S = self.S
        rel = np.empty(S.D); un = np.empty(S.D)
        hist = []
        for it in range(maxsweep + 1):
            k_jacobi(S.indptr, S.indices, S.data, s, Dg, u, E, thr, un, rel)
            m = float(rel.max())
            q99 = float(np.quantile(rel[:: max(1, S.D // 2000000)], 0.999))
            hist.append((it, m, q99))
            if it % 10 == 0 or m < tol:
                log(f'   jacobi sweep {it}: max rel change {m:.2e}  q99.9 {q99:.2e}')
            if m < tol or it == maxsweep:
                break
            u, un = un, u
        return u, hist

    def light(self, v, s, tag, solver_tol, maxsweep):
        """FN solve + a few Jacobi sweeps only (state needed to continue the loop, no diagnostics)."""
        S = self.S; t0 = time.time()
        efn, u, nmv = S.fn_solve(v, s, solver_tol, None, solver='lobpcg')
        a = np.maximum(v / S.sqrt_n, 1e-15)
        Dg = np.empty(S.D)
        C.k_fn_diag(S.indptr, S.indices, S.data, s, a * S.sqrt_n, Dg)
        E_loc = self.rayleigh(Dg, s, u)
        u, ph = self.polish(Dg, s, u, E_loc, maxsweep)
        u = np.maximum(u, 0.0); u /= np.linalg.norm(u)
        log(f'[{tag}] light step {time.time()-t0:.0f}s E_FN {efn:.10f} tail rel change {ph[0][1]:.1e} -> {ph[-1][1]:.1e}')
        return u

    def run(self, v, s, tag, solver_tol, maxsweep, trunc=True):
        S = self.S; N = self.N
        t0 = time.time()
        psi = v * s
        E_trial = float(psi @ S.H(psi) / (psi @ psi))
        # ---- FN solve (as in the loop)
        efn, u, nmv = S.fn_solve(v, s, solver_tol, None, solver='lobpcg')
        log(f'[{tag}] FN solve {time.time()-t0:.0f}s nmv {nmv} solver {S.last_solver} E_FN {efn:.10f}')
        a = np.maximum(v / S.sqrt_n, 1e-15)
        w = a * S.sqrt_n
        Dg = np.empty(S.D)
        C.k_fn_diag(S.indptr, S.indices, S.data, s, w, Dg)
        E_loc = self.rayleigh(Dg, s, u)
        u0 = u.copy()
        u, ph = self.polish(Dg, s, u, E_loc, maxsweep)
        u = np.maximum(u, 0.0); u /= np.linalg.norm(u)
        E_FN = self.rayleigh(Dg, s, u)
        Fu = self.Fmv(Dg, s, u)
        resid = float(np.linalg.norm(Fu - E_FN * u)); del Fu
        log(f'[{tag}] polished: E_FN {E_FN:.12f} (lobpcg {efn:.12f}), |res| {resid:.2e}, '
            f'tail rel change before {ph[0][1]:.2e} after {ph[-1][1]:.2e} ({len(ph)-1} sweeps)')
        # ---- gain
        E_Fa = self.rayleigh(Dg, s, w)
        G = (E_Fa - E_FN) / N
        ww = float(w @ w)
        lu = np.log(np.maximum(u, 1e-300)); lw = np.log(w)
        dl = lu - lw
        dl -= float(np.sum(u * u * dl))            # constant offset irrelevant; centre for conditioning
        rho = w / np.maximum(u, 1e-300)
        sel = u * u > 1e-10 * float(u.max()) ** 2
        mscale = float(np.median(rho[sel])) if sel.any() else 1.0
        rho = rho / mscale                          # conditioning only; undone below
        # per-configuration phi^2 and bins
        q = np.log10(np.maximum(u * u, 1e-300) / self.nrep)
        bn = np.clip(np.floor(-q * BPD), 0, NB - 1).astype(np.int32)
        nt = C.NTHR
        hist = np.zeros((nt, NCH, NB))
        gnorm = mscale ** 2 / ww / (N * G) * float(self.nrep.sum())       # rel density = 1 for a uniform spread of the gain
        k_hist(S.indptr, S.indices, S.data, s, u, dl, rho, bn, self.nrep, gnorm, hist)
        H = hist.sum(0); del hist
        H[10] /= float(self.nrep.sum())             # gain share (rel*n summed over bin / N_cfg == share of G)
        for ch in (4, 6, 8):                        # undo the rho scaling, divide by <w|w>
            H[ch] *= mscale ** 2 / ww
        Qq = H[3].sum() / N
        Gx = H[4].sum() / N
        log(f'[{tag}] G {G:.6e}/site, Q(delta) {Qq:.6e} (Q/G {Qq/G:.4f}), exact-decomposition sum {Gx:.6e} (/G {Gx/G:.8f})')
        # ---- second-order check: Q eps^2 vs exact gain of the guide phi*exp(-eps*delta), same F
        so = []
        for eps in (1.0, 0.5, 0.25, 0.1):
            Ge = (self.rayleigh(Dg, s, u * np.exp(-eps * dl)) - E_FN) / N
            so.append(dict(eps=eps, G_eps=Ge, Q_eps2_over_G=eps * eps * Qq / Ge))
        log(f'[{tag}] Q eps^2/G(eps): ' + ', '.join(f"{r['eps']}: {r['Q_eps2_over_G']:.4f}" for r in so))
        # ---- truncation (exact nonlinear) curves
        trunc_out = []
        if trunc:
            for tq in range(-3, -27, -self.trunc_step):
                bulk = q >= tq
                out = {'log10_thr': tq, 'n_bulk_orb': int(bulk.sum())}
                for name, mask in (('bulk', bulk), ('tail', ~bulk)):
                    x = np.where(mask, u, w)
                    out['frac_correct_' + name] = float((E_Fa - self.rayleigh(Dg, s, x)) / (E_Fa - E_FN))
                trunc_out.append(out)
            log(f'[{tag}] truncation curves done {time.time()-t0:.0f}s')
        # ---- accuracy and misc
        eps_trial = (E_trial - self.E0) / N if self.E0 is not None else None
        rec = dict(tag=tag, N=N, D=int(S.D), N_cfg=float(S.n_orb.sum()), E_trial_per_site_above_E0=eps_trial,
                   E_FN_per_site_above_E0=(E_FN - self.E0) / N if self.E0 is not None else None,
                   E_F_a_per_site_above_E0=(E_Fa - self.E0) / N if self.E0 is not None else None,
                   G=G, Q=Qq, G_exact_decomp=Gx, second_order=so, jacobi_hist=ph, resid_norm=resid,
                   lobpcg_vs_polished_max_rel=float(np.max(np.abs(u0 - u) / np.maximum(u, 1e-300))),
                   rms_delta=float(np.sqrt(np.sum(u * u * dl * dl))), max_abs_delta=float(np.max(np.abs(dl))),
                   hist=H.tolist(), bins_per_decade=BPD, channels=['norb', 'ncfg', 'mass', 'c_quad', 'c_exact',
                   'eq_min', 'ex_min', 'eq_max', 'ex_max', 'ncfg_by_gdens', 'gain_by_gdens'], truncation=trunc_out, sec=time.time() - t0)
        return rec, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster', required=True, choices=list(CLUSTERS))
    ap.add_argument('--kmax', type=int, default=3)
    ap.add_argument('--vit', action='store_true')
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--solver-tol', type=float, default=1e-10)
    ap.add_argument('--maxsweep', type=int, default=150)
    ap.add_argument('--gs-vec', default=None); ap.add_argument('--gs-states', default=None)
    ap.add_argument('--vit-tables', default='/project/theorie/a/A.Otaifi/chatty_stall6/data/sym_tables.npz')
    ap.add_argument('--vit-reps', default='/project/theorie/a/A.Otaifi/chatty_stall6/data/psi0_6x6_table.npz')
    ap.add_argument('--no-trunc', action='store_true')
    ap.add_argument('--diag-k', default=None, help='comma list of k with full diagnostics (default: all)')
    ap.add_argument('--trunc-step', type=int, default=1)
    ap.add_argument('--light-sweeps', type=int, default=10)
    args = ap.parse_args()
    L, torus = CLUSTERS[args.cluster]
    cluster = None; ctag = f'L{L}'
    if torus:
        import torus_cluster
        tv = [int(z) for z in torus.split(',')]
        cluster = torus_cluster.Cluster((tv[0], tv[1]), (tv[2], tv[3]))
        ctag = f"T{'_'.join(map(str, tv))}".replace('-', 'm')
    os.makedirs(args.cache_dir, exist_ok=True)
    log('threads', C.NTHR, vars(args))
    S = C.Sym(L, 0.5, log=log, cache=os.path.join(args.cache_dir, f'orb_{ctag}.npz'), cluster=cluster)
    E0, _ = C.target_gs(S, types.SimpleNamespace(gs_vec=args.gs_vec, gs_states=args.gs_states, ed_tol=1e-12), log)
    vinit, Einit = C.init_amplitude(S, 0.0, 1e-12, log, cache=os.path.join(args.cache_dir, f'init_{ctag}_J20.0.npz'))
    v = vinit / np.linalg.norm(vinit); s = C.canonical(S.marshall.copy())
    D = Diag(S, E0); D.trunc_step = args.trunc_step
    diag_k = None if args.diag_k is None else {int(z) for z in args.diag_k.split(',')}
    out = dict(cluster=args.cluster, N=S.N, D=int(S.D), E0=E0, E0_per_site=E0 / S.N, states=[])
    jp = lambda: json.dump(out, open(args.out, 'w'))
    for k in range(args.kmax + 1):
        if diag_k is None or k in diag_k:
            rec, u = D.run(v, s, f'loop k={k}', args.solver_tol, args.maxsweep, trunc=not args.no_trunc)
            rec['guide'] = f'loop_k{k}'; rec['k'] = k
            out['states'].append(rec); jp()
        else:
            u = D.light(v, s, f'loop k={k}', args.solver_tol, args.light_sweeps)
        if k < args.kmax:
            sn, _, G_, _ = S.krylov(u, s)
            log(f'  Krylov step -> k={k+1}: flips {int(np.sum(S.n_orb[sn != s]))} configs, groups {G_}')
            v, s = u, sn
    if args.vit:
        z = np.load(args.vit_tables); lP = z['lP']; sP = z['sP'].astype(np.int8)
        reps = np.load(args.vit_reps)['reps']
        assert np.array_equal(reps, S.states), 'ViT table ordering differs from the symmetric basis'
        v = np.exp(lP - lP.max()) * S.sqrt_n; v /= np.linalg.norm(v); s = sP
        rec, u = D.run(v, s, 'ViT psi_P', args.solver_tol, args.maxsweep, trunc=not args.no_trunc)
        rec['guide'] = 'vit'; rec['k'] = None
        out['states'].append(rec); jp()
    log('DONE', args.out)


if __name__ == '__main__':
    main()
