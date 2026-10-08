"""Amendment 4 of experiments/itfit_6x6: exact projection of the ideal FN loop's updates onto functions of one-hop
features of the frozen base |psi_P| (and, for reference, of the current guide).  No training, float64, one GPU.

Energy metric of iteration k: Q_k(e) = e.L_k e / N with L_k e = u e (K u) - u K(u e) (FN Laplacian at u = phi_k, kept
hops of s_k).  Galerkin projection: f = argmin_{f in family} Q_k(delta_k - f).  Families: smooth basis (b) and
piecewise-constant bins (a).  Exact score: frac = (E_f,k[a_k] - E_f,k[a_k exp(f)]) / G_k.
  python itfit_feat.py OUT [SPEC.json]
"""
import json, os, sys, time
from functools import partial
import numpy as np
from itfit_common import Exact, log, dump, N, f64, decade_table, E0_SITE
import jax
import jax.numpy as jnp

OUT = sys.argv[1]
SPEC = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
os.makedirs(OUT, exist_ok=True)
F_OUT = os.path.join(OUT, 'itfit_feat.json')
T00 = time.time()
ex = Exact(); sec = ex.sec
lP, sP = ex.lP, ex.sP
Hd = ex.Hdiag
sqn = ex.sqn
res = dict(spec=SPEC, iters=[])
es = lambda E: E / N - E0_SITE


def ops(X, s):
    """same-sign (incl. diagonal) and kept parts of H on sector vectors X (D, k)."""
    sm, kp = [], []
    for j in range(X.shape[1]):                     # one column at a time (11 GB cards)
        Y = sec.Hm(jnp.stack([X[:, j], s * X[:, j]], 1))
        sm.append(0.5 * (Y[:, 0] + s * Y[:, 1])); kp.append(0.5 * (Y[:, 0] - s * Y[:, 1])); del Y
    return jnp.stack(sm, 1), jnp.stack(kp, 1)


def features(la, s):
    """one-hop features of the guide (la, s), per configuration."""
    la = la - jnp.max(la)
    a = jnp.maximum(jnp.exp(la), 1e-300)
    w = a * sqn
    same, kept = ops(jnp.stack([w, a * a * sqn, la * sqn, sqn], 1), s)
    F = dict(la=la, Hd=Hd)
    F['W'] = kept[:, 0] / w; F['V'] = same[:, 0] / w - Hd
    F['W2'] = kept[:, 1] / (a * a * sqn); F['V2'] = same[:, 1] / (a * a * sqn) - Hd
    F['nK'] = kept[:, 3] / sqn; F['nV'] = same[:, 3] / sqn - Hd
    F['LK'] = kept[:, 2] / sqn - la * F['nK']; F['LV'] = (same[:, 2] / sqn - Hd * la) - la * F['nV']
    return F


def standardise(x, p):
    m = float(jnp.sum(p * x)); sd = float(jnp.sqrt(jnp.sum(p * (x - m) ** 2))) + 1e-30
    return ((x - m) / sd)


def basis(F, p, E, zc=6.0):
    """smooth family (b) as callables (memory-light): standardised transforms, squares, pairwise products of the main
    six, T(tau) functions.  Each callable returns a (D,) float64 column."""
    lg = lambda x: jnp.log(jnp.maximum(x, 0.0) + 1e-6)
    raw = dict(la=lambda: F['la'], Hd=lambda: F['Hd'], W=lambda: lg(F['W']), V=lambda: lg(F['V']), nK=lambda: F['nK'],
               nV=lambda: F['nV'], W2=lambda: lg(F['W2']), V2=lambda: lg(F['V2']), LK=lambda: F['LK'], LV=lambda: F['LV'])
    st = {}
    for k, fn in raw.items():
        x = fn(); m = float(jnp.sum(p * x)); sd = float(jnp.sqrt(jnp.sum(p * (x - m) ** 2))) + 1e-30; st[k] = (m, sd)
    z = {k: (lambda k=k, fn=fn: jnp.clip((fn() - st[k][0]) / st[k][1], -zc, zc)) for k, fn in raw.items()}
    cols, names = [], []
    for k in raw:
        cols.append(z[k]); names.append(k)
        cols.append(lambda k=k: z[k]() ** 2); names.append(k + '^2')
    main = ['la', 'Hd', 'W', 'V', 'nK', 'nV']
    for i in range(len(main)):
        for j in range(i + 1, len(main)):
            cols.append(lambda a=main[i], b=main[j]: z[a]() * z[b]()); names.append(main[i] + '*' + main[j])
    for tau in SPEC.get('taus', [0.1, 0.3, 1.0, 3.0, 10.0, 100.0]):
        def t(tau=tau):
            return jnp.log1p(tau * jnp.maximum(F['W'], 0)) - jnp.log1p(tau * jnp.maximum(F['Hd'] + F['V'] - E, 1e-12))
        x = t(); m = float(jnp.sum(p * x)); sd = float(jnp.sqrt(jnp.sum(p * (x - m) ** 2))) + 1e-30
        cols.append(lambda t=t, m=m, sd=sd: jnp.clip((t() - m) / sd, -zc, zc)); names.append(f'T{tau}')
    return cols, names


def qbins(x, p, nb):
    o = jnp.argsort(x); cm = jnp.cumsum(p[o]); cm = cm / cm[-1]
    return jnp.zeros(x.shape[0], jnp.int32).at[o].set(jnp.clip(jnp.floor(cm * nb - 1e-12), 0, nb - 1).astype(jnp.int32))


@partial(jax.jit, static_argnums=(9,), donate_argnums=(0,))
def _lap_chunk(M, rb, inc, idx, cod, gid, u, s, sq, G):
    row = rb + jnp.cumsum(inc.astype(jnp.int32))
    w = cod.astype(f64) * 0.125 * sq[row] / sq[idx] * u[row] * u[idx]
    w = jnp.where((idx != row) & (s[row] * s[idx] < 0), w, 0.0)
    gr = gid[row]; gi = gid[idx]
    M = M.at[gr * G + gr].add(w).at[gi * G + gi].add(w)
    return M.at[gr * G + gi].add(-w).at[gi * G + gr].add(-w)


@jax.jit
def _coldots(Phi, v):
    return jax.lax.map(lambda row: row.astype(f64) @ v, Phi)


@jax.jit
def _combine(Phi, c):
    return jax.lax.fori_loop(0, Phi.shape[0], lambda i, acc: acc + c[i] * Phi[i].astype(f64), jnp.zeros(Phi.shape[1], f64))


class Proj:
    def __init__(self, u, s, delta, Dk, lak, Eg, Efn):
        self.u, self.s, self.delta, self.Dk, self.lak, self.Eg, self.Efn = u, s, delta, Dk, lak, Eg, Efn
        self.Ku = ex.Kop(u, s)
        self.Ld = self.L(delta)
        self.Qd = float(delta @ self.Ld)

    def L(self, e):
        return self.u * e * self.Ku - self.u * ex.Kop(self.u * e, self.s)

    def score(self, f, tag, extra=None):
        b = ex.pos_vec(self.lak + f)
        E, _ = ex.Ef(b, self.Dk, self.s)
        e = self.delta - f
        Qe = float(e @ self.L(e))
        out = dict(family=tag, frac=(self.Eg - E) / (self.Eg - self.Efn), quad_frac=1 - Qe / self.Qd)
        if extra: out.update(extra)
        log(f'    {tag}: frac {out["frac"]:.4f} quad {out["quad_frac"]:.4f}')
        return out

    def smooth(self, cols, names, tag):
        J = len(cols); M = np.zeros((J, J)); r = np.zeros(J)
        for j in range(J):
            cj = cols[j]()
            Lc = self.L(cj); r[j] = float(cj @ self.Ld); del cj
            for i in range(J):
                M[i, j] = float(cols[i]() @ Lc)
            del Lc
        M = 0.5 * (M + M.T)
        c = np.linalg.lstsq(M + 1e-12 * np.trace(M) / J * np.eye(J), r, rcond=None)[0]
        f = jnp.zeros_like(self.delta)
        for i in range(J): f = f + float(c[i]) * cols[i]()
        lo, hi = float(jnp.min(self.delta)), float(jnp.max(self.delta))
        mu = float(jnp.sum(self.u * self.u * (f - self.delta)))
        fc = jnp.clip(f - mu, lo, hi) + mu                   # f bounded by the range of the target itself
        sc = self.score(fc, tag, dict(J=J, names=names, frac_unclipped=None))
        return sc, fc

    def binned(self, gid, G, tag):
        M = jnp.zeros(G * G, f64)
        for (rb, inc, idx, cod) in sec.chunks:
            M = _lap_chunk(M, rb, inc, idx, cod, gid, self.u, self.s.astype(f64), sqn, G)
        M = np.asarray(M).reshape(G, G)
        r = np.asarray(jax.ops.segment_sum(self.Ld, gid, G))
        used = np.where(np.abs(np.diag(M)) > 0)[0]
        c = np.zeros(G)
        Mu = M[np.ix_(used, used)]
        c[used] = np.linalg.lstsq(Mu + 1e-12 * np.trace(Mu) / len(used) * np.eye(len(used)), r[used], rcond=None)[0]
        f = jnp.asarray(c)[gid]
        return self.score(f, tag, dict(groups=int(len(used)))), f

    def missing(self, f, pc, wall):
        e = self.delta - f
        ce = ex.node_c(e, self.u, self.Ku, self.s); cd = ex.node_c(self.delta, self.u, self.Ku, self.s)
        return dict(resid_phi2_decades=decade_table(pc, dict(resid=ce, gain=cd), lo=-13, hi=-5),
                    resid_wall_decades=decade_table(jnp.maximum(wall, 1e-300), dict(resid=ce, gain=cd), lo=-1, hi=3))


def families(P, Fb, Fc, pmeas, E, bmeas=None):
    bm = pmeas if bmeas is None else bmeas
    """all families for one target; returns list of scores and the best base-family f."""
    out = []; best = (-np.inf, None, None)
    for zc in SPEC.get('zclips', [6.0, 1e9]):
        cb, nb = basis(Fb, pmeas, E, zc)
        sc, f = P.smooth(cb, nb, f'base: smooth basis (b), feature clip {zc:g}'); out.append(sc)
        if sc['frac'] > best[0]: best = (sc['frac'], f, sc['family'])
        del f
    nbin = SPEC.get('nbin4', 8)
    lg = lambda x: jnp.log(jnp.maximum(x, 0.0) + 1e-6)
    g4 = qbins(Fb['la'], bm, nbin)
    for k_ in (lg(Fb['W']), lg(Fb['V']), Fb['Hd']):
        g4 = g4 * nbin + qbins(k_, bm, nbin)
    sc, f = P.binned(g4, nbin ** 4, f'base: bins (log a, W, V, H_xx) {nbin}^4'); out.append(sc)
    if sc['frac'] > best[0]: best = (sc['frac'], f, sc['family'])
    T1 = jnp.log1p(jnp.maximum(Fb['W'], 0)) - jnp.log1p(jnp.maximum(Fb['Hd'] + Fb['V'] - E, 1e-12))
    sc, f = P.binned(qbins(T1, bm, 256), 256, 'base: bins T(tau=1) 256'); out.append(sc)
    if sc['frac'] > best[0]: best = (sc['frac'], f, sc['family'])
    g2 = qbins(T1, bm, 64) * 64 + qbins(Fb['la'], bm, 64)
    sc, f = P.binned(g2, 4096, 'base: bins (T, log a) 64^2'); out.append(sc)
    if sc['frac'] > best[0]: best = (sc['frac'], f, sc['family'])
    if Fc is not None:
        cc, nc = basis(Fc, pmeas, E, 1e9)
        sc, _ = P.smooth(cc, nc, 'current guide: smooth basis (b)'); out.append(sc)
        T1c = jnp.log1p(jnp.maximum(Fc['W'], 0)) - jnp.log1p(jnp.maximum(Fc['Hd'] + Fc['V'] - E, 1e-12))
        g2c = qbins(T1c, bm, 64) * 64 + qbins(Fc['la'], bm, 64)
        sc, _ = P.binned(g2c, 4096, 'current guide: bins (T, log a) 64^2'); out.append(sc)
    return out, best


# ------------------------------------------------------------------ base features (fixed)
Efn1, u1, _ = sec.fn_solve(lP, sP)
pmeas = u1 * u1
Ku1 = ex.Kop(u1, sP); cd1 = ex.node_c(jnp.log(jnp.maximum(u1 / sqn, 1e-300)) - lP, u1, Ku1, sP); del Ku1
bmeas = 0.5 * pmeas + 0.5 * cd1 / jnp.sum(cd1); del cd1
Fb = features(lP, sP)
wall_b = Fb['V']
log('base features done', time.time() - T00)

# ------------------------------------------------------------------ ideal loop, per-iteration targets
la, s = lP, sP
for k in range(SPEC.get('n_iter', 3)):
    t0 = time.time()
    Efn, u, info = sec.fn_solve(la, s)
    Dk, w, Kw = ex.fn_diag(la, s)
    b = w / jnp.linalg.norm(w); Eg, _ = ex.Ef(b, Dk, s, Kw / jnp.linalg.norm(w)); del w, Kw, b
    lphi = jnp.log(jnp.maximum(u / sqn, 1e-300))
    delta = lphi - la
    P = Proj(u, s, delta, Dk, la, Eg, Efn)
    Fc = features(la, s) if k > 0 else None
    log(f'== ideal iteration {k + 1}: E_FN {es(Efn):.4e} G {(Eg - Efn) / N:.4e} Q(delta) {P.Qd / N:.4e}')
    out, best = families(P, Fb, Fc, pmeas, Efn, bmeas)
    pc = jnp.maximum(u * u / sec.n, 1e-300)
    rec = dict(it=k + 1, E_FN_dE_site=es(Efn), G_site=(Eg - Efn) / N, Qdelta_site=P.Qd / N, families=out,
               best_base=best[2], best_base_frac=best[0], missing_best_base=P.missing(best[1], pc, wall_b), sec=time.time() - t0)
    res['iters'].append(rec); dump(F_OUT, res)
    sn, _, _ = sec.krylov(u, s)
    la, s = lphi, sn
    del P, Fc, Dk, u, delta, lphi

# ------------------------------------------------------------------ projected loop with the base smooth family (b)
if SPEC.get('projected_loop', True):
    res['projected_loop'] = []
    la, s = lP, sP
    for k in range(SPEC.get('n_iter', 3)):
        t0 = time.time()
        Efn, u, _ = sec.fn_solve(la, s)
        Dk, w, Kw = ex.fn_diag(la, s)
        b = w / jnp.linalg.norm(w); Eg, _ = ex.Ef(b, Dk, s, Kw / jnp.linalg.norm(w)); del w, Kw, b
        delta = jnp.log(jnp.maximum(u / sqn, 1e-300)) - la
        P = Proj(u, s, delta, Dk, la, Eg, Efn)
        cb, nb = basis(Fb, pmeas, Efn, SPEC.get('loop_zclip', 1e9))
        sc, f = P.smooth(cb, nb, f'projected loop it {k + 1}: base smooth basis')
        la_new = la + f
        Hold = ex.energy_H(la_new, s)
        sn, Ek, _ = sec.krylov(ex.pos_vec(la_new), s)
        sc.update(it=k + 1, E_FN_guide_dE_site=es(Efn), H_oldsign_dE_site=es(Hold), H_kry_dE_site=es(Ek), sec=time.time() - t0)
        res['projected_loop'].append(sc); dump(F_OUT, res)
        log(f'  projected loop it {k + 1}: E_FN guide {es(Efn):.4e} frac {sc["frac"]:.4f} <H>(Krylov) {es(Ek):.4e}')
        la, s = la_new, sn
        del P, Dk, u, delta, f
    Efn, _, _ = sec.fn_solve(la, s)
    res['projected_loop_final_E_FN_dE_site'] = es(Efn)
res['sec'] = time.time() - T00
dump(F_OUT, res)
log('DONE', res['sec'])
