"""Exact projections of one FN iteration (no training), 6x6 sector.

1. Bulk / tail restriction: best arbitrary f supported on a set S (f = 0 elsewhere), quadratic optimum
   L_SS f = (L delta)_S solved by Jacobi-preconditioned CG; captured share of Q(delta) and the exact frozen-FN frac.
2. Linear structured bases: piecewise-constant functions of guide-local quantities (log a, violating weight V,
   kept-edge weight W). The maximal quadratic capture is b^T G^+ b with G the coarse-grained FN Laplacian
   (allowed edges summed between groups) and b the group sums of L delta. Exact.
3. Proposal diagnostics: effective sample size of the importance weights p/q for the old and tail proposals.

  python wt_exact.py OUT_DIR
"""
import json, os, sys, time
from functools import partial
import numpy as np
from wt_common import Setup, log, dump, N, f64, BULK_CUT
import jax
import jax.numpy as jnp
import scipy.linalg

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
T0 = time.time()
S = Setup(); sec = S.sec; u = S.u
res = dict(G0=S.G0, Qdelta=S.Qdelta, E_FN_dE_site=(S.Efn - sec.E0) / N, fn_info=S.fn_info)
res['delta_gain_decades'] = S.decades(S.c_delta)
res['delta_gain_bulk_tail'] = S.bulk_tail(S.c_delta)
res['mass_bulk'] = float(jnp.sum(jnp.where(S.bulk, S.p, 0)))
log('gain split', res['delta_gain_bulk_tail'], 'bulk mass', res['mass_bulk'])
Ld = S.Lop(S.delta)
log('check Q(delta) via L:', float(S.delta @ Ld) / N, 'node sum:', S.Qdelta)
F = lambda: os.path.join(OUT, 'wt_exact.json')


def exact_frac(f):
    ef = sec.fn_rayleigh(S.lP, S.sP, S.lP + f)
    return (S.E_f0 - ef) / (S.E_f0 - S.Efn)


def quad_frac(f):
    return 1.0 - S.Q(S.delta - f) / S.Qdelta


# ============================================================================ 1. restricted optimum (CG)
def pcg(mask, maxit=1500, rtol=1e-7, tag=''):
    m = mask.astype(f64)
    b = m * Ld
    Dg = jnp.maximum(u * S.Ku, 1e-300)
    x = jnp.zeros_like(b); r = b; z = m * r / Dg; pd = z; rz = float(r @ z); rz0 = rz
    hist = []
    for it in range(1, maxit + 1):
        Ap = m * S.Lop(m * pd)
        al = rz / float(pd @ Ap)
        x = x + al * pd; r = r - al * Ap
        z = m * r / Dg; rzn = float(r @ z)
        pd = z + (rzn / rz) * pd; rz = rzn
        if it % 25 == 0 or it == 1:
            cap = float(x @ b) / N / S.Qdelta              # at the CG iterate: approx 2 x.b - x.Lx -> x.b at optimum
            hist.append((it, cap, (rz / rz0) ** 0.5))
            log(f'  cg {tag} it {it} capture~{cap:.4f} rel.res {(rz / rz0) ** 0.5:.2e}')
        if (rz / rz0) ** 0.5 < rtol: break
    return x, it, (rz / rz0) ** 0.5, hist


res['restricted'] = []
cuts = [('full', None, None), ('bulk>=1e-9', 1e-9, True), ('bulk>=1e-10', 1e-10, True), ('bulk>=1e-11', 1e-11, True),
        ('bulk>=1e-12', 1e-12, True), ('tail<1e-10', 1e-10, False), ('tail<1e-9', 1e-9, False)]
for tag, cut, isbulk in cuts:
    t0 = time.time()
    if cut is None:
        mask = jnp.ones_like(S.bulk)
    else:
        mask = (S.lpc >= np.log(cut)) if isbulk else (S.lpc < np.log(cut))
    f, it, rr, hist = pcg(mask, tag=tag)
    qf = quad_frac(f); ef = exact_frac(f)
    cf = S.node_c(S.delta - f)
    r = dict(set=tag, mass=float(jnp.sum(jnp.where(mask, S.p, 0))), reps=int(jnp.sum(mask)), cg_it=it, cg_relres=rr,
             quad_frac=qf, exact_frac=ef, residual_decades=S.decades(cf, tot=S.Qdelta * N),
             captured_decades=S.decades(S.c_delta - cf, tot=S.Qdelta * N), cg_hist=hist, sec=time.time() - t0)
    res['restricted'].append(r)
    log(f'RESTRICTED {tag}: mass {r["mass"]:.4f} quad {qf:.4f} exact {ef:.4f} it {it} relres {rr:.1e}')
    dump(F(), res)
    del f, cf


# ============================================================================ 2. piecewise-constant guide bases
@partial(jax.jit, static_argnums=(9,), donate_argnums=(0,))
def _coarse_chunk(M, rbase, inc, idx, cod, gid, uu, sq, s, ng):
    row = rbase + jnp.cumsum(inc.astype(jnp.int32))
    A = cod.astype(f64) * 0.125 * sq[row] / sq[idx] * uu[row] * uu[idx]
    A = jnp.where((idx != row) & (s[row] * s[idx] < 0), A, 0.0)
    gi = gid[row]; gj = gid[idx]
    M = M.at[gi * ng + gj].add(A)
    return M.at[gj * ng + gi].add(A)


def coarse_projection(gid, ng, tag):
    t0 = time.time()
    sec.reload()
    M = jnp.zeros(ng * ng, f64)
    for (rb, inc, idx, cod) in sec.chunks:
        M = _coarse_chunk(M, rb, inc, idx, cod, gid, u, sec.sqrt_n, S.sP, ng)
        M.block_until_ready()
    M = np.asarray(M).reshape(ng, ng)
    G = np.diag(M.sum(1)) - M
    b = np.asarray(jax.ops.segment_sum(Ld, gid, ng))
    keep = (np.diag(G) > 0) | (np.abs(b) > 0)
    G = G[np.ix_(keep, keep)]; bb = b[keep]
    sc = 1.0 / np.sqrt(np.maximum(np.diag(G), 1e-300))           # symmetric Jacobi scaling for the eigensolve
    lam, V = scipy.linalg.eigh(sc[:, None] * G * sc[None, :])
    ok = lam > 1e-12 * lam.max()
    bt = sc * bb
    c = sc * (V[:, ok] @ ((V[:, ok].T @ bt) / lam[ok]))
    cap = float(bb @ c) / N / S.Qdelta
    cf = np.zeros(ng); cf[keep] = c
    f = jnp.asarray(cf)[gid]
    qf = quad_frac(f); ef = exact_frac(f)
    r = dict(basis=tag, groups=int(keep.sum()), capture_formula=cap, quad_frac=qf, exact_frac=ef,
             sec=time.time() - t0)
    log(f'BASIS {tag}: groups {r["groups"]} capture {cap:.4f} quad {qf:.4f} exact {ef:.4f}')
    return r


def wq_bins(feat, m, nb):
    order = jnp.argsort(feat)
    cm = jnp.cumsum(m[order]); cm = cm / cm[-1]
    bins = jnp.clip(jnp.floor(cm * nb - 1e-12), 0, nb - 1).astype(jnp.int32)
    return jnp.zeros(feat.shape[0], jnp.int32).at[order].set(bins)


V, W = S.guide_features()
res['guide_features'] = dict(V_mean_phi2=float(jnp.sum(S.p * V)), W_mean_phi2=float(jnp.sum(S.p * W)),
                             corr_logV_delta_c=None)
lV = jnp.log(V + 1e-6); lW = jnp.log(W + 1e-6)
meas = 0.5 * S.p + 0.5 * S.c_delta / jnp.sum(S.c_delta)        # bins resolve both the mass and the gain
res['bases'] = []
b_a64 = wq_bins(S.lP, meas, 64)
b_a256 = wq_bins(S.lP, meas, 256)
for tag, gid, ng in [('log a, 64 bins', b_a64, 64), ('log a, 256 bins', b_a256, 256),
                     ('(log a, log V), 32x32', wq_bins(S.lP, meas, 32) * 32 + wq_bins(lV, meas, 32), 1024),
                     ('(log a, log W), 32x32', wq_bins(S.lP, meas, 32) * 32 + wq_bins(lW, meas, 32), 1024),
                     ('(log a, log V, log W), 16^3',
                      (wq_bins(S.lP, meas, 16) * 16 + wq_bins(lV, meas, 16)) * 16 + wq_bins(lW, meas, 16), 4096),
                     ('delta itself, 64 bins (oracle reference)', wq_bins(S.delta, meas, 64), 64)]:
    res['bases'].append(coarse_projection(gid, ng, tag))
    dump(F(), res)

# ============================================================================ 3. proposal diagnostics
def ess(q):                                  # ESS / n of importance weights p/q under q (rep-level)
    q = q / jnp.sum(q)
    w = S.p / jnp.maximum(q, 1e-300)
    return float(1.0 / jnp.sum(q * w * w))


n = sec.n
old = n * jnp.exp(0.5 * (S.lpc - jnp.max(S.lpc)))
old = old / jnp.sum(old)
tail = S.c_delta / jnp.sum(S.c_delta)
res['proposals'] = dict(
    ess_phi2=1.0, ess_old_beta05=ess(old), ess_mix=ess(0.5 * old + 0.5 * tail),
    tail_share_samples_old=float(jnp.sum(jnp.where(S.bulk, 0, old))),
    tail_share_samples_mix=float(jnp.sum(jnp.where(S.bulk, 0, 0.5 * old + 0.5 * tail))),
    tail_share_samples_phi2=float(jnp.sum(jnp.where(S.bulk, 0, S.p))),
    gain_share_tail=res['delta_gain_bulk_tail']['tail'])
log('PROPOSALS', res['proposals'])
res['sec'] = time.time() - T0
dump(F(), res)
log('DONE', res['sec'])
