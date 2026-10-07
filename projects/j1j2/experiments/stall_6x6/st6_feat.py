"""Exactly optimisable symmetric correction family on top of a fixed base amplitude (6x6 J1-J2 sector).

  log a_c(x) = base(x) + sum_k c_k F_k(x),
  F_k(x) = sum_{m in orbit_k} prod_{i in m} sigma_i   (Ising cluster functions, summed over the full space group
           (translations x D4) -> exactly symmetric; even cluster sizes -> spin-flip even).
Classes: 'pair_all' (all 2-body distances, 9), 'w3b4' (4-site clusters in a 3x3 window, 19), 'w3b6' (16),
'w4b4' (4-site clusters in a 4x4 window, 140).
Optimisation is exact (no sampling): linear method in the K-dim tangent space with all matrices computed over the
full sector (K+1 matvecs per iteration), exact line search.  Objective: variational <H> at fixed signs s, or the
frozen lattice-FN energy <H_FN[a_g, s]> (whose unrestricted minimiser is phi_FN).
"""
import itertools, time
import numpy as np
import scipy.linalg
import jax
import jax.numpy as jnp
import st6_sector as SS

L = 6; N = 36


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


def _site(x, y):
    return (x % L) + L * (y % L)


def cluster_classes(kind):
    G = SS.space_group()
    if kind == 'pair_all':
        subs = [(0, j) for j in range(1, N)]
    else:
        W = int(kind[1]); body = int(kind[3:])
        win = [_site(x, y) for y in range(W) for x in range(W)]
        subs = [s for s in itertools.combinations(win, body) if 0 in s]
    seen = set(); out = []
    for sub in subs:
        o = frozenset(tuple(sorted(G[g][list(sub)])) for g in range(len(G)))
        if o in seen: continue
        seen.add(o)
        masks = np.array([sum(1 << i for i in m) for m in o], np.uint64)
        out.append((f'{kind}:{sorted(sub)}', masks))
    return out


@jax.jit
def _feat_chunk(x, masks):
    pc = jax.lax.population_count(x[:, None] & masks[None, :]) & jnp.uint64(1)
    return jnp.sum(1.0 - 2.0 * pc.astype(jnp.float32), axis=1)


def build_features(sec, kinds, chunk=1 << 20):
    names, rows = [], []
    t0 = time.time()
    for kind in kinds:
        for name, masks in cluster_classes(kind):
            m = jnp.asarray(masks)
            f = jnp.concatenate([_feat_chunk(sec.reps[i:i + chunk], m) for i in range(0, sec.D, chunk)])
            names.append(name); rows.append(f)
    # standardise under psi0^2 (conditioning only; the family is unchanged).  Kept as a list of (D,) rows: no
    # (K, D) matrix on the device (XLA transposes/copies of it exhausted the 11 GB card).
    p0 = sec.p0
    F = []
    for f in rows:
        f64 = f.astype(jnp.float64); mu = float(f64 @ p0); sd = float(jnp.sqrt(jnp.maximum((f64 * f64) @ p0 - mu * mu, 1e-12)))
        F.append(((f64 - mu) / sd).astype(jnp.float16)); del f64   # float16 storage: the family is defined by
        # the stored (rounded) features, used exactly in float64 afterwards -> still exactly symmetric
    del rows
    log(f'[feat] {len(names)} features ({kinds}) in {time.time()-t0:.0f}s')
    return F, names


def zz_diag(sec, chunk=1 << 18):
    """physical diagonal (sum_bonds J S^z S^z) per rep."""
    BI = jnp.asarray(SS.BI); BJ = jnp.asarray(SS.BJ); JB = jnp.asarray(SS.JB)
    out = []
    for i in range(0, sec.D, chunk):
        x = sec.reps[i:i + chunk]
        par = (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(jnp.float64)
        out.append(0.25 * ((1.0 - 2.0 * par) @ JB))
    return jnp.concatenate(out)


def hop_features(sec, la_g, s, which=('viol', 'allow', 'viol2', 'allow2', 'cross', 'logviol', 'logallow')):
    """one-hop features of a guide (a_g, s): violating / allowed off-diagonal sums of amplitude ratios,
       viol(x) = sum_{y: s_x s_y H_xy > 0} H_xy a(y)/a(x),  allow(x) = sum_{y: s_x s_y H_xy < 0} H_xy a(y)/a(x)
    (E_L^FN = zz + viol - allow' for b = a).  Same normalisation as fn_solve.  Standardised, float16 storage."""
    la = la_g - jnp.max(la_g)
    a = jnp.exp(la); a = a / jnp.sqrt(jnp.sum(sec.n * a * a)); a = jnp.maximum(a, 1e-15)
    w = a * sec.sqrt_n; del a
    h1, h2 = sec.H2(w, s * w)
    zz = zz_diag(sec)
    viol = 0.5 * (h1 + s * h2) / w - zz
    allow = 0.5 * (h1 - s * h2) / w
    del h1, h2, w
    vc = jnp.clip(viol, 0.0, 20.0); ac = jnp.clip(allow, 0.0, 40.0)    # ratios are heavy-tailed on tiny-|a| configs
    raw = dict(viol=vc, allow=ac, viol2=vc * vc, allow2=ac * ac, cross=vc * ac,
               logviol=jnp.log(jnp.maximum(viol, 0) + 1e-3), logallow=jnp.log(jnp.maximum(allow, 0) + 1e-3), zz=zz)
    F, names = [], []
    for nm in which:
        f64 = raw[nm]; mu = float(f64 @ sec.p0); sd = float(jnp.sqrt(jnp.maximum((f64 * f64) @ sec.p0 - mu * mu, 1e-12)))
        F.append(((f64 - mu) / sd).astype(jnp.float32)); names.append('hop:' + nm)
    del raw, viol, allow, zz
    return F, names


class Family:
    def __init__(self, sec, base, F):
        self.sec = sec; self.base = base; self.F = F; self.K = len(F)

    def la(self, c):
        # sum_k c_k F_k as K axpys (c @ F would make XLA transpose the whole (K, D) matrix -> a 2.8 GB copy)
        out = self.base
        for k, ck in enumerate(np.asarray(c, np.float64)):
            if ck != 0.0: out = out + ck * self.F[k].astype(jnp.float64)
        return out

    def objective_op(self, mode, s, la_g=None):
        sec = self.sec
        if mode == 'var':
            return (lambda la: sec.vec(la, s)), sec.H
        A = sec.fn_op(la_g, s)
        return (lambda la: sec.vec(la, jnp.ones_like(la))), A

    def energy(self, c, mode, s, la_g=None, ops=None):
        mk, Hop = ops or self.objective_op(mode, s, la_g)
        v = mk(self.la(c))
        return float(v @ Hop(v))

    def optimize(self, c0, mode, s, la_g=None, iters=10, shift=1e-3, tol=1e-9, ts=(1.0, 0.5, 0.25, 0.1)):
        """linear method; returns c, history of exact objective values (total energy, not per site)."""
        sec = self.sec; F = self.F; K = self.K
        mk, Hop = self.objective_op(mode, s, la_g)
        c = np.asarray(c0, np.float64).copy()
        v = mk(self.la(c)); Hv = Hop(v); E = float(v @ Hv)
        hist = [E]
        for it in range(iters):
            p = (v * v)
            Fdot = lambda x: np.array([float(Fk.astype(jnp.float64) @ x) for Fk in F])   # F @ x
            m = Fdot(p)                                                        # <F_k>
            S = np.zeros((K, K))
            for k in range(K):
                Fpk = F[k].astype(jnp.float64) * p
                for l in range(k, K):
                    S[k, l] = S[l, k] = float(Fpk @ F[l].astype(jnp.float64))
                del Fpk
            S -= np.outer(m, m)
            b = Fdot(v * Hv) - m * E                                           # <V_k|H|v>
            Hk = np.zeros((K, K))
            for l in range(K):
                Vl = (F[l].astype(jnp.float64) - m[l]) * v
                HVl = Hop(Vl)
                Hk[:, l] = Fdot(v * HVl) - m * float(v @ HVl)
                del Vl, HVl
            Hk = 0.5 * (Hk + Hk.T)
            Hb = np.zeros((K + 1, K + 1)); Sb = np.zeros((K + 1, K + 1))
            Hb[0, 0] = E; Hb[0, 1:] = b; Hb[1:, 0] = b; Hb[1:, 1:] = Hk + shift * np.diag(np.diag(S))
            Sb[0, 0] = 1.0; Sb[1:, 1:] = S + 1e-12 * np.eye(K)
            w, X = scipy.linalg.eigh(Hb, Sb)
            x = X[:, 0]
            d = x[1:] / x[0]
            best = (E, None)
            for t in ts:
                ct = c + t * d
                vt = mk(self.la(ct)); Et = float(vt @ Hop(vt))
                if Et < best[0]: best = (Et, ct)
                if best[1] is not None and t < 1.0: break
            if best[1] is None:
                shift *= 10; hist.append(E)
                if shift > 1e3: break
                continue
            dE = E - best[0]
            c = best[1]; v = mk(self.la(c)); Hv = Hop(v); E = float(v @ Hv)
            hist.append(E)
            if dE < tol: break
        return c, hist
