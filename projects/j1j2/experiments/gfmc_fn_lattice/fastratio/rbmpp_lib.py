"""RBM x pair-product (RBM+PP) wave function for the J1-J2 Heisenberg model on an L x L torus (JAX).

Ansatz (Nomura & Imada, PRX 11, 031034 (2021), arXiv:2005.14142, Sec. II A and App. A):
  Psi(sigma) = phi_RBM(sigma) psi_PP(sigma),   sigma_i = +-1 (= 2 S^z_i), |sigma> = prod_i c^dag_{i sigma_i}|0>
  phi_RBM(sigma) = prod_k 2 cosh(b_k + sum_i W_ik sigma_i),  M = 16 hidden units, b, W complex, no symmetry imposed
  psi_PP(sigma) = <sigma| P_G (sum_ij f_ij c^dag_{i up} c^dag_{j dn})^(N/2) |0>,  f real, full N x N (6x6: no sublattice)
               = eps(sigma) det f[U, D]   (U, D = up / down sites in increasing order;
                 eps = (-1)^{#(down site j, up site i) with j < i}, the reordering sign to the site-ordered string)
Projection (spin language, trivial characters = sector k = 0, A1 / A, spin-parity even):
  'simple' (published, Eq. (5) + (A2)): sum over 36 translations x {E, R_pi/2} x {sigma, -sigma}   (144 images)
  'full'  : the whole space group (288) x spin flip                                               (576 images)
  'translations' : the N translations x spin flip only (2N images; 8x8: 128), no point group (cheap walkers)
  'none'  : no projection.
psi_P(sigma) = sum_g Psi(z_g),  z_g = flip_g * sigma[IMG_g].   Variational parameters theta are REAL
(f, Re b, Im b, Re W, Im W); O_k = d log psi_P / d theta_k is complex.

Local energy with fast updates: for a bond swap (p up, q down in image z) the determinant ratio follows from the
rank-2 update of F = f[U, D] (row of p replaced by q, column of q replaced by p), with G = F^-1, R = f[:, D] G:
  det F'/det F = R[q,a] (1 + (G v)_b) - G[b,a] (R[q,:] . v - v_a),   v_k = f[U_k,p] - f[U_k,q] (k != a),
  v_a = f[q,p] - f[q,q];  psi_PP(z')/psi_PP(z) = - det F'/det F  (eps ratio (-1)^{|q-p|} x re-sort sign
  (-1)^{|q-p|-1}).  Checked against brute force in rbmpp_test.py.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------------------------------ lattice
class Lattice:
    def __init__(self, L, J2=0.5, proj='simple'):
        self.L = L; self.N = N = L * L; self.J2 = J2
        site = lambda x, y: (x % L) + L * (y % L)
        nn_, nnn = [], []
        for y in range(L):
            for x in range(L):
                i = site(x, y)
                nn_ += [(i, site(x + 1, y)), (i, site(x, y + 1))]
                nnn += [(i, site(x + 1, y + 1)), (i, site(x + 1, y - 1))]
        bl = nn_ + nnn
        self.BI = np.array([b[0] for b in bl], np.int32); self.BJ = np.array([b[1] for b in bl], np.int32)
        self.JB = np.array([1.0] * len(nn_) + [J2] * len(nnn))
        self.NB = len(bl)
        self.G = self.space_group()                         # (8 L^2, N), same order as st6_sector.space_group
        self.set_proj(proj)

    def space_group(self):
        L, N = self.L, self.N
        site = lambda x, y: (x % L) + L * (y % L)

        def pm(f):
            p = np.empty(N, np.int64)
            for y in range(L):
                for x in range(L):
                    xx, yy = f(x, y); p[site(x, y)] = site(xx, yy)
            return p
        c4 = pm(lambda x, y: (-y, x)); sx = pm(lambda x, y: (-x, y)); r = np.arange(N)
        pts = []
        for k in range(4):
            for refl in (False, True):
                q = r.copy()
                for _ in range(k): q = c4[q]
                if refl: q = sx[q]
                pts.append(q)
        out = []
        for ty in range(L):
            for tx in range(L):
                t = pm(lambda x, y: (x + tx, y + ty))
                for q in pts: out.append(t[q])
        return np.array(out)

    def set_proj(self, proj):
        self.proj = proj
        G = self.G
        if proj == 'none':
            P = G[:1]; fl = [1.0]
        else:
            if proj == 'simple':                            # point index 0 (E) and 2 (C4, no reflection)
                P = np.concatenate([G[8 * t + k][None] for t in range(self.N) for k in (0, 2)])
            elif proj == 'full':
                P = G
            elif proj == 'translations':                    # point index 0 (E) only: N translations x spin flip (2N images)
                P = np.stack([G[8 * t] for t in range(self.N)])
            else:
                raise ValueError(proj)
            fl = [1.0, -1.0]
        IMG = np.concatenate([P for _ in fl]); FL = np.concatenate([[f] * len(P) for f in fl])
        self.IMG = jnp.asarray(IMG.astype(np.int32)); self.FL = jnp.asarray(FL)
        self.INV = jnp.asarray(np.argsort(IMG, axis=1).astype(np.int32))   # z_{INV[g,i]} = flip_g sigma_i
        self.nI = IMG.shape[0]


# ------------------------------------------------------------------------------------------ parameters
def init_params(lat, M=16, seed=0, rbm_scale=0.05, fmax0=0.5, a=2.0, rbm=True):
    """published initialisation: Re/Im of b, W uniform in [-0.05, 0.05]; f_ij uniform in [-fmax, fmax],
    fmax ~ R_ij^-a (a ~ 2), R_ij the periodic distance (f_ii is redundant)."""
    rng = np.random.default_rng(seed)
    L, N = lat.L, lat.N
    xy = np.array([(i % L, i // L) for i in range(N)])
    d = np.abs(xy[:, None, :] - xy[None, :, :]); d = np.minimum(d, L - d)
    R = np.sqrt((d ** 2).sum(-1)); R[R == 0] = 1.0
    f = rng.uniform(-1, 1, (N, N)) * fmax0 * R ** (-a)
    u = lambda *s: rng.uniform(-rbm_scale, rbm_scale, s) if rbm else np.zeros(s)
    return dict(f=jnp.asarray(f), bre=jnp.asarray(u(M)), bim=jnp.asarray(u(M)),
                Wre=jnp.asarray(u(N, M)), Wim=jnp.asarray(u(N, M)))


def logcosh(z):
    """log cosh for complex z (stable)."""
    s = jnp.where(jnp.real(z) >= 0, 1.0, -1.0)
    return s * z + jnp.log1p(jnp.exp(-2.0 * s * z)) - jnp.log(2.0)


def lse_c(l, axis=-1):
    """complex log-sum-exp."""
    m = jax.lax.stop_gradient(jnp.max(jnp.real(l), axis=axis, keepdims=True))
    return jnp.squeeze(m, axis) + jnp.log(jnp.sum(jnp.exp(l - m), axis=axis))


# ------------------------------------------------------------------------------------------ amplitudes
def make_fns(lat, dtype=jnp.float64, use_rbm=True, ratio='ref', _break=None, kappa_max=1e12):
    """ratio = 'ref' (default, the launched production code: rank-2 update with the V array and two bond reductions) or
    'fast' (determinant contraction: batched n x n GEMMs P = A G, Q = G B, S = P B and the table
    D = P*Q + G*(C - S), A = f[D,D], B = f[U,U], C = f[D,U], G = F^-1; psi_PP(z')/psi_PP(z) = -D[b,a]; exactly singular
    images (non-finite, or 1-norm condition estimate ||F||_1 ||G||_1 > kappa_max: there the rank-2 / contraction formulas lose eps*kappa digits)
    are flagged by eloc_flag / eloc_full_flag and the whole configuration is re-evaluated directly by eloc_full_direct, outside the jitted batch)."""
    assert ratio in ('ref', 'fast') and _break in (None, 'sign', 'noCS')   # _break: deliberately broken fast variants, for the agreement test only
    fast = ratio == 'fast'
    N = lat.N; n = N // 2
    cdt = jnp.complex128 if dtype == jnp.float64 else jnp.complex64
    IMG, FL, INV = lat.IMG, lat.FL.astype(dtype), lat.INV
    BI = jnp.asarray(lat.BI); BJ = jnp.asarray(lat.BJ); JB = jnp.asarray(lat.JB, dtype)
    PI = jnp.pi

    def parts(p):
        f = p['f'].astype(dtype)
        b = (p['bre'] + 1j * p['bim']).astype(cdt); W = (p['Wre'] + 1j * p['Wim']).astype(cdt)
        return f, b, W

    def log_one(f, b, W, z):
        """log Psi(z) (complex) for one configuration z (N,) of +-1."""
        order = jnp.argsort(-z, stable=True)
        U = order[:n]; D = order[n:]
        F = f[U][:, D]
        sg, ld = jnp.linalg.slogdet(F)
        up = (z > 0).astype(jnp.int32); dn = 1 - up
        inv = jnp.sum(up * (jnp.cumsum(dn) - dn))
        par = (jnp.where(sg < 0, 1, 0) + inv) % 2
        out = ld.astype(cdt) + 1j * PI * par
        if use_rbm:
            out = out + jnp.sum(logcosh(b + z.astype(cdt) @ W))
        return out

    def images(sigma):
        return FL[:, None] * sigma.astype(dtype)[IMG]

    def logpsi(p, sigma):
        """log psi_P(sigma), complex, brute force (all images)."""
        f, b, W = parts(p)
        Z = images(sigma)
        l = jax.vmap(lambda z: log_one(f, b, W, z))(Z)
        return lse_c(l)

    def logpsi_ri(p, sigma):
        l = logpsi(p, sigma)
        return jnp.stack([jnp.real(l), jnp.imag(l)])

    def diag(sigma):
        s = sigma.astype(dtype)
        return 0.25 * jnp.sum(JB * s[BI] * s[BJ])

    def img_quant_k(f, b, W, z, inv_g):
        """per image: log Psi(z) and the ratios Psi(z with bond b swapped)/Psi(z) for all bonds (0 if aligned)."""
        order = jnp.argsort(-z, stable=True)
        U = order[:n]; D = order[n:]
        F = f[U][:, D]
        sg, ld = jnp.linalg.slogdet(F)
        Gm = jnp.linalg.inv(F)                               # (n, n): G = F^-1
        up = (z > 0).astype(jnp.int32); dn = 1 - up
        inv = jnp.sum(up * (jnp.cumsum(dn) - dn))
        par = (jnp.where(sg < 0, 1, 0) + inv) % 2
        lg = ld.astype(cdt) + 1j * PI * par
        pos = jnp.where(up > 0, jnp.cumsum(up) - 1, jnp.cumsum(dn) - 1)
        si = inv_g[BI]; sj = inv_g[BJ]                       # image sites of each bond
        zi = z[si]; zj = z[sj]
        valid = zi != zj
        p_ = jnp.where(zi > 0, si, sj); q_ = jnp.where(zi > 0, sj, si)   # p up, q down
        a_ = pos[p_]; bc = pos[q_]
        if fast:
            # determinant contraction (DECISION.md): ratio = -D[b,a], D = P*Q + G*(C - S); rows of F = U (sorted), columns D (sorted);
            # G[b,a] is (D index, U index); a = position of the up site p in U, b = position of the down site q in D.
            fD = f[D]                                        # (n, N)
            Am = fD[:, D]; Cm = fD[:, U]; Bm = f[U][:, U]    # f[D,D], f[D,U], f[U,U]
            Pm = Am @ Gm; Qm = Gm @ Bm; Sm = Pm @ Bm         # three n x n GEMMs
            ii = (bc, a_)                                    # gather the NB needed entries before the elementwise arithmetic
            rdet = Pm[ii] * Qm[ii] + Gm[ii] * ((Cm[ii] - Sm[ii]) if _break != 'noCS' else 0.0)
            ratio = (-rdet if _break != 'sign' else rdet).astype(cdt)
        else:
            R = f[:, D] @ Gm                                 # (N, n)
            fU = f[U]                                        # (n, N)
            V = fU[:, p_] - fU[:, q_]                        # (n, NB)
            ar = jnp.arange(lat.NB)
            V = V.at[a_, ar].set(f[q_, p_] - f[q_, q_])
            Gb = Gm[bc]                                      # (NB, n)
            Rq = R[q_]                                       # (NB, n)
            Gv = jnp.sum(Gb * V.T, 1); rGv = jnp.sum(Rq * V.T, 1)
            rdet = Rq[ar, a_] * (1.0 + Gv) - Gb[ar, a_] * (rGv - V[a_, ar])
            ratio = -rdet.astype(cdt)
        if use_rbm:
            th = b + z.astype(cdt) @ W                       # (M,)
            dth = -2.0 * W[p_] + 2.0 * W[q_]                 # (NB, M)
            lg = lg + jnp.sum(logcosh(th))
            ratio = ratio * jnp.exp(jnp.sum(logcosh(th[None] + dth) - logcosh(th)[None], 1))
        kap = jnp.max(jnp.sum(jnp.abs(F), 0)) * jnp.max(jnp.sum(jnp.abs(Gm), 0))     # 1-norm condition estimate (G = F^-1)
        return lg, jnp.where(valid, ratio, 0.0), kap

    def img_quant(f, b, W, z, inv_g):
        return img_quant_k(f, b, W, z, inv_g)[:2]

    def eloc_full(p, sigma):
        """local energy (fast updates), log psi_P and the bond ratios psi_P(y_b)/psi_P(x) (0 for aligned bonds)."""
        f, b, W = parts(p)
        Z = images(sigma)
        lg, rat = jax.vmap(lambda z, ig: img_quant(f, b, W, z, ig))(Z, INV)    # (nI,), (nI, NB)
        lP = lse_c(lg)
        w = jnp.exp(lg - lP)
        rP = w @ rat                                         # psi_P(y_b)/psi_P(x)
        s = sigma.astype(dtype)
        valid = s[BI] != s[BJ]
        off = jnp.sum(jnp.where(valid, 0.5 * JB * rP, 0.0))
        return diag(sigma) + off, lP, rP

    def eloc_full_flag(p, sigma):
        """eloc_full plus bad = any image with non-finite log Psi (singular F) or non-finite bond ratio (needs the direct fallback)."""
        f, b, W = parts(p)
        Z = images(sigma)
        lg, rat, kap = jax.vmap(lambda z, ig: img_quant_k(f, b, W, z, ig))(Z, INV)
        bad = jnp.any(~jnp.isfinite(lg.real)) | jnp.any(~jnp.isfinite(rat.real)) | jnp.any(~jnp.isfinite(rat.imag)) | jnp.any(~(kap <= kappa_max))
        lP = lse_c(lg)
        w = jnp.exp(lg - lP)
        rP = w @ rat
        s = sigma.astype(dtype)
        valid = s[BI] != s[BJ]
        off = jnp.sum(jnp.where(valid, 0.5 * JB * rP, 0.0))
        return diag(sigma) + off, lP, rP, bad

    def eloc_flag(p, sigma):
        e, lP, _, bad = eloc_full_flag(p, sigma)
        return e, lP, bad

    def eloc_full_direct(p, sigma):
        """same outputs as eloc_full, every neighbour re-evaluated with its own slogdets (all images); exact for singular images."""
        lP = logpsi(p, sigma)
        s = sigma.astype(dtype)

        def nb(k):
            y = s.at[BI[k]].set(s[BJ[k]]).at[BJ[k]].set(s[BI[k]])
            return jnp.exp(logpsi(p, y) - lP)
        valid = s[BI] != s[BJ]
        rP = jnp.where(valid, jax.lax.map(nb, jnp.arange(lat.NB)), 0.0)
        return diag(sigma) + jnp.sum(jnp.where(valid, 0.5 * JB * rP, 0.0)), lP, rP

    def eloc_one(p, sigma):
        """local energy (fast updates) and log psi_P for one configuration."""
        e, lP, _ = eloc_full(p, sigma)
        return e, lP

    def eloc_brute(p, sigma):
        """local energy by brute force (all neighbours re-evaluated); for tests."""
        lP = logpsi(p, sigma)
        s = sigma.astype(dtype)

        def nb(k):
            y = s.at[BI[k]].set(s[BJ[k]]).at[BJ[k]].set(s[BI[k]])
            return jnp.exp(logpsi(p, y) - lP)
        r = jax.vmap(nb)(jnp.arange(lat.NB))
        valid = s[BI] != s[BJ]
        return diag(sigma) + jnp.sum(jnp.where(valid, 0.5 * JB * r, 0.0)), lP

    return dict(logpsi=logpsi, logpsi_ri=logpsi_ri, eloc=eloc_one, eloc_full=eloc_full, eloc_brute=eloc_brute, eloc_flag=eloc_flag, eloc_full_flag=eloc_full_flag,
                eloc_full_direct=eloc_full_direct, img_quant=img_quant, img_quant_k=img_quant_k, diag=diag,
                log_one=log_one, parts=parts)


# ------------------------------------------------------------------------------------------ Metropolis sampler
def make_sampler(lat, fns32):
    """exchange moves along a uniformly chosen anti-aligned bond (Hastings factor n_anti(x)/n_anti(y)),
    acceptance from log psi_P in single precision (only the acceptance uses it)."""
    BI = jnp.asarray(lat.BI); BJ = jnp.asarray(lat.BJ)
    logpsi = jax.vmap(fns32['logpsi'], in_axes=(None, 0))

    def nanti(S):
        return jnp.sum(S[:, BI] != S[:, BJ], 1)

    def step(p32, S, lS, key):
        k1, k2 = jax.random.split(key)
        anti = S[:, BI] != S[:, BJ]
        sc = jnp.where(anti, jax.random.uniform(k1, anti.shape), -1.0)
        bsel = jnp.argmax(sc, 1)
        i = BI[bsel]; j = BJ[bsel]
        r = jnp.arange(S.shape[0])
        si = S[r, i]; sj = S[r, j]
        Y = S.at[r, i].set(sj).at[r, j].set(si)
        lY = logpsi(p32, Y)
        lacc = 2.0 * jnp.real(lY - lS) + jnp.log(nanti(S) / nanti(Y))
        acc = jnp.log(jax.random.uniform(k2, (S.shape[0],))) < lacc
        S = jnp.where(acc[:, None], Y, S); lS = jnp.where(acc, lY, lS)
        return S, lS, acc

    from functools import partial

    @partial(jax.jit, static_argnums=(3,))
    def sweep(p32, S, key, nstep):
        lS = logpsi(p32, S)

        def body(c, k):
            S, lS, na = c
            S, lS, acc = step(p32, S, lS, k)
            return (S, lS, na + jnp.mean(acc)), None
        (S, lS, na), _ = jax.lax.scan(body, (S, lS, 0.0), jax.random.split(key, nstep))
        return S, na / nstep
    return sweep


def random_configs(N, B, seed):
    rng = np.random.default_rng(seed)
    S = np.ones((B, N), np.int8); S[:, N // 2:] = -1
    for r in range(B): rng.shuffle(S[r])
    return jnp.asarray(S)


def to32(p):
    return jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), p)
