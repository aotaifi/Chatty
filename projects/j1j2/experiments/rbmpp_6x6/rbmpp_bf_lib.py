"""PP + 3x3 local backflow (the cx_t1s.py 'ppbf' network) for the J1-J2 Heisenberg model on the 6x6 torus (JAX), with the
variational local energy of Test B0 (README compress_6x6, "Test B0").  No RBM.  Real wave function.

  psi(sigma) = sum_g sgn_g(sigma) exp(ld_g(sigma)),  g over the images z_g = flip_g sigma[IMG_g]  (rbmpp_lib.Lattice, 'full' =
  whole space group x spin flip = 576 images, 'simple' = 144),
  ld_g, sgn_g = log|det F(z_g)|, eps(z_g) sign(det F(z_g)),   F = f[U, D] + backflow:
  F_ij(z) = f_ij + B_ij(z),  B_ij(z) = sum_{d in 3x3} [u1(d, r_j - r_i) z_{i+d} + u2(d, r_i - r_j) z_{j+d}]
  (U, D = up / down sites of z in increasing order; eps = reordering sign, as rbmpp_lib.log_one).
Parameter vector theta = [f.ravel() (N*N, diagonal unused), u1 (9, N), u2 (9, N)], N = 36, 1296 + 648 = 1944 entries
(the layout of cx_t1s.py 'ppbf': theta = concat(f, u.reshape(2, 9, N))).

Local energy: with backflow no rank-2 update of F exists, so psi(y_b) of every bond swap is evaluated by brute force
(all images).  E_loc(x) = H_xx + sum_{b anti-aligned} 0.5 J_b psi(y_b)/psi(x)  (rbmpp_lib convention).

Nothing here modifies rbmpp_lib.py; it is imported for Lattice, make_sampler, random_configs, to32.
"""
import numpy as np
import jax
import jax.numpy as jnp
from rbmpp_lib import Lattice, make_sampler, random_configs, to32  # noqa: F401  (re-exported)
jax.config.update("jax_enable_x64", True)

L = 6
N = L * L
NH = N // 2
NBR = np.array([[((x + dx) % L) + L * ((y + dy) % L) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
                for y in range(L) for x in range(L)], np.int32)                   # (36, 9): 3x3 window around each site
_xy = np.array([(i % L, i // L) for i in range(N)])
DISP = np.array([[((_xy[j, 0] - _xy[i, 0]) % L) + L * ((_xy[j, 1] - _xy[i, 1]) % L) for j in range(N)] for i in range(N)],
                np.int32)                                                          # DISP[i, j] = index of r_j - r_i
NTH = N * N + 2 * 9 * N


def theta_from_f(f, u=None):
    f = np.asarray(f, np.float64).reshape(-1)
    u = np.zeros(2 * 9 * N) if u is None else np.asarray(u, np.float64).reshape(-1)
    return np.concatenate([f, u])


def active_mask():
    """True for the trainable entries: f off the diagonal, all backflow entries."""
    m = np.ones(NTH, bool)
    m[:N * N].reshape(N, N)[np.arange(N), np.arange(N)] = False
    return m


def make_bf_fns(lat, dtype=jnp.float64):
    """log|psi|, sign, local energy for one configuration (vmap them).  theta: flat vector of length NTH."""
    IMG, FL = lat.IMG, lat.FL.astype(dtype)
    BI = jnp.asarray(lat.BI); BJ = jnp.asarray(lat.BJ); JB = jnp.asarray(lat.JB, dtype)
    NB = lat.NB
    NBRj = jnp.asarray(NBR); DISPj = jnp.asarray(DISP)

    def split(theta):
        th = theta.astype(dtype)
        return th[:N * N].reshape(N, N), th[N * N:].reshape(2, 9, N)[0], th[N * N:].reshape(2, 9, N)[1]

    def log_img(f, u1, u2, zz):
        """log|psi_PP(z)|-term and sign of one image z (cx_t1s.py log_one, BF = True)."""
        order = jnp.argsort(-zz, stable=True); Uo = order[:NH]; Do = order[NH:]
        Fm = f[Uo][:, Do]
        zn = zz[NBRj]                                                       # (36, 9)
        M1 = zn[Uo] @ u1                                                    # (18, 36): [up site a, displacement]
        M2 = zn[Do] @ u2                                                    # (18, 36): [down site b, displacement]
        Fm = Fm + jnp.take_along_axis(M1, DISPj[Uo][:, Do], 1) + jnp.take_along_axis(M2, DISPj[Do][:, Uo], 1).T
        sg, ld = jnp.linalg.slogdet(Fm)
        up = (zz > 0).astype(jnp.int32); dn = 1 - up
        inv = jnp.sum(up * (jnp.cumsum(dn) - dn))
        par = (jnp.where(sg < 0, 1, 0) + inv) % 2
        return ld, 1.0 - 2.0 * par.astype(dtype)

    def amp_parts(f, u1, u2, sigma):
        Z = FL[:, None] * sigma.astype(dtype)[IMG]
        ld, sgn = jax.vmap(lambda zz: log_img(f, u1, u2, zz))(Z)
        m = jax.lax.stop_gradient(jnp.max(ld))
        s = jnp.sum(sgn * jnp.exp(ld - m))
        tiny = jnp.finfo(dtype).tiny
        return m + jnp.log(jnp.maximum(jnp.abs(s), tiny)), jnp.where(s >= 0, 1.0, -1.0).astype(dtype)

    def logpsi(theta, sigma):
        """log|psi_P(sigma)| (real)."""
        f, u1, u2 = split(theta)
        return amp_parts(f, u1, u2, sigma)[0]

    def sign(theta, sigma):
        f, u1, u2 = split(theta)
        return amp_parts(f, u1, u2, sigma)[1]

    def diag(sigma):
        s = sigma.astype(dtype)
        return 0.25 * jnp.sum(JB * s[BI] * s[BJ])

    def eloc(theta, sigma):
        """(E_loc, log|psi|) by brute force over all 144 bonds (aligned bonds are computed and masked)."""
        f, u1, u2 = split(theta)
        la, sg = amp_parts(f, u1, u2, sigma)
        s = sigma.astype(dtype)

        def bond(k):
            y = s.at[BI[k]].set(s[BJ[k]]).at[BJ[k]].set(s[BI[k]])
            ly, sy = amp_parts(f, u1, u2, y)
            return sy * sg * jnp.exp(ly - la)
        r = jax.lax.map(bond, jnp.arange(NB))
        valid = s[BI] != s[BJ]
        return diag(sigma) + jnp.sum(jnp.where(valid, 0.5 * JB * r, 0.0)), la

    def eloc_bonds(theta, sigma, kmax):
        """as eloc but only the first kmax anti-aligned bonds (kmax >= number of anti-aligned bonds of sigma, else wrong)."""
        f, u1, u2 = split(theta)
        la, sg = amp_parts(f, u1, u2, sigma)
        s = sigma.astype(dtype)
        valid = s[BI] != s[BJ]
        sel = jnp.argsort(~valid, stable=True)[:kmax]

        def bond(k):
            y = s.at[BI[k]].set(s[BJ[k]]).at[BJ[k]].set(s[BI[k]])
            ly, sy = amp_parts(f, u1, u2, y)
            return sy * sg * jnp.exp(ly - la)
        r = jax.lax.map(bond, sel)
        return diag(sigma) + jnp.sum(jnp.where(valid[sel], 0.5 * JB[sel] * r, 0.0)), la

    return dict(logpsi=logpsi, sign=sign, eloc=eloc, eloc_bonds=eloc_bonds, diag=diag, split=split)
