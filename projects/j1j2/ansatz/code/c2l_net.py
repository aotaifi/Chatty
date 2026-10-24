"""RBM+PP network as a log-amplitude model for the G-tax refit (README compress_6x6 "M1 + gate for the loop", G-tax).
Amplitude A(sigma) = |Re(e^{-i theta} psi_P(sigma))|, sign = sgn Re(e^{-i theta} psi_P); psi_P = the fully projected RBM+PP
(rbmpp_lib: 576 images).  theta = atan2(c1, c0) of the guide's 2x2 combination (table_A.json guide.c; c = (1, 0) -> theta 0).
Imported by cx_gtax.py (kept apart so the net can be tested on a CPU without the sector)."""
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from c2l_rbmpp_lib import Lattice, make_fns
jax.config.update("jax_enable_x64", True)
N = 36
AR = jnp.arange(N, dtype=jnp.uint64)


def bits64(Sx):
    return (((Sx[:, None] >> AR[None, :]) & jnp.uint64(1)).astype(jnp.float64) * 2 - 1)


def make_net(params_npz, theta_ph):
    z = np.load(params_npz)
    assert bool(z['rbm']), 'RBM+PP expected'
    p0 = {k: jnp.asarray(z[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}
    theta0, unravel = ravel_pytree(p0)
    lat = Lattice(6, 0.5, 'full')
    fns = make_fns(lat, jnp.float64, True)

    def l_of(th, sigma):                                   # complex log psi_P
        return fns['logpsi'](unravel(th), sigma)

    def logA_one(th, sigma):
        l = l_of(th, sigma)
        return jnp.real(l) + jnp.log(jnp.maximum(jnp.abs(jnp.cos(jnp.imag(l) - theta_ph)), 1e-300))

    def parts_one(th, sigma):                              # [Re l, cos(Im l - theta), sin(Im l - theta)]
        l = l_of(th, sigma); ph = jnp.imag(l) - theta_ph
        return jnp.stack([jnp.real(l), jnp.cos(ph), jnp.sin(ph)])
    fwd_j = jax.jit(lambda th, Sx: jax.vmap(lambda s_: logA_one(th, s_))(bits64(Sx)))
    parts_j = jax.jit(lambda th, Sx: jax.vmap(lambda s_: parts_one(th, s_))(bits64(Sx)))
    jac_j = jax.jit(lambda th, Sx: jax.vmap(jax.grad(logA_one), in_axes=(None, 0))(th, bits64(Sx)))
    return dict(theta0=theta0, unravel=unravel, fwd_j=fwd_j, parts_j=parts_j, jac_j=jac_j, logA_one=logA_one, fns=fns)


def make_rbm_net(params_npz, M=33, seed=0, init_scale=1e-2):
    """C2-L-1 arm (ii): POSITIVE symmetrised RBM, no Pfaffian.  A(sigma) = sum_g |phi_RBM(z_g)| over the 576 images
    (full space group x spin flip), phi_RBM = prod_k 2 cosh(b_k + z.W_k) with complex b, W (real params Re/Im).
    Warm start: A's 16 RBM units (b, W from params_npz) + M-16 extra units drawn N(0, init_scale) (nonzero so their
    gradients are nonzero).  Params: 2 M (N + 1) = 2442 for M = 33.  Sign is +1 everywhere (parts: [logA, 1, 0])."""
    z = np.load(params_npz)
    rng = np.random.default_rng(seed); M0 = z['bre'].shape[0]; ex = M - M0
    p0 = {'bre': jnp.asarray(np.concatenate([z['bre'], init_scale * rng.standard_normal(ex)])),
          'bim': jnp.asarray(np.concatenate([z['bim'], init_scale * rng.standard_normal(ex)])),
          'Wre': jnp.asarray(np.concatenate([z['Wre'], init_scale * rng.standard_normal((N, ex))], 1)),
          'Wim': jnp.asarray(np.concatenate([z['Wim'], init_scale * rng.standard_normal((N, ex))], 1))}
    theta0, unravel = ravel_pytree(p0)
    lat = Lattice(6, 0.5, 'full')
    from c2l_rbmpp_lib import logcosh
    IMG, FL = lat.IMG, lat.FL.astype(jnp.float64)

    def logA_one(th, sigma):
        p = unravel(th); b = p['bre'] + 1j * p['bim']; W = p['Wre'] + 1j * p['Wim']
        Z = FL[:, None] * sigma[IMG]                       # (576, N) images
        lr = jnp.real(jnp.sum(logcosh(b[None, :] + Z.astype(jnp.complex128) @ W), 1))
        return jax.scipy.special.logsumexp(lr)

    def parts_one(th, sigma):
        return jnp.stack([logA_one(th, sigma), jnp.ones(()), jnp.zeros(())])
    fwd_j = jax.jit(lambda th, Sx: jax.vmap(lambda s_: logA_one(th, s_))(bits64(Sx)))
    parts_j = jax.jit(lambda th, Sx: jax.vmap(lambda s_: parts_one(th, s_))(bits64(Sx)))
    jac_j = jax.jit(lambda th, Sx: jax.vmap(jax.grad(logA_one), in_axes=(None, 0))(th, bits64(Sx)))
    return dict(theta0=theta0, unravel=unravel, fwd_j=fwd_j, parts_j=parts_j, jac_j=jac_j, logA_one=logA_one, fns=None)
