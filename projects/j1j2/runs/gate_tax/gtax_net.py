"""RBM+PP network as a log-amplitude model for the G-tax refit (README compress_6x6 "M1 + gate for the loop", G-tax).
Amplitude A(sigma) = |Re(e^{-i theta} psi_P(sigma))|, sign = sgn Re(e^{-i theta} psi_P); psi_P = the fully projected RBM+PP
(rbmpp_lib: 576 images).  theta = atan2(c1, c0) of the guide's 2x2 combination (table_A.json guide.c; c = (1, 0) -> theta 0).
Imported by cx_gtax.py (kept apart so the net can be tested on a CPU without the sector)."""
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from rbmpp_lib import Lattice, make_fns
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
