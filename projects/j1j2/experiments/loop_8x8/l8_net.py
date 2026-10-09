"""FEAT residual network of the 8x8 write-back (writeback_tail wt_loop.ResCNN/Corr adapted to 8x8)."""
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
from jax.flatten_util import ravel_pytree
import l8_core as C
N = C.N; L = C.L

NBR = jnp.asarray(np.array([[((x + dx) % L) + L * ((y + dy) % L) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
                            for y in range(L) for x in range(L)], np.int32))


class ResCNN(nn.Module):
    C: int = 32
    layers: int = 4
    hid: int = 64

    @nn.compact
    def __call__(self, x, t):
        kw = dict(dtype=jnp.float32, param_dtype=jnp.float32)

        def conv(h):
            g = h[:, NBR, :]
            return nn.Dense(self.C, **kw)(g.reshape(g.shape[0], N, -1))
        h = nn.gelu(conv(x[:, :, None]))
        for _ in range(1, self.layers):
            h = h + nn.gelu(conv(nn.LayerNorm(**kw)(h)))
        h = h.sum(axis=1) / 8.0
        om = jnp.asarray([0.25, 0.5, 1.0, 2.0, 4.0], jnp.float32)
        B_ = t.shape[0]
        emb = jnp.concatenate([t, 0.1 * t * t, jnp.sin(t[:, :, None] * om).reshape(B_, -1),
                               jnp.cos(t[:, :, None] * om).reshape(B_, -1)], 1)
        h = jnp.concatenate([h, emb], 1)
        h = nn.gelu(nn.Dense(self.hid, **kw)(h))
        h = nn.gelu(nn.Dense(self.hid, **kw)(h))
        return nn.Dense(1, kernel_init=nn.initializers.zeros, **kw)(h)[:, 0]


class Corr:
    def __init__(self, Cc=32, layers=4, seed=0, nf=3):
        self.net = ResCNN(Cc, layers)
        params = self.net.init(jax.random.PRNGKey(seed), jnp.zeros((1, N), jnp.float32), jnp.zeros((1, nf), jnp.float32))['params']
        self.flat0, self.unravel = ravel_pytree(params); self.npar = int(self.flat0.size)
        perm = jnp.asarray(C.PERM); sg = jnp.asarray([1.0, -1.0], jnp.float32)
        net, unravel = self.net, self.unravel

        def f(flat, X, t):                       # exactly D4 x flip symmetric (t = guide features of the configuration)
            p = unravel(flat)
            return jax.lax.map(lambda k: net.apply({'params': p}, X[:, perm[k // 2]] * sg[k % 2], t), jnp.arange(16)).mean(0)

        def f_aug(flat, X, t, G):                # one image per row
            p = unravel(flat)
            Xg = jnp.take_along_axis(X, perm[G // 2], axis=1) * sg[G % 2][:, None]
            return net.apply({'params': p}, Xg, t)
        self.f = f; self.f_aug = f_aug


def feats(la, V, W, stats):
    cols = [la, np.log(np.maximum(V, 0) + 1e-6), np.log(np.maximum(W, 0) + 1e-6)]
    return np.stack([(c - m) / s for c, (m, s) in zip(cols, stats)], -1).astype(np.float32)


