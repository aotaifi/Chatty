"""Stored composite sign net at 8x8 (jax/flax): translation-invariant periodic CNN (as stored_signs_6x6 SignCNN and
sign_learnability: 3x3 wrap convolutions, residual GELU + LayerNorm blocks, per-site head summed over sites / sqrt N).
The stored sign is  s_net(x) = Marshall(x) * sgn(logit(x)),  logit averaged over the 16 images of D4 x spin flip (the
composite target is exactly invariant: |psi_P|, its binarised sign and Marshall are invariant on the 8x8 torus).
Training: BCE on y in {+1,-1}, one random D4 x flip image per sample, AdamW with warmup-cosine, uniform minibatches.
"""
import time, json, pickle
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')   # full fp32 (no TF32)
import jax.numpy as jnp
import flax.linen as nn
import optax

L = 8; N = 64; F32 = jnp.float32


def d4_perms():
    ops = [lambda x, y: (x, y), lambda x, y: (-y, x), lambda x, y: (-x, -y), lambda x, y: (y, -x),
           lambda x, y: (-x, y), lambda x, y: (x, -y), lambda x, y: (y, x), lambda x, y: (-y, -x)]
    P = np.zeros((8, N), np.int32)
    for g, op in enumerate(ops):
        for y in range(L):
            for x in range(L):
                u, v = op(x, y); P[g, (x % L) + L * (y % L)] = (u % L) + L * (v % L)
    return P


PERM = jnp.asarray(d4_perms())
SG = jnp.asarray([1.0, -1.0], F32)


class SignCNN(nn.Module):
    ch: int = 64
    depth: int = 6

    @nn.compact
    def __call__(self, x):                                   # x: (B, 64) +-1, site i = x + 8 y
        h = x.reshape(-1, L, L, 1)
        def conv(h, c, name):
            h = jnp.pad(h, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='wrap')
            return nn.Conv(c, (3, 3), padding='VALID', dtype=F32, param_dtype=F32, name=name)(h)
        h = nn.gelu(conv(h, self.ch, 'c0'))
        for d in range(1, self.depth):
            h = h + nn.gelu(conv(nn.LayerNorm(dtype=F32, param_dtype=F32, name=f'ln{d}')(h), self.ch, f'c{d}'))
        o = nn.Dense(1, dtype=F32, param_dtype=F32, name='head')(h)
        return o.sum(axis=(1, 2, 3)) / 8.0


def bits_to_f32(S):
    S = jnp.asarray(S, jnp.uint64)
    return (((S[:, None] >> jnp.arange(N, dtype=jnp.uint64)[None, :]) & jnp.uint64(1)).astype(F32) * 2 - 1)


def marshall_np(S):
    from s8_q import marshall
    return marshall(S)


# gather (im2col) + matmul forward, the same function as SignCNN.apply (checked: max |logit diff| 7e-6 at init,
# s8_bench.py); kept as a reference. With XLA autotuning ON, flax nn.Conv is faster (3e5 vs 1.4e5 states/s on an
# a40-8gb slice) and is used everywhere (all jobs after s8trV run with autotuning on).
_NBR = np.zeros((N, 9), np.int32)
for _y in range(L):
    for _x in range(L):
        for _ky in range(3):
            for _kx in range(3):
                _NBR[_x + L * _y, _ky * 3 + _kx] = ((_x + _kx - 1) % L) + L * ((_y + _ky - 1) % L)
NBRj = jnp.asarray(_NBR)


def fast_apply(params, X, depth):
    h = X[:, :, None]
    def conv(h, p):
        k = p['kernel']; ci, co = k.shape[2], k.shape[3]
        return h[:, NBRj, :].reshape(h.shape[0], N, 9 * ci) @ k.reshape(9 * ci, co) + p['bias']
    def ln(h, p):
        m = h.mean(-1, keepdims=True); v = ((h - m) ** 2).mean(-1, keepdims=True)
        return (h - m) / jnp.sqrt(v + 1e-6) * p['scale'] + p['bias']
    h = jax.nn.gelu(conv(h, params['c0']))
    for d in range(1, depth):
        h = h + jax.nn.gelu(conv(ln(h, params[f'ln{d}']), params[f'c{d}']))
    o = h @ params['head']['kernel'] + params['head']['bias']
    return o.sum(axis=(1, 2)) / 8.0


class SignNet:
    def __init__(self, ch=64, depth=6, seed=0):
        self.cfg = dict(ch=ch, depth=depth)
        self.m = SignCNN(ch=ch, depth=depth)
        self.params = self.m.init(jax.random.PRNGKey(seed), jnp.zeros((1, N), F32))['params']
        self.npar = int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(self.params)))
        m = self.m

        def sym_logit(params, X):
            Xs = jnp.concatenate([X[:, PERM[g]] * SG[f] for g in range(8) for f in range(2)], 0)
            return m.apply({'params': params}, Xs).reshape(16, -1).mean(0)
        self.sym_logit = sym_logit
        self._lo = jax.jit(lambda p, S: sym_logit(p, bits_to_f32(S)))
        self.nevals = 0

    def logit(self, S, B=1024):
        """synchronous per batch (bounded device memory; autotuned convs need large workspaces on 11 GB GPUs)."""
        S = np.asarray(S, np.uint64); out = np.empty(len(S), np.float32)
        for i in range(0, len(S), B):
            s = S[i:i + B]; m = len(s)
            if m < B: s = np.concatenate([s, np.repeat(s[:1], B - m)])
            out[i:i + m] = np.asarray(self._lo(self.params, jnp.asarray(s)))[:m]
        self.nevals += len(S)
        return out

    def sign(self, S):
        """stored composite sign M(x) sgn(logit(x)) as int8."""
        return (marshall_np(S) * np.where(self.logit(S) >= 0, 1, -1)).astype(np.int8)

    def save(self, path):
        pickle.dump(dict(cfg=self.cfg, params=jax.device_get(self.params)), open(path, 'wb'))

    @staticmethod
    def load(path):
        d = pickle.load(open(path, 'rb'))
        n = SignNet(**d['cfg']); n.params = jax.tree_util.tree_map(jnp.asarray, d['params']); return n


def train(net, S, y, Sv, yv, steps, lr=2e-3, wd=1e-4, batch=1024, seed=0, nlog=20, log=print):
    """BCE on y (+1/-1 -> logit > 0 / < 0); one random D4 x flip image per sample; uniform minibatches with replacement.
    Logs training BCE and held-out (symmetrised) error every steps/nlog steps."""
    params = net.params
    sched = optax.warmup_cosine_decay_schedule(0., lr, max(1, steps // 50), steps, end_value=0.0)
    opt = optax.adamw(sched, weight_decay=wd); st = opt.init(params)
    m = net.m
    Sj = jnp.asarray(np.asarray(S, np.uint64)); yj = jnp.asarray((np.asarray(y) > 0).astype(np.float32)); n = len(S)

    def loss_fn(p, X, yb):
        lo = m.apply({'params': p}, X)
        return jnp.mean(optax.sigmoid_binary_cross_entropy(lo, yb))

    def body(carry, _):
        p, st, key = carry
        key, k1, k2 = jax.random.split(key, 3)
        ib = jax.random.randint(k1, (batch,), 0, n)
        X = bits_to_f32(Sj[ib]); gi = jax.random.randint(k2, (batch,), 0, 16)
        Xi = jnp.take_along_axis(X, PERM[gi // 2], axis=1) * SG[gi % 2][:, None]
        l, gr = jax.value_and_grad(loss_fn)(p, Xi, yj[ib])
        u, st = opt.update(gr, st, p)
        return (optax.apply_updates(p, u), st, key), l

    every = max(1, steps // nlog)

    @jax.jit
    def run(p, st, key):
        (p, st, key), ls = jax.lax.scan(body, (p, st, key), None, length=every)
        return p, st, key, ls.mean()

    key = jax.random.PRNGKey(seed); hist = []; t0 = time.time(); done = 0
    yv = np.asarray(yv)
    while done < steps:
        params, st, key, lm = run(params, st, key); done += every
        net.params = params
        lv = net.logit(Sv); err = float(np.mean(np.where(lv >= 0, 1, -1) != yv))
        h = dict(step=done, bce=float(lm), held_err=err, sec=time.time() - t0); hist.append(h); log('TRAIN', json.dumps(h))
    return hist
