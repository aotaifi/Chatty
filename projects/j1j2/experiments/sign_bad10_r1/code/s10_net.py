"""10x10 stored composite sign net (T-BAD10 step 0): the 8x8 s8_net.SignCNN architecture (screen arm d: 64 ch x 10 layers, 3x3 wrap convs, residual GELU + LayerNorm,
per-site head summed / sqrt N), D4 x spin-flip averaged logit (16 images), stored sign  n(x) = Marshall(x) sgn(logit(x)).  s10_net_base.py = s8_net.py with L=10, N=100, /10.
Input here is a +-1 float array (B,100) (the 8x8 code used uint64 bit keys, impossible at 100 sites)."""
import time, numpy as np
from s10_net_base import *          # SignCNN, PERM, SG, d4_perms, L, N, F32, jax, jnp, nn, optax
import jax, jax.numpy as jnp
from vitlib import marshall

class Net10:
    def __init__(self, ch=64, depth=10, seed=0, mode='random'):
        self.cfg = dict(ch=ch, depth=depth, seed=seed, mode=mode)
        self.m = SignCNN(ch=ch, depth=depth)
        p = self.m.init(jax.random.PRNGKey(seed), jnp.zeros((1, N), F32))['params']
        if mode == 'marshall':       # head kernel 0, bias +1: logit == 1 everywhere -> n == Marshall exactly (timing/control net; every layer is still computed)
            p = dict(p); h = dict(p['head']); h['kernel'] = jnp.zeros_like(h['kernel']); h['bias'] = jnp.ones_like(h['bias']); p['head'] = h
        self.params = p
        self.npar = int(sum(np.prod(q.shape) for q in jax.tree_util.tree_leaves(p)))
        m = self.m
        def sym_logit(params, X):
            Xs = jnp.concatenate([X[:, PERM[g]] * SG[f] for g in range(8) for f in range(2)], 0)
            return m.apply({'params': params}, Xs).reshape(16, -1).mean(0)
        self.sym_logit = sym_logit; self._lo = jax.jit(sym_logit); self.nevals = 0; self.sec = 0.0
    def logit(self, X, B=512):
        X = np.asarray(X, np.float32); K = len(X); out = np.empty(K, np.float32); t0 = time.perf_counter()
        for i in range(0, K, B):
            s = X[i:i + B]; n = len(s)
            if n < B: s = np.concatenate([s, np.repeat(s[:1], B - n, 0)])
            out[i:i + n] = np.asarray(self._lo(self.params, jnp.asarray(s)))[:n]
        self.nevals += K; self.sec += time.perf_counter() - t0
        return out
    def save(self, path):
        import pickle; pickle.dump(dict(cfg=self.cfg, params=jax.device_get(self.params)), open(path, 'wb'))
    @staticmethod
    def load(path):
        import pickle; d = pickle.load(open(path, 'rb')); c = dict(d['cfg']); c['mode'] = 'random'
        n = Net10(**c); n.params = jax.tree_util.tree_map(jnp.asarray, d['params']); return n
    def sign(self, X, B=512):
        return (marshall(X) * np.where(self.logit(X, B) >= 0, 1, -1)).astype(np.int8)

def train_step_bench(ch, depth, batch=1024, nsteps=20, reps=3, seed=0):
    """ms per AdamW train step (fwd+bwd, one random D4 x flip image per sample, batch 1024) on random +-1 states / random labels, as s8_net.train."""
    net = Net10(ch, depth, seed); m = net.m; params = net.params
    opt = optax.adamw(2e-3, weight_decay=1e-4); st = opt.init(params)
    rng = np.random.default_rng(seed); n = 8192
    Sj = jnp.asarray(np.where(rng.random((n, N)) < .5, 1., -1.).astype(np.float32)); yj = jnp.asarray((rng.random(n) < .5).astype(np.float32))
    def loss_fn(p, X, yb): return jnp.mean(optax.sigmoid_binary_cross_entropy(m.apply({'params': p}, X), yb))
    def body(carry, _):
        p, st, key = carry; key, k1, k2 = jax.random.split(key, 3)
        ib = jax.random.randint(k1, (batch,), 0, n); X = Sj[ib]; gi = jax.random.randint(k2, (batch,), 0, 16)
        Xi = jnp.take_along_axis(X, PERM[gi // 2], axis=1) * SG[gi % 2][:, None]
        l, gr = jax.value_and_grad(loss_fn)(p, Xi, yj[ib]); u, st = opt.update(gr, st, p)
        return (optax.apply_updates(p, u), st, key), l
    @jax.jit
    def run(p, st, key):
        (p, st, key), ls = jax.lax.scan(body, (p, st, key), None, length=nsteps); return p, st, key, ls.mean()
    key = jax.random.PRNGKey(1); params, st, key, l = run(params, st, key); l.block_until_ready(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); params, st, key, l = run(params, st, key); l.block_until_ready(); ts.append((time.perf_counter() - t0) / nsteps * 1e3)
    return dict(ch=ch, depth=depth, ms_per_step=float(np.median(ts)), all_ms=ts, npar=net.npar, loss_finite=bool(np.isfinite(float(l))))
