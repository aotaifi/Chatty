"""sign8.py: sign CNN for the stored s_G sign (B design item ii). SCOPE-REDUCED (lead, 2026-11-03): only the network, sign_fn,
save/load and a CPU run check. s_G labels, tempered sampler, training and the acceptance metric are NOT written (not built).
Net = translation-invariant periodic CNN (3x3 wrap convs, residual GELU + LayerNorm, per-site head summed / sqrt(N)), as
runs/archive/chatty_signlearn/code/sl_train.py (SIZES XL=(128,6), L=(64,6)). Logit is averaged over x and -x (s_G is flip-even).
NOTE: no D4 averaging (the 8x8 guide has translations only, so s_G is not point-group invariant).
API:  net = SignNet(L=8, size='XL'|'L'|(ch,depth), seed=0)       random weights
      net.sign_fn(sigma)  / net(sigma): numpy (B,N) of +-1 (any dtype) -> int8 (B,) in {+1,-1}; batched in fixed chunks
      net.logit(sigma) -> float32 (B,);  net.jax_logit(params, x) pure jittable (B,N) float -> (B,)
      net.save(path.npz); SignNet.load(path.npz)
"""
import json
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn

SIZES = dict(XS=(8, 2), S=(16, 3), M=(32, 4), L=(64, 6), XL=(128, 6))


class SignCNN(nn.Module):
    L: int = 8
    ch: int = 128
    depth: int = 6

    @nn.compact
    def __call__(self, x):                                   # x (B, L*L) +-1, site = x + L y
        L = self.L
        h = x.reshape(-1, L, L, 1)
        def conv(h, c, name):
            h = jnp.pad(h, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='wrap')
            return nn.Conv(c, (3, 3), padding='VALID', name=name)(h)
        h = nn.gelu(conv(h, self.ch, 'c0'))
        for d in range(1, self.depth):
            h = h + nn.gelu(conv(nn.LayerNorm(name=f'ln{d}')(h), self.ch, f'c{d}'))
        o = nn.Dense(1, name='head')(h)
        return o.sum(axis=(1, 2, 3)) / float(L)             # sum over L^2 sites / sqrt(N)


def _flat(p, pre=''):
    out = {}
    for k, v in p.items():
        out.update(_flat(v, pre + k + '/') if isinstance(v, dict) else {pre + k: np.asarray(v)})
    return out


def _unflat(d):
    out = {}
    for k, v in d.items():
        cur = out
        ks = k.split('/')
        for q in ks[:-1]: cur = cur.setdefault(q, {})
        cur[ks[-1]] = jnp.asarray(v)
    return out


class SignNet:
    def __init__(self, L=8, size='XL', seed=0, chunk=1024):
        ch, depth = SIZES[size] if isinstance(size, str) else tuple(size)
        self.cfg = dict(L=L, ch=ch, depth=depth); self.L = L; self.N = L * L; self.chunk = chunk
        self.m = SignCNN(L=L, ch=ch, depth=depth)
        self.params = self.m.init(jax.random.PRNGKey(seed), jnp.zeros((1, self.N), jnp.float32))['params']
        self.npar = int(sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(self.params)))
        m = self.m
        self.jax_logit = lambda p, x: 0.5 * (m.apply({'params': p}, x) + m.apply({'params': p}, -x))
        self._f = jax.jit(self.jax_logit)

    def logit(self, sigma):
        s = np.asarray(sigma, np.float32); B = len(s); out = np.empty(B, np.float32)
        for i in range(0, B, self.chunk):
            blk = s[i:i + self.chunk]; n = len(blk)
            if n < self.chunk: blk = np.concatenate([blk, np.repeat(blk[:1], self.chunk - n, 0)])   # fixed shape: one compile
            out[i:i + n] = np.asarray(self._f(self.params, jnp.asarray(blk)))[:n]
        return out

    def sign_fn(self, sigma):
        return np.where(self.logit(sigma) >= 0, 1, -1).astype(np.int8)
    __call__ = sign_fn

    def save(self, path):
        np.savez(path, cfg=json.dumps(self.cfg), **{'p:' + k: v for k, v in _flat(jax.device_get(self.params)).items()})

    @staticmethod
    def load(path):
        z = np.load(path); c = json.loads(str(z['cfg']))
        n = SignNet(L=c['L'], size=(c['ch'], c['depth']))
        n.params = _unflat({k[2:]: z[k] for k in z.files if k.startswith('p:')}); return n


if __name__ == '__main__':      # CPU check:  JAX_PLATFORMS=cpu python3 sign8.py
    import time, tempfile, os
    rng = np.random.default_rng(0)
    for size in ('L', 'XL'):
        net = SignNet(8, size); X = np.ones((300, 64), np.int8); X[:, 32:] = -1
        for r in X: rng.shuffle(r)
        t = time.time(); s = net(X); dt = time.time() - t; t = time.time(); s = net(X); dt2 = time.time() - t
        sh = np.roll(X.reshape(-1, 8, 8), (3, 5), (1, 2)).reshape(-1, 64)
        p = tempfile.mktemp(suffix='.npz'); net.save(p); n2 = SignNet.load(p); os.remove(p)
        print(f'{size}: npar {net.npar}  out {s.shape} {s.dtype} vals {sorted(set(s.tolist()))}  +1 frac {np.mean(s>0):.2f}  '
              f'translation-invariant {np.array_equal(s, net(sh))}  flip-invariant {np.array_equal(s, net(-X))}  '
              f'save/load equal {np.array_equal(s, n2(X))}  CPU {dt2/300*1e3:.2f} ms/config (first call {dt:.1f}s)')
