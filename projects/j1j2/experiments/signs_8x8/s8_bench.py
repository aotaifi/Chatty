"""Speed check of the sign CNN forward/backward: flax nn.Conv (wrap pad) vs an equivalent gather (im2col) + matmul
forward using the SAME parameters.  Prints throughput and the max |logit difference|.
  python s8_bench.py OUT
"""
import sys, time, json
import numpy as np
import jax, jax.numpy as jnp
import s8_net as SN

L = 8; N = 64
# neighbour table: for site (x, y) and kernel offset (dy, dx) in flax order (kh, kw) = (dy+1, dx+1)
NBR = np.zeros((N, 9), np.int32)
for y in range(L):
    for x in range(L):
        for ky in range(3):
            for kx in range(3):
                NBR[x + L * y, ky * 3 + kx] = ((x + kx - 1) % L) + L * ((y + ky - 1) % L)
NBRj = jnp.asarray(NBR)


def fast_apply(params, X, depth):
    """X (B, 64) +-1 -> logit, same function as SignCNN.apply."""
    h = X[:, :, None]                                             # (B, N, C)
    def conv(h, p):
        k = p['kernel']; kh, kw, ci, co = k.shape
        g = h[:, NBRj, :]                                         # (B, N, 9, C)
        return g.reshape(h.shape[0], N, 9 * ci) @ k.reshape(9 * ci, co) + p['bias']
    def ln(h, p):
        m = h.mean(-1, keepdims=True); v = ((h - m) ** 2).mean(-1, keepdims=True)
        return (h - m) / jnp.sqrt(v + 1e-6) * p['scale'] + p['bias']
    h = jax.nn.gelu(conv(h, params['c0']))
    for d in range(1, depth):
        h = h + jax.nn.gelu(conv(ln(h, params[f'ln{d}']), params[f'c{d}']))
    o = h @ params['head']['kernel'] + params['head']['bias']
    return o.sum(axis=(1, 2)) / 8.0


if __name__ == '__main__':
    net = SN.SignNet(64, 6, seed=0)
    rg = np.random.default_rng(0); B = 8192
    X = jnp.asarray(rg.choice([-1.0, 1.0], size=(B, N)).astype(np.float32))
    f1 = jax.jit(lambda p, X: net.m.apply({'params': p}, X)); f2 = jax.jit(lambda p, X: fast_apply(p, X, 6))
    res = {}
    for nm, f in (('flax', f1), ('gather', f2)):
        y = f(net.params, X).block_until_ready(); t = time.time()
        for _ in range(5): y = f(net.params, X).block_until_ready()
        res[nm + '_fwd_per_s'] = 5 * B / (time.time() - t)
        g = jax.jit(jax.grad(lambda p, X: f(p, X).sum())); r = g(net.params, X[:1024]); jax.block_until_ready(r); t = time.time()
        for _ in range(10): r = g(net.params, X[:1024]); jax.block_until_ready(r)
        res[nm + '_grad_step_ms_b1024'] = (time.time() - t) / 10 * 1e3
    res['max_abs_diff'] = float(jnp.max(jnp.abs(f1(net.params, X) - f2(net.params, X))))
    print(json.dumps(res)); json.dump(res, open(sys.argv[1] + '/bench.json', 'w'))
