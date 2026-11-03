import os, sys, time
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b1_lib import *
f = load_f('/scratch/a/A.Otaifi/ansatz_runs/fastratio/data/params_final.npz', jnp.complex128)
A = Arms(8)
Z = jnp.asarray(random_configs(64, 20, 1).astype(np.int8))
keys = jax.random.split(jax.random.PRNGKey(0), (20, 20))
st = jax.jit(jax.vmap(A.init_state, (None, 0)))(f, Z)
mv = jax.jit(jax.vmap(lambda f, st, k: A.move(f, st, *A.pick(st[0], k))[0], (None, 0, 0)))
Lr = jax.jit(jax.vmap(lambda f, st: A.Q_ratios(f, st[1], st[2], st[3], st[0]), (None, 0)))
def diag(f, st):
    z,G,U,D = st
    F = f[U][:, D]
    return jnp.max(jnp.abs(G@F-jnp.eye(A.n))), jnp.linalg.cond(F)
dg = jax.jit(jax.vmap(diag, (None, 0)))
brute = jax.jit(jax.vmap(A.brute, (None, 0)))
np.set_printoptions(linewidth=200, precision=2)
for s in range(20):
    st = mv(f, st, keys[s])
    z = st[0]; valid = z[:, A.BI] != z[:, A.BJ]
    el, nw = relerr(Lr(f, st), brute(f, z), valid)
    e, c = dg(f, st)
    print(s+1, 'cond max %.2e med %.2e' % (float(c.max()), float(jnp.median(c))), 'GF-I max %.2e' % float(e.max()), 'nw', np.asarray(nw)[:6])
