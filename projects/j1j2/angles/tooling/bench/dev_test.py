import os, sys, time
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b1_lib import *
print(jax.config.jax_enable_x64, jax.devices())
f = load_f('/scratch/a/A.Otaifi/ansatz_runs/fastratio/data/params_final.npz', jnp.complex128)
A = Arms(8)
Z = jnp.asarray(random_configs(64, 20, 1).astype(np.int8))
t=time.time()
bf = jax.jit(jax.vmap(A.brute, (None, 0)))(f, Z); print('bf', time.time()-t)
valid = Z[:, A.BI] != Z[:, A.BJ]
lgR, R = jax.jit(jax.vmap(A.R_ratios, (None, 0)))(f, Z)
def Qf(f, z, **kw):
    return A.Q_fresh(f, z, **kw)
lgQ, Q = jax.jit(jax.vmap(A.Q_fresh, (None, 0)))(f, Z)
_, Qraw = jax.jit(jax.vmap(lambda f,z: A.Q_fresh(f,z,raw=True), (None, 0)))(f, Z)
_, Qb = jax.jit(jax.vmap(lambda f,z: A.Q_fresh(f,z,swap_idx=True), (None, 0)))(f, Z)
for nm, r in [('R', R), ('Q', Q), ('Qraw', Qraw), ('Qbroken', Qb)]:
    el, nw = relerr(r, bf, valid)
    print(nm, float(el.max()), float(nw.max()), int(sign_mismatch(r, bf, valid).sum()))

# persistent arms: 1 and 20 steps
keys = jax.random.split(jax.random.PRNGKey(0), (20, 20))
st = jax.jit(jax.vmap(A.init_state, (None, 0)))(f, Z)
stepH = jax.jit(jax.vmap(A.stepH, (None, 0, 0)))
mv = jax.jit(jax.vmap(lambda f, st, k: A.move(f, st, *A.pick(st[0], k))[0], (None, 0, 0)))
Lr = jax.jit(jax.vmap(lambda f, st: A.Q_ratios(f, st[1], st[2], st[3], st[0]), (None, 0)))
Hr = jax.jit(jax.vmap(lambda f, st: A.H_ratios(f, st[1], st[2], st[3], st[0]), (None, 0)))
for s in range(20):
    st = mv(f, st, keys[s])
    if s in (0, 19):
        z = st[0]; valid = z[:, A.BI] != z[:, A.BJ]
        bf = jax.jit(jax.vmap(A.brute, (None, 0)))(f, z)
        for nm, r in [('L', Lr(f, st)), ('H', Hr(f, st))]:
            el, nw = relerr(r, bf, valid)
            print(s+1, nm, float(el.max()), float(nw.max()), int(sign_mismatch(r, bf, valid).sum()))
# delayed
T=8
car = jax.jit(jax.vmap(lambda f, z: A.init_carrier(f, z, T), (None, 0)))(f, Z)
for s in range(20):
    t = s % T
    car, chk = jax.jit(jax.vmap(lambda f, st, k: A.stepLd(f, st, k, t, T), (None, 0, 0)))(f, car, keys[s])
z = car[0]; valid = z[:, A.BI] != z[:, A.BJ]
print('delayed chk finite', bool(jnp.all(jnp.isfinite(chk))))
# compare delayed ratios to brute after 20 steps (20 = 2*8+4, so cd=5): recompute explicit
