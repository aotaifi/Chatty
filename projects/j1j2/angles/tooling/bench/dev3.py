import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b1_lib import *
f = load_f('/scratch/a/A.Otaifi/ansatz_runs/fastratio/data/params_final.npz', jnp.complex128)
A = Arms(8); fn = np.asarray(f)
Z = jnp.asarray(random_configs(64, 6, 1).astype(np.int8))
st = jax.jit(jax.vmap(A.init_state, (None, 0)))(f, Z)
keys = jax.random.split(jax.random.PRNGKey(0), (20, 6))
pick = jax.jit(jax.vmap(lambda z,k: A.pick(z,k)))
mv = jax.jit(jax.vmap(lambda f, st, k: A.move(f, st, *A.pick(st[0], k))[0], (None, 0, 0)))
# numpy replay for config c
c = 3
z, G, U, D = [np.asarray(x[c]) for x in st]
Gn = G.copy(); Un = U.copy(); Dn = D.copy()
for s in range(20):
    p, q = [int(x[c]) for x in pick(st[0], keys[s])]
    st = mv(f, st, keys[s])
    a = int(np.where(Un == p)[0][0]); b = int(np.where(Dn == q)[0][0])
    F = fn[Un][:, Dn]
    # A1 = A0 + e_a r^T + c e_b^T
    r = fn[q, Dn] - fn[p, Dn]
    Un2 = Un.copy(); Un2[a] = q
    cc = fn[Un2, p] - fn[Un2, q]
    Uu = np.zeros((A.n, 2), complex); Uu[:, 0] = r; Uu[b, 1] = 1     # u cols
    Vv = np.zeros((A.n, 2), complex); Vv[a, 0] = 1; Vv[:, 1] = cc    # v cols : A1 = A0 + Vv Uu^T
    Rm = np.eye(2) + Uu.T @ Gn @ Vv
    Gn = Gn - Gn @ Vv @ np.linalg.inv(Rm) @ Uu.T @ Gn
    Un = Un2; Dn = Dn.copy(); Dn[b] = p
    F1 = fn[Un][:, Dn]
    Gj = np.asarray(st[1][c])
    print(s+1, 'numpy GF-I %.2e   lrux GF-I %.2e   |Gnp-Glrux| %.2e  cond %.1e' % (np.abs(Gn@F1-np.eye(A.n)).max(), np.abs(Gj@F1-np.eye(A.n)).max(), np.abs(Gn-Gj).max(), np.linalg.cond(F1)))
