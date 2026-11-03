import os, sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b1_lib import *
f = load_f('/scratch/a/A.Otaifi/ansatz_runs/fastratio/data/params10_random.npz', jnp.complex128); A = Arms(10)
Z = jnp.asarray(random_configs(100, 40, 3)); valid = Z[:, A.BI] != Z[:, A.BJ]
K = int(valid.sum(1).max()) + 3
Q = jax.jit(jax.vmap(lambda f, z: A.Q_fresh(f, z)[1], (None, 0)))(f, Z)
Qc = jax.jit(jax.vmap(lambda f, z: A.Q_fresh(f, z, K=K)[1], (None, 0)))(f, Z)
print('Qc-Q', float(jnp.abs(Q - Qc).max()), 'K', K, 'nonzero where valid', bool(jnp.all((Qc != 0) | ~valid)))
Kt = int(valid.sum(1).max()) - 5   # too small K must visibly fail (control)
Qt = jax.jit(jax.vmap(lambda f, z: A.Q_fresh(f, z, K=Kt)[1], (None, 0)))(f, Z)
print('control K too small: max diff', float(jnp.abs(Q - Qt).max()))
