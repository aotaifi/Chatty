"""Smoke tests for a JAX GPU environment (used for the A40-slice HOWTO).
  python a40_slice_test.py OUTDIR [CODEDIR]
1) 1000x1000 matmul, 2) fp32 (matmul precision highest) ViT forward pass on fixed random states
(uses experiments/learned_loop_6x6 ll6_core.Net; CODEDIR must contain ll6_core.py/vit_dt.py, cwd must contain
the checkpoint 'vit_J2=0.50_N=6x6_k=0.mpack').  Writes OUTDIR/vit_logabs.npy for cross-GPU comparison."""
import sys, os, time
import numpy as np
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
if len(sys.argv) > 2: sys.path.insert(0, sys.argv[2])
import jax, jax.numpy as jnp
print('jax', jax.__version__, jax.devices(), flush=True)
t = time.time()
print('matmul sum', float((jnp.ones((1000, 1000)) @ jnp.ones((1000, 1000))).sum()), '(expect 1e9)', f'{time.time()-t:.1f}s', flush=True)
import ll6_core as C
net = C.Net('vit_J2=0.50_N=6x6_k=0.mpack', dtype='float32', batch=1024)
rng = np.random.default_rng(0)
# 1024 random Sz=0 states of 36 spins
S = np.zeros(1024, np.uint64)
for k in range(1024):
    up = rng.choice(36, 18, replace=False)
    S[k] = np.uint64(sum(1 << int(i) for i in up))
t = time.time()
la = net.logabs(net.flat0, S)
print('vit logabs[:4]', la[:4], 'mean', la.mean(), f'{time.time()-t:.1f}s', flush=True)
np.save(os.path.join(out, 'vit_logabs.npy'), la)
# float64 reference of the same network on the same device (shows the fp32 error of this build)
net64 = C.Net('vit_J2=0.50_N=6x6_k=0.mpack', dtype='float64', batch=1024)
la64 = net64.logabs(net64.flat0, S)
np.save(os.path.join(out, 'vit_logabs_f64.npy'), la64)
d = np.abs(la - la64)
print(f'fp32 vs fp64 (same device): max|d| {d.max():.2e} rms {np.sqrt((d**2).mean()):.2e}', flush=True)
try:
    print('device mem stats:', jax.devices()[0].memory_stats())
except Exception as e:
    print('no memory_stats', e)
