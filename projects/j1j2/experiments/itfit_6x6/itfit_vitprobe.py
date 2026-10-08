"""Throughput probe for parametrisation (a) of experiments/itfit_6x6: fine-tuning the symmetrised 6x6 ViT itself.
log|psi_P,theta(x)| = log|sum_{16 D4 x flip images} psi_ViT(g x)| (each psi_ViT = logsumexp over 4 patch translations),
i.e. 64 ViT passes per configuration.  Times forward and per-sample gradients; extrapolates the cost of one exact
table and of one Gauss-Newton fit iteration of the ITE arms.   python itfit_vitprobe.py OUT
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p_ in (HERE, os.path.join(HERE, '..', 'learned_loop_6x6'), os.path.join(HERE, '..', 'stall_6x6')):
    if p_ not in sys.path: sys.path.insert(0, p_)
import jax
import jax.numpy as jnp
import flax
from jax.flatten_util import ravel_pytree
import vit_dt
import st6_sector as SS
jax.config.update('jax_default_matmul_precision', 'highest')
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
CKPT = '/project/theorie/a/A.Otaifi/chatty_stall6/data/vit_J2=0.50_N=6x6_k=0.mpack'
N = 36
vit_dt.set_dtype('float32')
vit = vit_dt.make_model_6x6()
tmpl = vit.init(jax.random.PRNGKey(1234), jnp.zeros((1, N), jnp.float32))
obj = flax.serialization.msgpack_restore(open(CKPT, 'rb').read())
if 'variables' in obj: obj = obj['variables']
tmpl = flax.serialization.from_state_dict(tmpl, obj)
params = jax.tree_util.tree_map(lambda q: jnp.asarray(q, jnp.float32), tmpl['params'])
flat0, unravel = ravel_pytree(params)
INV = jnp.asarray(np.argsort(SS.space_group()[:8], axis=1)); SG = jnp.asarray([1.0, -1.0], jnp.float32)


def f(flat, X):
    p = {'params': unravel(flat)}
    zs = jnp.stack([vit_dt.logpsi_transl_2d(vit.apply, 2, p, X[:, INV[k // 2]] * SG[k % 2]) for k in range(16)], 0)
    m = jnp.max(jnp.real(zs), 0)
    return m + jnp.log(jnp.abs(jnp.sum(jnp.exp(zs - m[None, :]), 0)))


fj = jax.jit(f)
jac = jax.jit(jax.vmap(jax.grad(lambda q, x: f(q, x[None])[0]), in_axes=(None, 0)))
out = dict(npar=int(flat0.size))
for B in (256, 1024):
    X = jnp.asarray(np.random.default_rng(0).choice([-1.0, 1.0], size=(B, N)).astype(np.float32))
    fj(flat0, X).block_until_ready(); t = time.time()
    for _ in range(3): fj(flat0, X).block_until_ready()
    out[f'fwd_sec_per_config_B{B}'] = (time.time() - t) / 3 / B
for B in (32, 128):
    X = jnp.asarray(np.random.default_rng(1).choice([-1.0, 1.0], size=(B, N)).astype(np.float32))
    try:
        jac(flat0, X).block_until_ready(); t = time.time()
        for _ in range(2): jac(flat0, X).block_until_ready()
        out[f'jac_sec_per_config_B{B}'] = (time.time() - t) / 2 / B
    except Exception as e:
        out[f'jac_B{B}_error'] = str(e)[:200]
fw = out['fwd_sec_per_config_B1024']; jc = out.get('jac_sec_per_config_B128', out.get('jac_sec_per_config_B32', float('nan')))
D = 15804956
out['table_sec_est'] = fw * D
# one Q-fit GN iteration (B = 1024, K_b = 4): Jacobian rows 5120, forward shells 2 x 5120 x ~79 configurations
out['gn_iter_sec_est_shells'] = fw * 2 * 5120 * 79 + jc * 5120
out['gn_iter_sec_est_tables'] = jc * 5120
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(OUT, 'vitprobe.json'), 'w'), indent=1)
