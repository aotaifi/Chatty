"""Measured evaluation cost of the residual-re-base guide vs k (README round 4).
t_vit = one symmetrised psi_P evaluation (16 images x 4 patch translations, fp32); t_net = one small FEAT net (16
images, fp32), per configuration, batch 4096, median of 5 after warm-up. Cost per configuration:
  guide value G_k(x):     n1 t_vit + k t_net               (n1 = 1 + one-hop count)
  guide V, W at x:        n2 t_vit + n1 k t_net            (n2 = distinct configurations within two hops)
  edge-loss sample:       (1 + K) x the V, W cost (upper bound; overlaps reduce it)
Hop counts: writeback_tail/hopcount.json (6x6: 77 / 2.8e3; 8x8: 134 / 8.7e3).
  python cx_cost.py OUT
"""
import os, sys, time
import numpy as np
from cx_lib import *
import flax, vit_dt

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
lab = Lab(); sec = lab.sec
vit_dt.set_dtype('float32')
vit = vit_dt.make_model_6x6()
tmpl = vit.init(jax.random.PRNGKey(0), jnp.zeros((1, N), jnp.float32))
obj = flax.serialization.msgpack_restore(open(VIT_CKPT, 'rb').read()); obj = obj.get('variables', obj)
p = jax.tree_util.tree_map(lambda q: jnp.asarray(q, jnp.float32), flax.serialization.from_state_dict(tmpl, obj)['params'])


@jax.jit
def vit_eval(Sx):
    X = bits(Sx); B_ = X.shape[0]
    Xg = jnp.stack([X[:, INV8[k // 2]] * SG2[k % 2] for k in range(16)], 0).reshape(16 * B_, N)
    zs = vit_dt.logpsi_transl_2d(vit.apply, 2, {'params': p}, Xg).reshape(16, B_)
    m = jnp.max(jnp.real(zs), 0)
    return m + jnp.log(jnp.abs(jnp.sum(jnp.exp(zs - m[None, :]), 0)))


FEAT_BASE = jnp.zeros((sec.D, 3), jnp.float32)
net = FeatNet(FEAT_BASE, 0)
xs = sec.reps[:4096]; ts = lab.TIDX[:4096]


def tm(fn, *a):
    jax.block_until_ready(fn(*a)); out = []
    for _ in range(5):
        t0 = time.time(); jax.block_until_ready(fn(*a)); out.append(time.time() - t0)
    return float(np.median(out)) / 4096
t_vit = tm(vit_eval, xs); t_net = tm(net._tab, net.flat0, xs, ts)
res = dict(t_vit=t_vit, t_net=t_net, gpu=os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip(), table=[])
for L, n1, n2 in ((6, 78, 2834), (8, 135, 8749)):
    for k in range(0, 9):
        g = n1 * t_vit + k * t_net
        vw = n2 * t_vit + n1 * k * t_net
        res['table'].append(dict(L=L, k=k, guide_s=g, guide_VW_s=vw, edge_sample_s=9 * vw,
                                 small_net_share_guide=k * t_net / g, small_net_share_VW=n1 * k * t_net / vw))
dump(os.path.join(OUT, 'cx_cost.json'), res)
log(f'[cost] t_vit {t_vit:.3e} s  t_net {t_net:.3e} s per configuration ({res["gpu"]})')
for r in res['table']:
    log(f'  L={r["L"]} k={r["k"]}: guide {r["guide_s"]:.3e} s  V,W {r["guide_VW_s"]:.3e} s  edge sample {r["edge_sample_s"]:.3e} s  small-net share {r["small_net_share_guide"]:.3f}/{r["small_net_share_VW"]:.3f}')
