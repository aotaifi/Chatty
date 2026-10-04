"""Benchmark / consistency check for ll6_core on one GPU.
Usage: python ll6_bench.py float32|float64
"""
import sys, time, json
import numpy as np
dtype = sys.argv[1]
import ll6_core as C
import jax, jax.numpy as jnp

C.log('backend', jax.default_backend(), jax.devices())
net = C.Net('vit_J2=0.50_N=6x6_k=0.mpack', dtype=dtype)
C.log('npar', net.npar)
ab = np.load('krylov_scaling_6x6_a1.20_tr4096_va2048.npz')
S = ab['train_states'].astype(np.uint64)[:4096]
res = {}
# reference: original nqsmagic network, fp64 (as all previous runs)
import netket.jax as nkjax, flax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d
m0 = ViT(num_layers=4, d_model=60, heads=10, L_eff=9, b=2, transl_invariant=True, two_dimensional=True)
v = m0.init(jax.random.PRNGKey(1234), jnp.zeros((1, C.N)))
obj = flax.serialization.msgpack_restore(open('vit_J2=0.50_N=6x6_k=0.mpack', 'rb').read())
if 'variables' in obj: obj = obj['variables']
v = flax.serialization.from_state_dict(v, obj)
ref = np.asarray(jax.jit(lambda x: _logpsi_transl_2d(m0.apply, 2, v, x))(jnp.asarray(C.bits_to_spins_np(S)))).real
mine = net.logabs(net.flat0, S)
res['max_abs_dlog_vs_nqsmagic'] = float(np.max(np.abs(mine - ref)))
res['rms_dlog_vs_nqsmagic'] = float(np.sqrt(np.mean((mine - ref - np.mean(mine - ref)) ** 2)))
C.log('CHECK', res)
# throughput
big = np.tile(S, 64)
t = time.time(); net.logabs(net.flat0, big[:16384]); t0 = time.time()
net.logabs(net.flat0, big); dt = time.time() - t0
res['evals_per_s'] = len(big) / dt
C.log('THROUGHPUT', res['evals_per_s'])
A0 = C.Amp(net, net.flat0)
g1 = C.Guide([A0], [C.T_FN], A0)
s1 = g1.signs(S[:1024])
res['K1_flip_frac'] = float(np.mean(s1 != C.marshall_vec(S[:1024])))
np.save(f'bench_K1signs_{dtype}.npy', s1)
# level sizes
for D, n in ((1, 16), (2, 16), (3, 4), (3, 64), (4, 1)):
    t = time.time(); lv, nb = C.build_levels(S[:n], D); dt = time.time() - t
    res[f'levels_D{D}_n{n}'] = dict(sizes=[len(x) for x in lv], sec=dt)
    C.log('LEVELS', D, n, [len(x) for x in lv], dt)
# local data K=1 (D=2) for 128 states; K=2-like cost: threshold edges with D=3 on 16 states
t = time.time(); n0 = net.neval; d = g1.local(S[:128]); res['local_K1_128'] = dict(sec=time.time() - t, evals=net.neval - n0)
C.log('LOCAL K1', res['local_K1_128'], 'eh mean/site', d['eh'].mean() / 36)
t = time.time(); n0 = net.neval; E = C.threshold_edges(g1, A0, S[:16]); res['thr_edges_K1_16'] = dict(sec=time.time() - t, evals=net.neval - n0)
C.log('THR EDGES', res['thr_edges_K1_16'])
# jacobian
t = time.time(); O = net.jac(net.flat0, S[:2048]); O.block_until_ready(); res['jac_2048_sec'] = time.time() - t
t = time.time(); O = net.jac(net.flat0, S[:2048]); O.block_until_ready(); res['jac_2048_sec_2nd'] = time.time() - t
C.log('JAC', O.shape, O.dtype, res['jac_2048_sec'], res['jac_2048_sec_2nd'])
del O
# sampler
ch = C.Chains(net, 2048, 1)
t = time.time(); ch.advance(net.flat0, 1); t1 = time.time(); X = ch.advance(net.flat0, 4); t2 = time.time()
res['sampler_2048x4sweeps_sec'] = t2 - t1; res['sampler_first_sec'] = t1 - t; res['accept'] = ch.acc / ch.props
C.log('SAMPLER', res)
json.dump(res, open(f'bench_{dtype}.json', 'w'), indent=1)
