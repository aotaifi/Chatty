"""Cost per fast-update evaluation (E_L + log psi_P, and the variant that also returns the NB bond ratios) of the RBM+PP guide
at L x L for a real parameter file, as a function of the chunk (batch) size, on Metropolis configurations.
  python timing8.py OUT.json PARAMS.npz [L proj chunk chunk ...]
"""
import sys, os, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'rbmpp_6x6'))
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from rbmpp_lib import Lattice, make_fns, make_sampler, random_configs, to32
out, pf = sys.argv[1], sys.argv[2]
L = int(sys.argv[3]) if len(sys.argv) > 3 else 8; proj = sys.argv[4] if len(sys.argv) > 4 else 'translations'
chunks = [int(c) for c in sys.argv[5:]] or [64, 128, 256, 512, 1024]
z = np.load(pf); use_rbm = bool(z['rbm']) if 'rbm' in z.files else True
p = {k: jnp.asarray(z[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}
lat = Lattice(L, 0.5, proj); N = lat.N
f64 = make_fns(lat, jnp.float64, use_rbm)
fe = jax.jit(jax.vmap(f64['eloc'], in_axes=(None, 0))); ff = jax.jit(jax.vmap(f64['eloc_full'], in_axes=(None, 0)))
sw = make_sampler(lat, make_fns(lat, jnp.float32, use_rbm)); S = random_configs(N, max(chunks), 5); key = jax.random.PRNGKey(0)
for _ in range(10):
    key, k = jax.random.split(key); S, acc = sw(to32(p), S, k, N)
Z = np.asarray(S).astype(np.float64)
dev = jax.devices()[0]; res = []
for ch in chunks:
    r = dict(L=L, proj=proj, images=int(lat.nI), chunk=ch)
    for name, fn in (('eloc', fe), ('eloc_full', ff)):
        try:
            t0 = time.time(); o = fn(p, jnp.asarray(Z[:ch])); jax.block_until_ready(o); tc = time.time() - t0
            t0 = time.time()
            for _ in range(5): o = fn(p, jnp.asarray(Z[:ch])); jax.block_until_ready(o)
            r[name + '_ms'] = (time.time() - t0) / 5 / ch * 1e3; r[name + '_compile_s'] = tc
        except Exception as e:
            r[name + '_err'] = str(e)[:100]
    try: r['peak_gb'] = dev.memory_stats()['peak_bytes_in_use'] / 1e9
    except Exception: pass
    print(json.dumps(r), flush=True); res.append(r); json.dump(res, open(out, 'w'), indent=1)
print('TIMING_DONE', flush=True)
