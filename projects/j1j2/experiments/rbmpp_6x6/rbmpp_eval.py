"""Independent VMC energy evaluations of an RBM+PP parameter file (fresh random starts, own seed per chain-set).

  python rbmpp_eval.py OUT SPEC.json
SPEC: params (npz), L [6], J2 [0.5], proj ['translations'], seeds [[11, 12]], chains [2048], nsamp [per seed, samples
  = chains x sweeps], therm [60 sweeps], sweep [N proposals between samples], chunk [128], M [16].
Per seed writes OUT/E_s<seed>.npy: the time series (one entry per sweep) of the mean over chains of Re E_loc / N, in
time order (input of tools/quotable.py series), and OUT/eval.json with the chain-mean error (chains are independent
starts), the Var(E_loc)/N and the acceptance. Seeds give independent starts AND independent RNG streams; the
parameters are the same file, so the seeds are replicas of the sampler only (not of the training).
"""
import json, os, sys, time
import numpy as np
import jax
import jax.numpy as jnp
from rbmpp_lib import Lattice, make_fns, make_sampler, random_configs, to32
jax.config.update("jax_enable_x64", True)

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
g = SPEC.get
z = np.load(g('params'))
use_rbm = bool(z['rbm']) if 'rbm' in z.files else True
p = {k: jnp.asarray(z[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}
lat = Lattice(g('L', 6), g('J2', 0.5), g('proj', 'translations')); N = lat.N
f64 = make_fns(lat, jnp.float64, use_rbm); f32 = make_fns(lat, jnp.float32, use_rbm)
ev = jax.jit(jax.vmap(f64['eloc'], in_axes=(None, 0))); sw = make_sampler(lat, f32)
p32 = to32(p); CH = g('chunk', 128); nsw = g('sweep', N); C = g('chains', 2048)
res = dict(spec=SPEC, images=lat.nI, runs=[])
for seed in g('seeds', [11, 12]):
    t0 = time.time()
    key = jax.random.PRNGKey(5000 + seed); Sx = random_configs(N, C, 9000 + seed)
    for _ in range(g('therm', 60)):
        key, k = jax.random.split(key); Sx, acc = sw(p32, Sx, k, nsw)
    means = []; accs = []
    while len(means) * C < g('nsamp', 400000):
        key, k = jax.random.split(key); Sx, acc = sw(p32, Sx, k, nsw); accs.append(float(acc))
        El = jnp.concatenate([ev(p, Sx[i:i + CH])[0] for i in range(0, C, CH)])
        means.append(np.asarray(jnp.real(El)))
        if len(means) % 20 == 0:
            print(f'PROGRESS seed {seed} sweeps {len(means)} E/N {np.mean(means) / N:.6f} ({time.time() - t0:.0f}s)', flush=True)
    A = np.array(means)
    np.save(os.path.join(OUT, f'E_s{seed}.npy'), A.mean(1) / N)
    cm = A.mean(0)
    r = dict(seed=seed, E=A.mean() / N, err_chains=cm.std(ddof=1) / np.sqrt(C) / N, var=float(A.var()) / N,
             nsamp=int(A.size), sweeps=A.shape[0], acc=float(np.mean(accs)), sec=time.time() - t0)
    res['runs'].append(r); print('EVAL', r, flush=True)
    json.dump(res, open(os.path.join(OUT, 'eval.json'), 'w'), indent=1, default=float)
print('DONE', flush=True)
