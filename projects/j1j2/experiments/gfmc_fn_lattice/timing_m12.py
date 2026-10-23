"""M1.2 timing: seconds per amplitude evaluation (fast-update E_L + log psi_P of one configuration) of the RBM+PP guide at
L x L for three projections (full space group x flip, translations x flip, none), random RBM+PP parameters (timing only).
  python timing_m12.py OUT.json [L:proj:chunk ...]
Per hop of the walker the one-hop guide needs 1 + n_anti(x) evaluations (x and each antiparallel neighbour); n_anti = 0.57 * NB
measured in the 6x6 pilot (82 of 144 bonds)."""
import sys, os, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'rbmpp_6x6'))
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from rbmpp_lib import Lattice, make_fns, init_params, random_configs

def set_trans(lat):
    G = lat.G; P = np.stack([G[8 * t] for t in range(lat.N)])
    IMG = np.concatenate([P, P]); FL = np.concatenate([np.ones(len(P)), -np.ones(len(P))])
    lat.proj = 'trans'; lat.IMG = jnp.asarray(IMG.astype(np.int32)); lat.FL = jnp.asarray(FL)
    lat.INV = jnp.asarray(np.argsort(IMG, axis=1).astype(np.int32)); lat.nI = IMG.shape[0]

jobs = []
for a in sys.argv[2:]:
    L, proj, ch = a.split(':'); jobs.append((int(L), proj, int(ch)))
res = []
for L, proj, ch in jobs:
    lat = Lattice(L, 0.5, 'none')
    if proj == 'trans': set_trans(lat)
    else: lat.set_proj(proj)
    p = init_params(lat, M=16, seed=1)
    fns = make_fns(lat, jnp.float64, True)
    f = jax.jit(jax.vmap(fns['eloc'], in_axes=(None, 0)))
    S = random_configs(L * L, ch, 3).astype(jnp.float64)
    t0 = time.time(); E, l = f(p, S); E.block_until_ready(); tc = time.time() - t0
    nrep = 3; t0 = time.time()
    for _ in range(nrep): E, l = f(p, S); E.block_until_ready()
    dt = (time.time() - t0) / nrep / ch
    r = dict(L=L, proj=proj, images=int(lat.nI), chunk=ch, compile_s=tc, sec_per_eval=dt, evals_per_s=1 / dt, NB=int(lat.NB))
    print(json.dumps(r), flush=True); res.append(r)
    json.dump(res, open(sys.argv[1], 'w'), indent=1)
