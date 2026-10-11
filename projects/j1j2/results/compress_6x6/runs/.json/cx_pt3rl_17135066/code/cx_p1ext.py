"""P1 extension (README amendment 1): the pre-registered etas (0.01, 0.03, 0.1) all lost > 10% of the step, so eta*
is located with eta = 0.001, 0.003 (same noise models and seeds rule as cx_replay.py; guides from the replay tables).
  python cx_p1ext.py OUT
"""
import os, sys, time
import numpy as np
from cx_lib import *

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_p1ext.json'); T00 = time.time()
lab = Lab(); sec = lab.sec
res = dict(p1=[])
for k in (1, 2, 3):
    la, s = (lab.lP0, lab.sP0) if k == 1 else load_guide(k - 1)
    Efn, u, info = sec.fn_solve(la, s); del u
    params = load_loop_params(k)
    PA = lab.UA(la) ** 2
    lan, V, W = lab.VW(la, s)
    for model in ('VW', 'bond'):
        for eta in (0.001, 0.003):
            key = jax.random.PRNGKey(1000 * k + int(round(eta * 1000)) + (0 if model == 'VW' else 500) + 77)
            if model == 'VW':
                xi = jax.random.normal(key, (2, sec.D), f64)
                feats = (lan, V * jnp.exp(eta * xi[0]), W * jnp.exp(eta * xi[1])); del xi
            else:
                xi = jax.random.normal(key, (sec.D,), f64)
                feats = lab.VW(la + (eta / np.sqrt(2.0)) * xi, s); del xi
            okv = V > 1e-12
            lv = jnp.log(jnp.maximum(feats[1], 1e-300) / jnp.maximum(V, 1e-300))
            lw = jnp.log(jnp.maximum(feats[2], 1e-300) / jnp.maximum(W, 1e-300))
            rv = float(jnp.sqrt(jnp.sum(jnp.where(okv, PA * lv * lv, 0)) / jnp.sum(jnp.where(okv, PA, 0))))
            rw = float(jnp.sqrt(jnp.sum(PA * lw * lw))); del lv, lw
            _, _, r = loop_step(lab, la, s, params, Efn=Efn, feats=feats, krylov=False)
            del feats
            res['p1'].append(dict(it=k, model=model, eta=eta, frac=r['frac'], SI_target_frac=r['SI_target_frac'],
                                  logV_rms_pa=rv, logW_rms_pa=rw)); dump(F, res)
            log(f'[P1ext] it {k} {model} eta {eta}: frac {r["frac"]:.4f} SI {r["SI_target_frac"]:.4f} logV {rv:.4f} logW {rw:.4f}')
    del lan, V, W, PA, la, s
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
