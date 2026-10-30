"""Matched-input kernel timing, ref vs fast: one chunk (256 configurations, the p = 1 E_L / log psi_P call that dominates GFMC) at 8x8 (production params,
met configurations of gpu_profile/acc_f64/cfg.npz) and 10x10 (random params from rbmpp_lib.init_params, 16 hidden units, random Sz = 0 configurations).
  python bench_gpu.py OUT.json --p8 params_final.npz --consts C --cfg8 cfg.npz --p10 params10_random.npz [--reps 12]
Reports ms/call (synchronised, includes host<->device copies and padding as in the GFMC driver), temp memory of the compiled executable, fast-vs-ref agreement."""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from rbmpp_lib import Lattice, random_configs
from gfmc_fn_lib import RBMPPGuideG
from gfmc_run import load_consts
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--p8', required=True); ap.add_argument('--consts', required=True)
ap.add_argument('--cfg8', required=True); ap.add_argument('--p10', required=True); ap.add_argument('--reps', type=int, default=12); ap.add_argument('--chunk', type=int, default=256)
a = ap.parse_args(); res = dict(device=str(jax.devices()[0]), jax=jax.__version__, chunk=a.chunk)
c = load_consts(a.consts, 1)
for name, L, pf, Z in (('8x8', 8, a.p8, np.load(a.cfg8)['met'][:a.chunk].astype(np.float64)),
                       ('10x10', 10, a.p10, np.asarray(random_configs(100, a.chunk, 5)).astype(np.float64))):
    z = np.load(pf); lat = Lattice(L, 0.5, 'translations'); out = {}; vals = {}
    for mode in ('ref', 'fast'):
        try:
            g = RBMPPGuideG({k: z[k] for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}, lat, c['alpha'], c['E'] * L * L / 64, w=c['w'], chunk=a.chunk, use_rbm=True, ratio=mode)
            v = g.amp(Z)                              # warm-up (compile)
            t = []
            for _ in range(a.reps):
                t0 = time.time(); v = g.amp(Z); t.append(time.time() - t0)
            ma = g._f.lower(g.p, jnp.asarray(Z)).compile().memory_analysis()
            out[mode] = dict(ms_per_call_median=1e3 * float(np.median(t)), ms_min=1e3 * min(t), ms_all=[1e3 * x for x in t], temp_GiB=ma.temp_size_in_bytes / 2**30,
                             n_fallback=g.n_fallback, images=lat.nI, NB=lat.NB)
            vals[mode] = v
            print(name, mode, out[mode]['ms_per_call_median'], 'ms', flush=True)
        except Exception as ex:
            out[mode] = dict(error=repr(ex)[:300]); print(name, mode, 'FAILED', repr(ex)[:300], flush=True)
    if 'ms_per_call_median' in out.get('ref', {}) and 'ms_per_call_median' in out.get('fast', {}):
        out['speedup_median'] = out['ref']['ms_per_call_median'] / out['fast']['ms_per_call_median']
        out['EL_rel_err_max'] = float(np.max(np.abs(vals['fast'][0] - vals['ref'][0]) / np.abs(vals['ref'][0])))
        out['logpsi_abs_err_max'] = float(np.max(np.abs(vals['fast'][1] - vals['ref'][1])))
        print(name, 'speedup', out['speedup_median'], 'EL err', out['EL_rel_err_max'], flush=True)
    res[name] = out
json.dump(res, open(a.out, 'w'), indent=1)
