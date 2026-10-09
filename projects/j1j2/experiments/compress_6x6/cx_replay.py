"""P1 + references: replay the wtLOOP8 stack exactly from its stored FEAT nets (read-only), save the guides
(la_k, s_k), k = 1..6, and measure the sensitivity of one write-back step to errors in V and W (README, P1).

P1 noise models (iterations 1-3, guide k-1 exact, stored net k, write-back applied to the exact guide):
  'VW'  : V -> V exp(eta xi_V), W -> W exp(eta xi_W), xi iid N(0,1) per orbit (as reviewer E5(i));
  'bond': the guide used for V, W and the log-a input is la + (eta/sqrt 2) xi, i.e. a student with white pointwise
          error whose bond log-ratio error has rms eta (partly coherent across the bonds of one configuration).
  Each perturbation enters the net input and the semi-implicit target T.
  python cx_replay.py OUT
"""
import os, sys, time
import numpy as np
from cx_lib import *
from lanczos_lib import lanczos_ritz

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_replay.json'); T00 = time.time()
J8 = json.load(open(os.path.join(LOOP8, 'wt_loop.json')))
lab = Lab(); sec = lab.sec
res = dict(iters=[], p1=[], refs={}, loop8=[{k: it[k] for k in ('frac', 'H_kry', 'E_FN_next', 'SI_target_frac')}
                                              for it in J8['iters']])
ETAS = (0.01, 0.03, 0.1)
la, s = lab.lP0, lab.sP0
efn = {}
for k in range(1, 8):
    t0 = time.time()
    Efn = None
    if k in (1, 2, 3, 4, 7):
        Efn, u, info = sec.fn_solve(la, s); del u
        efn[k - 1] = info['dE_FN_site']
    params = load_loop_params(k)
    if k <= 3:                                                  # ---- P1 sensitivity
        PA = lab.UA(la) ** 2
        lan, V, W = lab.VW(la, s)
        for model in ('VW', 'bond'):
            for eta in ETAS:
                key = jax.random.PRNGKey(1000 * k + int(round(eta * 1000)) + (0 if model == 'VW' else 500))
                if model == 'VW':
                    xi = jax.random.normal(key, (2, sec.D), f64)
                    feats = (lan, V * jnp.exp(eta * xi[0]), W * jnp.exp(eta * xi[1])); del xi
                else:
                    xi = jax.random.normal(key, (sec.D,), f64)
                    feats = lab.VW(la + (eta / np.sqrt(2.0)) * xi, s); del xi
                lv = jnp.log(jnp.maximum(feats[1], 1e-300) / jnp.maximum(V, 1e-300))
                lw = jnp.log(jnp.maximum(feats[2], 1e-300) / jnp.maximum(W, 1e-300))
                okv = V > 1e-12
                rv = float(jnp.sqrt(jnp.sum(jnp.where(okv, PA * lv * lv, 0)) / jnp.sum(jnp.where(okv, PA, 0))))
                rw = float(jnp.sqrt(jnp.sum(PA * lw * lw)))
                del lv, lw
                _, _, r = loop_step(lab, la, s, params, Efn=Efn, feats=feats, krylov=False)
                del feats
                rec = dict(it=k, model=model, eta=eta, frac=r['frac'], SI_target_frac=r['SI_target_frac'],
                           logV_rms_pa=rv, logW_rms_pa=rw)
                res['p1'].append(rec); dump(F, res)
                log(f'[P1] it {k} {model} eta {eta}: frac {r["frac"]:.4f}  SI {r["SI_target_frac"]:.4f}  '
                    f'rms logV {rv:.4f} logW {rw:.4f}')
        del lan, V, W, PA
    la_new, s_new, rec = loop_step(lab, la, s, params, Efn=Efn, krylov=True)
    rec['it'] = k
    if Efn is not None: rec['guide_E_FN'] = efn[k - 1]
    ref = J8['iters'][k - 1]
    rec['loop8_frac'] = ref['frac']; rec['loop8_H_kry'] = ref['H_kry']
    rec['sec'] = time.time() - t0
    log(f'== replay it {k}: frac {rec.get("frac", float("nan")):.6f} (loop8 {ref["frac"]:.6f})  '
        f'<H> {rec["H_kry"]:.6e} (loop8 {ref["H_kry"]:.6e})  {rec["sec"]:.0f}s')
    res['iters'].append(rec); dump(F, res)
    if k <= 6: save_guide(k, la_new, s_new)
    la, s = la_new, s_new
del la, s

# ---- references for the distillation scores at k = 3 and 6 (same sign s_k throughout)
for k in (3, 6):
    t0 = time.time()
    la_k, s_k = load_guide(k)
    r = dict(k=k)
    r['H_stack'] = lab.H_site(la_k, s_k)
    r['E_FN_stack'] = efn[k]
    r['H_base_sk'] = lab.H_site(lab.lP0, s_k)
    _, u, i0 = sec.fn_solve(lab.lP0, s_k); del u
    r['E_FN_base_sk'] = i0['dE_FN_site']
    r['frozen_stack'] = (lab.frozen(la_k, s_k, la_k) - sec.E0) / N
    r['frozen_base'] = (lab.frozen(la_k, s_k, lab.lP0) - sec.E0) / N
    rl = lanczos_ritz(lambda x: sec.Hm(x), sec.vec(la_k, s_k), 1)[1]
    r['lanczos_H_stack'] = (rl['E'] - sec.E0) / N; del rl
    r['continuation_frac_table'] = res['iters'][k]['frac']             # iteration k+1 from the exact stack guide
    r['continuation_H_table'] = res['iters'][k]['H_kry']
    dec, _ = lab.decade_of(la_k)
    smp = sample_by_decade(lab, dec, 2048, seed=7)
    r['base_fidelity'] = fidelity(lab, la_k, s_k, smp, lambda idx, v, iy: lab.lP0[iy] - lab.lP0[idx][:, None],
                                  la_s=lab.lP0)
    r['decade_mass'] = {int(d): float(jnp.sum(jnp.where(dec == d, lab.UA(la_k) ** 2, 0))) for d in range(-16, -5)}
    r['sec'] = time.time() - t0
    res['refs'][str(k)] = r; dump(F, res)
    log(f'[refs k={k}] ' + json.dumps({q: v for q, v in r.items() if q not in ('base_fidelity', 'decade_mass')}))
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
