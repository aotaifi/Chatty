"""Unprojected reference for Stage 1 of experiments/itfit_6x6: the FN/Krylov loop whose amplitude update is K exact
semi-implicit ITE steps on the frozen F_k (no network, no fit), i.e. what projected ITE would give with perfect fits.
  python itfit_exactloop.py OUT SPEC.json     SPEC: {"variants": [[tau, K], ...], "n_loop": 3}
"""
import json, os, sys, time
import numpy as np
from itfit_common import Exact, log, dump, N, E0_SITE
import jax.numpy as jnp

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F_OUT = os.path.join(OUT, 'itfit_exactloop.json')
ex = Exact(); sec = ex.sec
res = dict(spec=SPEC, variants=[])
es = lambda E: E / N - E0_SITE
for tau, K in SPEC['variants']:
    t0 = time.time()
    la, s = ex.lP, ex.sP
    rows = []
    for k in range(SPEC.get('n_loop', 3)):
        Efn, u, _ = sec.fn_solve(la, s)
        D, w, Kw = ex.fn_diag(la, s)
        b = w / jnp.linalg.norm(w); Kb = Kw / jnp.linalg.norm(w); del Kw
        Eg = float(b @ (D * b) - b @ Kb)
        E = Eg
        for n in range(K):
            t = (b + tau * Kb) / (1.0 + tau * (D - E))
            b = t / jnp.linalg.norm(t); Kb = ex.Kop(b, s); E = float(b @ (D * b) - b @ Kb)
        la_new = jnp.log(jnp.maximum(b / ex.sqn, 1e-300))
        H_old = ex.energy_H(la_new, s)
        s_new, E_kry, _ = sec.krylov(b, s)
        rows.append(dict(it=k + 1, E_FN_guide_dE_site=es(Efn), G_site=(Eg - Efn) / N, frac=(Eg - E) / (Eg - Efn),
                         H_oldsign_dE_site=es(H_old), H_kry_dE_site=es(E_kry)))
        log(f'tau {tau} K {K} it {k + 1}:', rows[-1])
        la, s = la_new, s_new
        del D, w, b, Kb, u
    Efn, _, _ = sec.fn_solve(la, s)
    res['variants'].append(dict(tau=tau, K=K, loop=rows, final_E_FN_dE_site=es(Efn), sec=time.time() - t0))
    log(f'tau {tau} K {K}: final E_FN {es(Efn):.4e}')
    dump(F_OUT, res)
log('DONE')
