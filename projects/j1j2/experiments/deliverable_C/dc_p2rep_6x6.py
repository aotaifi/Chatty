"""Exact 6x6 check for the p = 2 coefficient choice (PREREG amendment 3): how much of the optimal p = 2 Lanczos gain does
the repeated step psi_rep = (1 + alpha H)^2 psi_P keep (alpha = optimal p = 1 coefficient), and the Lanczos step on psi_1
(optimal second step)?  Exact symmetric sector, code of experiments/lanczos_baseline_6x6.  Usage: python dc_p2rep_6x6.py OUT
"""
import json, os, sys
H_ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H_, '..', 'stall_6x6')); sys.path.insert(0, os.path.join(H_, '..', 'lanczos_baseline_6x6'))
import numpy as np
import jax.numpy as jnp
import st6_sector as SS
from lanczos_lib import lanczos_ritz, extrapolate
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
sec = SS.Sector(os.environ.get('ST6_CSR', '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr'),
                os.environ.get('ST6_TABLE', '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz'))
DATA = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/data'; N = 36; E0 = sec.E0
z = np.load(os.path.join(DATA, 'sym_tables.npz')); v = sec.vec(jnp.asarray(z['lP']), jnp.asarray(z['sP'].astype(np.float32)))


def ev(x):
    x = x / float(x @ x) ** 0.5; h = sec.Hm(x); E = float(x @ h); return dict(dE_site=(E - E0) / N, var_site=(float(h @ h) - E * E) / N)


r = lanczos_ritz(lambda x: sec.Hm(x), v, 2); al = r[1]['alpha']
v1 = v + al * sec.Hm(v); vrep = v1 + al * sec.Hm(v1)
r1 = lanczos_ritz(lambda x: sec.Hm(x), v1, 1)
res = dict(alpha=al, p0=ev(v), p1=ev(v1), p2_opt=dict(dE_site=(r[2]['E'] - E0) / N, var_site=r[2]['var'] / N),
           p2_repeated=ev(vrep), p2_lanczos_on_psi1=dict(dE_site=(r1[1]['E'] - E0) / N, var_site=r1[1]['var'] / N))
g_opt = res['p1']['dE_site'] - res['p2_opt']['dE_site']
res['repeated_keeps_fraction_of_opt_gain'] = (res['p1']['dE_site'] - res['p2_repeated']['dE_site']) / g_opt
res['lanczos_on_psi1_keeps_fraction'] = (res['p1']['dE_site'] - res['p2_lanczos_on_psi1']['dE_site']) / g_opt
for k in ('p2_opt', 'p2_repeated'):
    res['extrap_p012_' + k] = (extrapolate([res['p0']['dE_site'], res['p1']['dE_site'], res[k]['dE_site']],
                                           [res['p0']['var_site'], res['p1']['var_site'], res[k]['var_site']]))
print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(OUT, 'p2rep_6x6.json'), 'w'), indent=1)
