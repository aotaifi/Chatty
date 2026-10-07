"""Recompute the exact sector FN energy (st6_sector.Sector.fn_solve) of the three calibration guides from the SAME tables
that fncal_dmc.py uses (fncal_dmc.load_guide), to rule out any guide mismatch between DMC and the stall_6x6 numbers.
Run on a full A40 (sector H is 5 GB on device).  Output: exact_fn.json"""
import json, sys, os, time
import numpy as np
import jax, jax.numpy as jnp
import st6_sector as SS
import fncal_dmc as F

CSR = os.environ.get('ST6_CSR', '/project/theorie/a/A.Otaifi/chatty_stall6/csr')
TABLE = os.path.join(F.DATA, 'psi0_6x6_table.npz')
sec = SS.Sector(CSR, TABLE)
out = {}
for g in ['vitvit', 'vitex', 'exvit']:
    reps, la, s = F.load_guide(g)
    assert np.array_equal(reps, sec.reps_np)
    la = jnp.asarray(la); s = jnp.asarray(s)
    sc = sec.score(la, s)
    _, _, info = sec.fn_solve(la, s, tol=1e-12, maxit=600)
    out[g] = dict(E_FN_site=info['E_FN_site'], dE_FN_site=info['dE_FN_site'], res=info['res'], neg_weight=info['neg_weight'],
                  H_site=sc['E_site'], stall_value=F.EXACT[g], diff_vs_stall=info['E_FN_site'] - F.EXACT[g])
    SS.log(g, json.dumps(out[g]))
    json.dump(out, open('exact_fn.json', 'w'), indent=1)
