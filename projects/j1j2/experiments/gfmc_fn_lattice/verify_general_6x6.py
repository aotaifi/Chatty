"""6x6 regression of the GENERALISED walker guide (RBMPPGuideG, gfmc_fn_lib) against the exact table (copy of verify_alpha_6x6.py
with the guide provider swapped; the 6x6 path gfmc6_run.py / RBMPPGuide is unchanged and is compared row by row).
Exact 6x6 check of the one-hop guide used by the walkers (GPU, sector machinery; needs the 4.2 GB CSR).

  python verify_alpha_6x6.py OUT [--npair 300]
  1. sector vector v of the real guide (sym_tables_A.npz), E = <v|H|v>, var, Ritz alpha from the 2x2 problem in span{v, (H-E)v}
     (needs H^2 v, H^3-moment), compared with alpha = (E_1-E)/var from table_A.json (what the walker code uses);
     <H> of the p = 1 state against table_A lanczos[1].dE_site.
  2. psi_1(rep) / c(rep) constant over random representatives, c(z) = Re[exp(lP(z)) (1 + alpha (E_L(z) - E))] from rbmpp_lib
     (the walker's amplitude function): the walker guide is the table_A p = 1 state.
  3. exact E_FN of the p = 1 state (sec.fn_solve) against table_A lanczos[1].dE_FN_site = 5.5738332e-06.
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'rbmpp_6x6')); sys.path.insert(0, os.path.join(HERE, '..', 'stall_6x6'))
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import scipy.linalg
import st6_sector as SS
from gfmc6_run import make_guide, guide_constants, PARAMS, TABLE_A
from gfmc_fn_lib import RBMPPGuideG, RBMPPGuide
from rbmpp_lib import Lattice
N = 36
OUT = sys.argv[1]; npair = 300; os.makedirs(OUT, exist_ok=True)
SYMA = os.environ.get('GFMC_SYMA', '/project/theorie/a/A.Otaifi/chatty_rbmpp6/runs/rpTabA_17168764/sym_tables_A.npz')
ROOT = '/project/theorie/a/A.Otaifi/chatty_stall6'
CSR = os.environ.get('ST6_CSR', ROOT + '/csr'); TABLE = os.environ.get('ST6_TABLE', ROOT + '/data/psi0_6x6_table.npz')
log = SS.log
sec = SS.Sector(CSR, TABLE)
z = np.load(SYMA); la = jnp.asarray(z['lP']); s = jnp.asarray(z['sP'].astype(np.float32))
c = guide_constants(TABLE_A); res = dict(consts=c)
v = sec.vec(la, s); Hv = sec.H(v); E = float(v @ Hv); u1 = Hv - E * v; var = float(u1 @ u1)
Hu1 = sec.H(u1); m3 = float(u1 @ (Hu1 - E * u1))
G = np.array([[1, 0], [0, var]]); Hm = np.array([[0, var], [var, m3]])
ev, ec = scipy.linalg.eigh(Hm, G); alpha_ex = float(ec[1, 0] / ec[0, 0])
res.update(E_site=E / N, dE_site=(E - sec.E0) / N, var_site=var / N, alpha_exact=alpha_ex, alpha_table=c['alpha'],
           E_site_table=c['E_site'])
log(f'E_site {E/N:.12f} (table {c["E_site"]:.12f}) var_site {var/N:.6e} alpha exact {alpha_ex:.8f} table-formula {c["alpha"]:.8f}')
w1 = v + c['alpha'] * u1; w1n = w1 / jnp.linalg.norm(w1)
E1 = sec.energy(w1n); res['dE1_site'] = (E1 - sec.E0) / N
log(f'<H> of p=1 state dE/site {res["dE1_site"]:.6e} (table_A lanczos[1] 7.3797957e-06)')
la1 = jnp.log(jnp.maximum(jnp.abs(w1n) / sec.sqrt_n, 1e-300)); s1 = jnp.where(w1n >= 0, 1.0, -1.0).astype(jnp.float32)
# 2. per-configuration comparison with the walker's amplitude function
g_old, lat6, _ = make_guide(chunk=64)
zp = np.load(PARAMS)
g = RBMPPGuideG({k: zp[k] for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}, Lattice(6, 0.5, 'full'), c['alpha'], c['E_site'] * N, chunk=64,
                use_rbm=bool(zp['rbm']) if 'rbm' in zp.files else True)
rng = np.random.default_rng(5); idx = np.sort(rng.choice(sec.D, npair, replace=False))
reps = np.asarray(sec.reps_np)[idx].astype(np.uint64)
Z = (((reps[:, None] >> np.arange(N, dtype=np.uint64)) & np.uint64(1)).astype(np.float64) * 2 - 1)
Eloc, lP = g.amp(Z)
cz = (np.exp(lP) * (1.0 + c['alpha'] * (Eloc - c['E_site'] * N))).real
exact = np.asarray(w1n)[idx] / np.asarray(sec.sqrt_n)[idx]
ratio = exact / cz
res['ratio_mean'] = float(np.mean(ratio)); res['ratio_relspread'] = float(np.std(ratio) / abs(np.mean(ratio)))
res['max_im_over_re'] = float(np.max(np.abs((np.exp(lP) * (1.0 + c['alpha'] * (Eloc - c['E_site'] * N))).imag) / np.abs(cz)))
log(f'psi_1(exact sector)/c(walker amplitude) over {npair} reps: rel spread {res["ratio_relspread"]:.3e}, max Im/Re {res["max_im_over_re"]:.2e}')
res['PASS_amp'] = bool(res['ratio_relspread'] < 1e-8)
# deliberately broken: wrong alpha sign must fail
cz_b = (np.exp(lP) * (1.0 - c['alpha'] * (Eloc - c['E_site'] * N))).real
rb = exact / cz_b; res['broken_ratio_relspread'] = float(np.std(rb) / abs(np.mean(rb)))
log(f'broken (alpha -> -alpha) rel spread {res["broken_ratio_relspread"]:.3e} (must be >> 1e-8)')
json.dump(res, open(os.path.join(OUT, 'verify_alpha.json'), 'w'), indent=1, default=float)
# 2b. rows of the generalised provider against the unchanged 6x6 provider on Metropolis walkers (p = 1 and p = 0)
from gfmc6_run import init_walkers
Xw, _ = init_walkers(lat6, PARAMS, 200, 3, nsweep=500)
for pname, al in (('p1', c['alpha']), ('p0', 0.0)):
    go = RBMPPGuide({k: zp[k] for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}, lat6, al, c['E_site'] * N, chunk=64)
    gn = RBMPPGuideG({k: zp[k] for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}, lat6, al, c['E_site'] * N, chunk=64)
    ro, do = go.rows(Xw); rn, dn = gn.rows(Xw)
    sc = max(np.max(np.abs(ro)), np.max(np.abs(do)))
    res['rows_diff_' + pname] = float(max(np.max(np.abs(ro - rn)), np.max(np.abs(do - dn))) / sc)
    log(f'rows generalised vs old 6x6 provider ({pname}, 200 walkers): max rel diff {res["rows_diff_" + pname]:.2e}')
res['PASS_rows'] = bool(max(res['rows_diff_p1'], res['rows_diff_p0']) < 1e-9)
# 3. exact FN
_, _, info = sec.fn_solve(la1, s1)
res['dE_FN_site'] = info['dE_FN_site']; log(f'exact E_FN dE/site {info["dE_FN_site"]:.7e} (table_A 5.5738332e-06)')
json.dump(res, open(os.path.join(OUT, 'verify_alpha.json'), 'w'), indent=1, default=float)
log('DONE')
