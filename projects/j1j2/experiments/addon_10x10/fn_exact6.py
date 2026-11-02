"""A0.4 item 2 reference: exact lattice FN energy of the 6x6 RBM+PP guide (table A: per-rep log amplitude lP, own sign sP) with the
project's exact solver st6_sector.Sector.fn_solve (copy m1_code/st6_sector.py, md5 b0977569 = experiments/stall_6x6), and the exact
energy of the guide itself.  Compared with table_A.json guide.dE_FN_site = 3.524943767886491e-05.
  python fn_exact6.py OUT.json      (CPU or GPU JAX; ~15 GB host memory on CPU)"""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'm1_code'))
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import st6_sector as SS
R = '/project/theorie/a/A.Otaifi/Chatty/runs'
sec = SS.Sector(R + '/stall6/csr', R + '/stall6/data/psi0_6x6_table.npz')
z = np.load(R + '/rbmpp6/runs/rpTabA_17168764/sym_tables_A.npz')
la = jnp.asarray(z['lP']); s = jnp.asarray(z['sP'].astype(np.float64))
t0 = time.time(); v = sec.vec(la, s); E = sec.energy(v); SS.log(f'guide E_site {E / 36:.12f} dE_site {(E - sec.E0) / 36:.6e} ({time.time() - t0:.0f}s)')
E_FN, _, info = sec.fn_solve(la, s, verbose=True)
info.update(E0_site=sec.E0 / 36, guide_E_site=E / 36, guide_dE_site=(E - sec.E0) / 36, table_A_dE_FN_site=3.524943767886491e-05,
            table_A_guide_dE_site=4.7420471760675457e-05)
SS.log(f"FN_EXACT dE_FN_site {info['dE_FN_site']:.10e} E_FN_site {info['E_FN_site']:.12f} (table_A 3.5249437679e-05) it {info['lobpcg_it']} res {info['res']:.1e}")
json.dump(info, open(sys.argv[1], 'w'), indent=1, default=float)
