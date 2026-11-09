"""k3_splice_e.py: CPU check 'mu = 0 reproduces e(theta_5)' of the K=3 round-6 test (PLAN_pfaffian.md "K=3 ROUND-6 TEST at 6x6").
The full K=3 table (15.8M orbits x 576 images x 3 dets) is an H100 job (CPU ~ days), so: the stored exact table of theta_5 (pfaffian_r5/pp2_s1/tab/tab_t5, sha f5d9d1aa) is spliced, on the orbit index set
ix of the pp3 self test (all orbits of decades -5..-8 + a per-decade sample below), with the K=3 model's own ln|B| at mu = 0 (a_la_mu0.npy) and, as the broken control, at mu = mu_init (a_la_mu_init.npy);
bare e of each spliced table through the readout's solver (delta0.efn_vec, s_G).  Reports rel to e(theta_5) = 8.331857874447715e-06 and the share of the norm weight inside ix.
usage: python3 k3_splice_e.py SELFTEST_DIR OUT.json"""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
ST, OUTJ = sys.argv[1:3]; T0 = time.time(); log = lambda m: print(f'[{time.time() - T0:6.0f}s] {m}', flush=True)
R5 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5'; E5 = 8.331857874447715e-06
TAB = f'{R5}/pp2_s1/tab/tab_t5/round_1/la_table.npy'; SG = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest(); assert sha(TAB).startswith('f5d9d1aa')
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s1 = np.load(SG).astype(np.float64); K, Vm, D0 = BS.build(n, s1); log('K, Vm built')
la = np.load(TAB).astype(np.float64); ix = np.load(f'{ST}/a_orbit_index.npy'); l0 = np.load(f'{ST}/a_la_mu0.npy'); lm = np.load(f'{ST}/a_la_mu_init.npy')
w = n * np.exp(2 * (la - la.max())); res = dict(table=TAB, table_sha256=sha(TAB), n_spliced=int(ix.size), norm_weight_share_in_ix=float(w[ix].sum() / w.sum()), max_abs_dev_mu0=float(np.max(np.abs(l0 - la[ix]))),
                                           max_abs_dev_mu_init=float(np.max(np.abs(lm - la[ix]))), e_ref=E5)
for name, v in (('control_unspliced', None), ('splice_mu0', l0), ('splice_mu_init_BROKEN', lm)):
    t = la.copy()
    if v is not None: t[ix] = v
    r, _ = D0M.efn_vec(K, Vm, D0, n, t); e = r['dE_site']; res[name] = dict(e=e, rel=(e - E5) / E5, iters=r['iters'], res=r['res']); log(f'{name}: e {e:.12e} rel {(e - E5) / E5:.2e}')
    json.dump(res, open(OUTJ, 'w'), indent=1, default=float)
res['pass_mu0_rel_1e-9'] = bool(abs(res['splice_mu0']['rel']) <= 1e-9); res['broken_fails'] = bool(abs(res['splice_mu_init_BROKEN']['rel']) > 1e-9); json.dump(res, open(OUTJ, 'w'), indent=1, default=float); log(f'DONE {res["pass_mu0_rel_1e-9"]} {res["broken_fails"]}')
