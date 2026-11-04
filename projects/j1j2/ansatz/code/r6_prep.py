"""r6_prep.py: CPU prep of round 6, arm R6 (PLAN_pfaffian.md "Round 6 two-PP + parameter-matched control") = a COPY of r5_prep.py (sha 551ac09b; code path
unchanged) with theta_4 -> pp2_s1's theta_5 (pfaffian_r5/pp2_s1/tab/theta_5.npy, sha 4558d78f, the Paderborn race winner read by readout 17672082):
T_6 = norm_la(la(theta_5) + 4.0 * f_J^G[theta_5]), la(theta_5) = pfaffian_r5/pp2_s1/tab/tab_t5/round_1/la_table.npy (sha f5d9d1aa).
f_J^G[theta_5] (fJ_of verbatim from r5_prep.py) with E' = E_FN[theta_5, s_G] from a fresh efn_vec solve
(checked against pfaffian_r5/readout/readout.json tables.pp2_s1: e = 8.331857874447715e-06 rel 1e-8, and the E stored there).
Control: E_FN[T_6, s_G] = pfaffian_r5 readout tables.pp2_s1.scan['4.0'] = 3.258400759777026e-06 to rel 1e-6, else exit 2.   usage: python3 r6_prep.py OUTDIR"""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] PROGRESS {m}', flush=True)
RA = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5'; S1 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy'
TBAR = f'{RA}/pp2_s1/tab/tab_t5/round_1/la_table.npy'; THB = f'{RA}/pp2_s1/tab/theta_5.npy'; RDJ = f'{RA}/readout/readout.json'
CSTAR = 4.0
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RJ = f'{OUT}/prep_result.json'; res = {}
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
res.update(host=os.uname().nodename, sG=S1, sG_sha256=sha(S1), tbar_table=TBAR, tbar_table_sha256=sha(TBAR), theta_bar_sha256=sha(THB), E0=BS.E0, cstar=CSTAR)
assert res['theta_bar_sha256'].startswith('4558d78f7c46e9e6')      # pp2_s1 theta_5 (pfaffian_r5/readout/readout.json theta.pp2_s1); key names kept from r5_prep.py
RD = json.load(open(RDJ)); tb = RD['tables']['pp2_s1']; assert RD['done'] and tb['sha256'] == res['tbar_table_sha256']
assert RD['theta']['pp2_s1']['theta_5_cluster_sha256'] == res['theta_bar_sha256']
REF_E = tb['e']; REF_C4 = tb['scan']['4.0']; REF_EP = tb['E_FN']['E']; assert abs(REF_C4 - 3.258400759777026e-06) < 1e-18 and abs(REF_E - 8.331857874447715e-06) < 1e-18
res.update(ref_e_bare=REF_E, ref_e_c4=REF_C4, ref_E=REF_EP)
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
s1 = np.load(S1).astype(np.float64); assert set(np.unique(s1)) <= {-1.0, 1.0}
lF = np.load(TBAR).astype(np.float64); assert lF.shape == (D,) and np.isfinite(lF).all()
def norm_la(la): la = la - la.max(); return la - 0.5 * np.log((n * np.exp(2 * la)).sum())     # verbatim r3_prep.py / lswo2_prep.py
t = time.time(); K, Vm, D0 = BS.build(n, s1); log(f'built K,Vm (s_G) {time.time() - t:.0f}s')
t = time.time(); r0, _ = D0M.efn_vec(K, Vm, D0, n, lF); r0['sec'] = time.time() - t
res['E_bar_fresh'] = r0; rel0 = abs(r0['dE_site'] - REF_E) / REF_E; res['bar_rel'] = rel0; dump()
log(f'bar control: e {r0["dE_site"]:.12e} ref {REF_E:.12e} rel {rel0:.2e} E {r0["E"]:.12f} ref E {REF_EP:.12f}')
if rel0 > 1e-8: log('CONTROL FAIL (bar bare e): STOP'); sys.exit(2)
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
def fJ_of(lF, EP):                                                                           # verbatim r5_prep.py (= r3avg_bare.py)
    fJ = np.empty(D); nbad = 0; dmin = np.inf
    for i0 in range(0, D, 65536):
        i1 = min(D, i0 + 65536); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        ss = s1[i0:i1, None] * s1[tt]; blk = valid & (ss > 0); kpt = valid & (ss < 0)
        ratio = np.exp(np.clip((lF - lF.max())[tt] - (lF - lF.max())[i0:i1, None], -700, 700))
        VA = (blk * BS.HW[None, :] * ratio).sum(1); SK = (kpt * BS.HW[None, :] * ratio).sum(1)
        den = 27.0 - 0.5 * (valid * BS.JW[None, :]).sum(1) + VA - EP; dmin = min(dmin, float(den.min())); nbad += int((den <= 0).sum() + (SK <= 0).sum())
        with np.errstate(divide='ignore', invalid='ignore'): fJ[i0:i1] = np.log(SK) - np.log(den)
    bad = np.nonzero(~np.isfinite(fJ))[0]; info = dict(den_min=dmin, n_bad=nbad, bad_rows=[int(i) for i in bad]); assert len(bad) <= 3 and dmin > 0
    fJ[bad] = 0.0; return fJ, info
t = time.time(); fJ, info = fJ_of(lF, r0['E']); np.save(f'{OUT}/fJ_theta5s1.npy', fJ); res['fJ_info'] = info; res['fJ_sha256'] = sha(f'{OUT}/fJ_theta5s1.npy'); dump(); log(f'fJ {info} {time.time() - t:.0f}s')
lT = norm_la(lF + CSTAR * fJ); np.save(f'{OUT}/target_T6_sG.npy', lT); res['T6_sha256'] = sha(f'{OUT}/target_T6_sG.npy')
open(f'{OUT}/target_T6_sG.npy.sha256', 'w').write(res['T6_sha256'] + '  target_T6_sG.npy\n')
res['T6_norm_check'] = float((n * np.exp(2 * lT)).sum()); dump(); log(f'T6 saved sha {res["T6_sha256"]} norm {res["T6_norm_check"]}')
t = time.time(); r, _ = D0M.efn_vec(K, Vm, D0, n, lT); r['sec'] = time.time() - t
r['ref'] = REF_C4; r['rel'] = abs(r['dE_site'] - REF_C4) / REF_C4; r['pass_1e-6_rel'] = bool(r['rel'] <= 1e-6); res['E_T6_sG'] = r; res['done'] = True; dump()
log(f'VERDICT PREP: E_FN[T6,sG] {r["dE_site"]:.10e} ref {REF_C4:.10e} rel {r["rel"]:.2e} pass {r["pass_1e-6_rel"]} it {r["iters"]} res {r["res"]:.1e}')
sys.exit(0 if r['pass_1e-6_rel'] else 2)
