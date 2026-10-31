"""r3_prep.py: CPU prep of round 3 under s_G (PLAN_pfaffian.md "Round 3 under s_G"). T_3 = norm_la(la(B_2) + 4.0 * f_J^G[B_2]) built exactly as
lswo2_prep.py built T_G from B_1 (norm_la and the n-weighted normalisation copied verbatim; no re-derivation of f_J: the g2 readout's fJ_fit.npy, f_J^G[B_2], E' = E_FN[B_2, s_G]).
Control: E_FN[T_3, s_G] (bond_split.build + delta0.efn_vec, the solver of r2g_readout.py) = readout/g2 scan['4.0'] to rel 1e-6.   usage: python3 r3_prep.py OUTDIR"""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] PROGRESS {m}', flush=True)
R2 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2'; RG = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r2g'
SG = f'{R2}/s_G.npy'; FIT = f'{RG}/run_s2/round_1/la_table.npy'; FJ = f'{RG}/readout/g2/fJ_fit.npy'; RDJ = f'{RG}/readout/g2/readout.json'
CSTAR = 4.0
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RJ = f'{OUT}/prep_result.json'; res = json.load(open(RJ)) if os.path.exists(RJ) else {}
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
res.update(host=os.uname().nodename, sG=SG, sG_sha256=sha(SG), fit=FIT, fit_sha256=sha(FIT), fJ=FJ, fJ_sha256=sha(FJ), E0=BS.E0, cstar=CSTAR)
assert res['fit_sha256'].startswith('b00dd2da')
RD = json.load(open(RDJ)); REF = RD['scan']['4.0']; res['ref_e_c4_g2'] = REF; assert RD['done'] and abs(REF - 5.2902e-6) < 1e-10
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
sG = np.load(SG).astype(np.float64); assert set(np.unique(sG)) <= {-1.0, 1.0}
lF = np.load(FIT).astype(np.float64); assert lF.shape == (D,) and np.isfinite(lF).all()
def norm_la(la): la = la - la.max(); return la - 0.5 * np.log((n * np.exp(2 * la)).sum())     # verbatim lswo2_prep.py
fJ = np.load(FJ).astype(np.float64); assert fJ.shape == (D,) and np.isfinite(fJ).all()
lT3 = norm_la(lF + CSTAR * fJ); np.save(f'{OUT}/target_T3_sG.npy', lT3); res['T3_sha256'] = sha(f'{OUT}/target_T3_sG.npy')
open(f'{OUT}/target_T3_sG.npy.sha256', 'w').write(res['T3_sha256'] + '  target_T3_sG.npy\n')
res['T3_norm_check'] = float((n * np.exp(2 * lT3)).sum()); dump(); log(f'T3 saved sha {res["T3_sha256"]} norm {res["T3_norm_check"]}')
t = time.time(); K, Vm, D0b = BS.build(n, sG); log(f'built K,Vm (s_G) {time.time() - t:.0f}s')
t = time.time(); r, _ = D0M.efn_vec(K, Vm, D0b, n, lT3); r['sec'] = time.time() - t
r['ref'] = REF; r['rel'] = abs(r['dE_site'] - REF) / REF; r['pass_1e-6_rel'] = bool(r['rel'] <= 1e-6); res['E_T3_sG'] = r; res['done'] = True; dump()
log(f'VERDICT PREP: E_FN[T3,sG] {r["dE_site"]:.10e} ref {REF:.10e} rel {r["rel"]:.2e} pass {r["pass_1e-6_rel"]} it {r["iters"]} res {r["res"]:.1e}')
