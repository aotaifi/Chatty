"""v_bare.py: gate V readout (PLAN_pfaffian.md "B design" / "V (narrowed)") = r5_bare.py's control and bare-e part for ONE arm, the pool-driven fit.  K, Vm (sign s_G) built once;
  0. control: theta_4's table (pfaffian_r4/pdb_tabs/tab_t4/round_1/la_table.npy) must give e = 1.2184484428720862e-05 (rel 1e-8), else exit 2;
  1. theta bookkeeping: theta_V = mean of the 7 iterates (fit/round_1/theta_step{600..1350}.npy + theta.npy) recomputed locally vs the cluster's tab/theta_5.npy and
     pool_fit's own theta_bar.npy; then bare e of tab/tab_t5/round_1/la_table.npy and r_V = (1.2184484e-5 - e) / 0.7817e-5 (round 5's r_5 definition).
  2. verdict (pre-registered, PLAN "B design" (5) V): PASS r_V >= 0.35; KILL "the loop survives sampling" r_V < 0.25; between: neither.
     Next to it: round 5's exact-table pp2 s1 (same seed, warm start, target, steps): r_5 = 0.493 (pfaffian_r5/readout/readout.json, read, not recomputed).
No augmented c scan here (the f_J fix changes f_J; a scan belongs to the fixed-f_J readout).  Writes OUT/readout.json, readout.log.   usage: python3 v_bare.py OUTDIR"""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
Q = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_v'; R4 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4'; R5 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5'
S1 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy'
CTRL = (f'{R4}/pdb_tabs/tab_t4/round_1/la_table.npy', 1.2184484428720862e-05)
CTRL_SHA = json.load(open(f'{R4}/readout/readout.json'))['tables']['theta_4']['sha256']
PRE_E = 1.2184484e-5; PRE_G = 0.7817e-5; PASS_R = 0.35; KILL_R = 0.25
WIN = open(f'{Q}/readout/winner.txt').read().split()[0]; FD, TD = {'pdb': (f'{Q}/fit', f'{Q}/tab'), 'pdb2': (f'{Q}/pdb2/fit', f'{Q}/pdb2/tab'), 'th': (f'{Q}/fit_th', f'{Q}/tab_th')}[WIN]; STEPS = (600, 750, 900, 1050, 1200, 1350)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RJ = f'{OUT}/readout.json'; res = json.load(open(RJ)) if os.path.exists(RJ) else {}
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
res.update(winner=WIN, host=os.uname().nodename, s1=S1, s1_sha256=sha(S1), pre_e=PRE_E, pre_g=PRE_G, pass_r=PASS_R, kill_r=KILL_R,
           code_sha256={f: sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in ('v_bare.py', 'bond_split.py', 'delta0.py')},
           fit_spec=json.load(open(f'{FD}/spec_pool_fit.json')))
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
s1 = np.load(S1).astype(np.float64); assert set(np.unique(s1)) <= {-1.0, 1.0}
K, Vm, D0 = BS.build(n, s1); log('built K,Vm (s_G) once')
def load(p):
    la = np.load(p).astype(np.float64); assert la.shape == (D,) and np.isfinite(la).all(), p; return la
tabs = res.setdefault('tables', {})
def bare(name, path):
    if name in tabs and 'E_FN' in tabs[name]: return tabs[name]['E_FN']
    la = load(path); t = time.time(); r, _ = D0M.efn_vec(K, Vm, D0, n, la)
    tabs.setdefault(name, {}).update(path=path, sha256=sha(path), e=r['dE_site'], E_FN=r); dump()
    log(f'{name}: e = {r["dE_site"]:.9e}  it {r["iters"]} res {r["res"]:.1e}  {time.time() - t:.0f}s'); return r
assert sha(CTRL[0]) == CTRL_SHA, 'theta_4 table sha'
bare('control_theta_4', CTRL[0]); rel = (tabs['control_theta_4']['e'] - CTRL[1]) / CTRL[1]
res['control'] = dict(path=CTRL[0], e=tabs['control_theta_4']['e'], expect=CTRL[1], rel=rel, pass_=bool(abs(rel) <= 1e-8)); dump()
log(f'control: e {res["control"]["e"]:.9e} expect {CTRL[1]:.9e} rel {rel:.2e} pass {res["control"]["pass_"]}')
if not res['control']['pass_']: res['status'] = 'CONTROL FAIL'; dump(); log('CONTROL FAIL: STOP'); sys.exit(2)
names = [f'theta_step{k}.npy' for k in STEPS] + ['theta.npy']
th = [np.load(f'{FD}/round_1/{f}') for f in names]; m = np.mean(np.stack(th), axis=0); np.save(f'{OUT}/theta_V_local.npy', m)
cl = np.load(f'{TD}/theta_5.npy'); tb = np.load(f'{FD}/round_1/theta_bar.npy')
res['theta'] = dict(npar=int(m.size), shas={f: sha(f'{FD}/round_1/{f}') for f in names}, cluster_theta_5_sha256=sha(f'{TD}/theta_5.npy'),
                    max_abs_diff_cluster_vs_local=float(np.abs(cl - m).max()), max_abs_diff_pool_fit_theta_bar_vs_local=float(np.abs(tb - m).max()),
                    lambda_theta_V=[float(m[-2]), float(m[-1])]); dump()
log(f'theta_V: npar {m.size} max|cluster - local| {res["theta"]["max_abs_diff_cluster_vs_local"]:.2e} max|theta_bar - local| {res["theta"]["max_abs_diff_pool_fit_theta_bar_vs_local"]:.2e}')
bare('V', f'{TD}/tab_t5/round_1/la_table.npy'); e = tabs['V']['e']; r = (PRE_E - e) / PRE_G
r5 = json.load(open(f'{R5}/readout/readout.json'))['verdict_numbers']['pp2_s1']
verdict = 'PASS' if r >= PASS_R else ('KILL' if r < KILL_R else 'NEITHER (0.25 <= r_V < 0.35)')
res['verdict_numbers'] = dict(e=e, r_V=r, bare_drop=CTRL[1] - e, exact_table_pp2_s1=dict(e=r5['e'], r_5=r5['r_5']), r_V_over_r5=r / r5['r_5'], verdict=verdict)
res['status'] = 'BARE DONE'; dump()
log(f'VERDICT V: e(theta_V) {e:.6e}  r_V {r:.4f}  (exact-table pp2 s1 r_5 {r5["r_5"]:.4f}; ratio {r / r5["r_5"]:.3f})  -> {verdict}')
# 3. augmented e of theta_V with the FIXED f_J (fj_fix mode b1, E' = E_FN[theta_V]), c in 3, 3.5, 4, 4.5, 5 (the FIX re-scan grid); relative weight check vs the bare guide
import fj_fix as FX
lF = load(f'{TD}/tab_t5/round_1/la_table.npy'); raw, dmin = FX.raw_fJ(lF, tabs['V']['E_FN']['E'], s1); assert dmin > 0
def sh(la): w = n * np.exp(2 * (la - la.max())); return float(w.max() / w.sum())
aug = res.setdefault('aug_b1', {}); sb = sh(lF); aug['base_max_share'] = sb; aug['n_nonfinite'] = int((~np.isfinite(raw)).sum())
for c in (3.0, 3.5, 4.0, 4.5, 5.0):
    if str(c) in aug.get('scan', {}): continue
    add, info = FX.apply_mode(raw, n, 'b1', c); la = lF + add; sa = sh(la); r_, _ = D0M.efn_vec(K, Vm, D0, n, la)
    aug.setdefault('scan', {})[str(c)] = dict(e=r_['dE_site'], iters=r_['iters'], res=r_['res'], f0=info['f0'], max_share=sa, rel_weight_ok=bool(sa <= 10 * sb)); dump()
    log(f'aug b1 c={c}: e {r_["dE_site"]:.6e} max share {sa:.3e} (base {sb:.3e})')
cb = min(aug['scan'], key=lambda k: aug['scan'][k]['e']); aug['cstar_grid'] = float(cb); aug['e_aug_min'] = aug['scan'][cb]['e']
res['verdict_numbers']['e_aug_b1'] = aug['e_aug_min']; res['verdict_numbers']['cstar_b1'] = float(cb); res['aug_done'] = True; res['status'] = 'DONE'; res['done'] = True; dump()
log(f'aug b1: c* {cb} e_aug {aug["e_aug_min"]:.6e}')
