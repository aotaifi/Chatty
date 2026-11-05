"""m5p2x_read.py (= m5p2_read.py with the final block replaced for EXTENSION SEGMENT 1: e_end from the TAIL AVERAGE of theta 1200..2100 (7 saved iterates, tab_avg), warm-theta check against round-1 theta(1500), fit steps 600; env M5P2X_Q only for tests; first line of the m5p2_read docstring follows: the M5-prime rerun, PLAN_pfaffian.md 'M5' rerun'): M5' readouts (PLAN_pfaffian.md "M5'", lead GO 2026-11-04).  usage: python3 m5p_read.py OUTDIR MODE [PHASE]
  MODE rel PHASE : relax phase (rel1|rel2|rel3): control first (theta_4's table e = 1.2184484428720862e-05, rel 1e-8, else exit 2), then e_rel of PHASE's table
                   Q/PHASE/{tab|tab_th}/tab_rel/round_1/la_table.npy (cluster from OUT/arms.txt token 'PHASE:pdb|th'); pass_ = e_rel <= 1.25e-5.
  MODE final     : controls first, then e_rel (recomputed from the accepted relax table named in Q/relax_accepted.json, sha-checked), e_end of the round table
                   (OUT/arms.txt token 'round:pdb|th'), r_5(M5') = (e_rel - e_end)/(e_rel - 4.367093e-6), and the augmented c scan with the FIXED f_J
                   (fj_fix.py mode b1, E' = E_FN[theta_end], c grid 3..5 step 0.5, extended by 0.5 while the minimum sits at the edge), relative weight check as v_bare.py.
  MODE selftest  : deliberately checkable: tab_rel := theta_4's table, tab_end := pfaffian_r5/target_T5_sG.npy (E_FN 4.367093186728886e-6): r must be 1 (rel 1e-6), aug scan c = 4 only.
Writes OUT/readout.json (resumable), readout.log."""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT, MODE = sys.argv[1], sys.argv[2]; PHASE = sys.argv[3] if len(sys.argv) > 3 else None
os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
Q = os.environ.get('M5P2X_Q', '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_m5p2'); E = f'{Q}/ext1'; R4 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4'; R5 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5'
S1 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy'
CTRL = (f'{R4}/pdb_tabs/tab_t4/round_1/la_table.npy', 1.2184484428720862e-05)
CTRL_SHA = json.load(open(f'{R4}/readout/readout.json'))['tables']['theta_4']['sha256']
E_REL_MAX = 1.25e-5; E_FLOOR = 4.367093e-6; E_FLOOR_FULL = 4.367093186728886e-6; GATE_R = 0.4929; GATE_DR = 0.15; GATE_AUG = 3.2584e-6; GATE_DAUG = 0.3e-6
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RJ = f'{OUT}/readout.json'; res = json.load(open(RJ)) if os.path.exists(RJ) else {}
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
here = os.path.dirname(os.path.abspath(__file__))
res.update(mode=MODE, phase=PHASE, host=os.uname().nodename, s1=S1, s1_sha256=sha(S1), E0=BS.E0,
           code_sha256={f: sha(os.path.join(here, f)) for f in ('m5p2x_read.py', 'bond_split.py', 'delta0.py', 'fj_fix.py')},
           registered=dict(e_rel_max=E_REL_MAX, e_floor=E_FLOOR, gate_r5pp=GATE_R, gate_dr=GATE_DR, gate_aug=GATE_AUG, gate_daug=GATE_DAUG))
tok = dict(t.split(':') for t in open(f'{OUT}/arms.txt').read().split()) if MODE != 'selftest' else {}
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
s1 = np.load(S1).astype(np.float64); assert set(np.unique(s1)) <= {-1.0, 1.0}
K, Vm, D0 = BS.build(n, s1); log('built K,Vm (s_G) once')
def load(p):
    la = np.load(p).astype(np.float64); assert la.shape == (D,) and np.isfinite(la).all(), p; return la
tabs = res.setdefault('tables', {})
def bare(name, path, expect_sha=None):
    if expect_sha: assert sha(path) == expect_sha, f'sha mismatch {path}'
    if name in tabs and 'E_FN' in tabs[name]: return load(path), tabs[name]['E_FN']
    la = load(path); t = time.time(); r, _ = D0M.efn_vec(K, Vm, D0, n, la)
    tabs.setdefault(name, {}).update(path=path, sha256=sha(path), e=r['dE_site'], E_FN=r); dump()
    log(f'{name}: e = {r["dE_site"]:.9e}  it {r["iters"]} res {r["res"]:.1e}  {time.time() - t:.0f}s'); return la, r
# 0. control FIRST
assert sha(CTRL[0]) == CTRL_SHA, 'theta_4 table sha'
la, r = bare('control_theta_4', CTRL[0]); del la; rel = (tabs['control_theta_4']['e'] - CTRL[1]) / CTRL[1]
res['control'] = dict(path=CTRL[0], e=tabs['control_theta_4']['e'], expect=CTRL[1], rel=rel, pass_=bool(abs(rel) <= 1e-8)); dump()
log(f'control: e {res["control"]["e"]:.9e} expect {CTRL[1]:.9e} rel {rel:.2e} pass {res["control"]["pass_"]}')
if not res['control']['pass_']: res['status'] = 'CONTROL FAIL'; dump(); log('CONTROL FAIL: STOP'); sys.exit(2)
def tabdir(phase):
    c = tok[phase]; assert c in ('pdb', 'th'), tok; return (f'{Q}/{phase}/tab' if c == 'pdb' else f'{Q}/{phase}/tab_th'), (f'{Q}/{phase}/fit' if c == 'pdb' else f'{Q}/{phase}/fit_th')
if MODE == 'rel':
    assert PHASE in ('rel1', 'rel2', 'rel3'); TD, FD = tabdir(PHASE); la, r = bare('rel', f'{TD}/tab_rel/round_1/la_table.npy'); del la
    th_c = f'{TD}/theta_rel.npy'; th_l = f'{FD}/round_1/theta.npy'; same = sha(th_c) == sha(th_l)
    ft = json.load(open(f'{FD}/cx_r1.json'))['fit']
    res['relax'] = dict(phase=PHASE, cluster=tok[PHASE], e_rel=tabs['rel']['e'], pass_=bool(tabs['rel']['e'] <= E_REL_MAX), theta_rel=th_c, theta_rel_sha256=sha(th_c), theta_equals_fit_final=bool(same),
                        fit_steps=ft.get('steps'), L_ho_step0=ft.get('L_ho_step0'), L_ho_last=ft['L_ho_curve'][-1], table=tabs['rel']['path'], table_sha256=tabs['rel']['sha256'],
                        control_e_theta_4=CTRL[1], e_rel_over_theta_4=tabs['rel']['e'] / CTRL[1])
    res['status'] = 'DONE'; res['done'] = True; dump()
    log(f'VERDICT RELAX {PHASE}: e_rel {tabs["rel"]["e"]:.6e} (limit {E_REL_MAX:.3e}) -> {"PASS" if res["relax"]["pass_"] else "FAIL"}; fit steps {ft.get("steps")}; theta sha equals fit final: {same}'); sys.exit(0)
# ---- final / selftest
if MODE == 'selftest':
    TABR = (CTRL[0], 'selftest'); TABE = (f'{R5}/target_T5_sG.npy', None); GRID = (4.0,)
    la_rel, rr = bare('e_rel', TABR[0]); la_end, re_ = bare('e_end', TABE[0])
else:
    acc = json.load(open(f'{Q}/relax_accepted.json')); res['relax_accepted'] = acc
    la, rr = bare('e_rel', acc['table'], acc['table_sha256']); del la
    assert abs(tabs['e_rel']['e'] - acc['e_rel']) / acc['e_rel'] < 1e-8, ('recomputed e_rel differs from the accepted relax readout', tabs['e_rel']['e'], acc['e_rel'])
    # ---- EXTENSION SEGMENT 1 (PLAN_pfaffian.md "Extension segment 1"): the tail average is over 7 saved iterates, global steps 1200..2100 (spacing 150): round-1 theta_step1200, theta_step1350, theta.npy (= step 1500, the warm start of ext1),
    # ext1 theta_step150/300/450 (= 1650/1800/1950) and ext1 theta.npy (= step 2100, == theta_step600).  The 1500 itself is INCLUDED (the 1500-step average also included its own final iterate).
    TD, FD = f'{E}/tab_th', f'{E}/fit_th'; RF = f'{Q}/round/fit_th/round_1'
    SRC = [(1200, f'{RF}/theta_step1200.npy'), (1350, f'{RF}/theta_step1350.npy'), (1500, f'{RF}/theta.npy'), (1650, f'{FD}/round_1/theta_step150.npy'), (1800, f'{FD}/round_1/theta_step300.npy'),
           (1950, f'{FD}/round_1/theta_step450.npy'), (2100, f'{FD}/round_1/theta.npy')]
    th = [np.load(p) for _, p in SRC]; m = np.mean(np.stack(th), axis=0); cl = np.load(f'{TD}/theta_avg.npy'); np.save(f'{OUT}/theta_avg_local.npy', m)
    ftj = json.load(open(f'{FD}/cx_r1.json')); fsteps = ftj['fit']['steps']; assert fsteps == 600, ('ext1 fit steps', fsteps)
    res['theta'] = dict(npar=int(m.size), cluster_sha256=sha(f'{TD}/theta_avg.npy'), max_abs_diff_cluster_vs_local=float(np.abs(cl - m).max()), fit_steps=fsteps, sources=[dict(global_step=g, path=p, sha256=sha(p)) for g, p in SRC],
                        warm_theta_sha256_fit=json.load(open(f'{FD}/round_1/cx_t1s.json')).get('warm_theta_sha256'), warm_theta_sha256_accepted=sha(f'{RF}/theta.npy'), round_theta_1500_sha256=sha(f'{RF}/theta.npy'))
    assert res['theta']['warm_theta_sha256_fit'] == res['theta']['round_theta_1500_sha256'], 'ext1 warm theta is not the round-1 final iterate theta(1500)'
    assert res['theta']['max_abs_diff_cluster_vs_local'] <= 1e-14, 'cluster tail average differs from the local recomputation'; dump()
    la_end, re_ = bare('e_end', f'{TD}/tab_avg/round_1/la_table.npy'); GRID = (3.0, 3.5, 4.0, 4.5, 5.0)
e_rel = tabs['e_rel']['e']; e_end = tabs['e_end']['e']
r5 = (e_rel - e_end) / (e_rel - E_FLOOR); r5f = (e_rel - e_end) / (e_rel - E_FLOOR_FULL)
vn = res.setdefault('verdict_numbers', {}); vn.update(e_rel=e_rel, e_end=e_end, r_5_M5p=r5, r_5_M5p_fulldigits=r5f, bare_drop=e_rel - e_end)
if MODE == 'selftest':
    vn['selftest_r_expected'] = 1.0; vn['selftest_pass'] = bool(abs(r5f - 1.0) < 1e-6 and abs(e_end - E_FLOOR_FULL) / E_FLOOR_FULL < 1e-6 and abs(e_rel - CTRL[1]) / CTRL[1] < 1e-8)
dump(); log(f'VERDICT NUMBERS: e_rel {e_rel:.6e}  e_end {e_end:.6e}  r_5(M5p) {r5:.4f}')
import fj_fix as FX
raw, dmin = FX.raw_fJ(la_end, tabs['e_end']['E_FN']['E'], s1); assert dmin > 0
def sh(la): w = n * np.exp(2 * (la - la.max())); return float(w.max() / w.sum())
aug = res.setdefault('aug_b1', {}); sb = sh(la_end); aug['base_max_share'] = sb; aug['n_nonfinite'] = int((~np.isfinite(raw)).sum()); sc = aug.setdefault('scan', {})
def solve(c):
    if str(c) in sc: return
    add, info = FX.apply_mode(raw, n, 'b1', c); la = la_end + add; sa = sh(la); r_, _ = D0M.efn_vec(K, Vm, D0, n, la)
    sc[str(c)] = dict(e=r_['dE_site'], iters=r_['iters'], res=r_['res'], f0=info['f0'], max_share=sa, rel_weight_ok=bool(sa <= 10 * sb)); dump()
    log(f'aug b1 c={c}: e {r_["dE_site"]:.6e} max share {sa:.3e} (base {sb:.3e})')
for c in GRID: solve(c)
if MODE == 'final':
    for _ in range(8):
        cs = sorted(float(k) for k in sc); b = min(cs, key=lambda c: sc[str(c)]['e'])
        if b == cs[-1] and b + 0.5 <= 7.0: solve(b + 0.5)
        elif b == cs[0] and b - 0.5 >= 0.5: solve(b - 0.5)
        else: break
cb = min(sc, key=lambda k: sc[k]['e']); aug.update(cstar_grid=float(cb), e_aug_min=sc[cb]['e'], rel_weight_ok_all=all(v['rel_weight_ok'] for v in sc.values()))
vn.update(e_aug_b1=aug['e_aug_min'], cstar_b1=float(cb)); res['status'] = 'DONE'; res['done'] = True; dump()
log(f'aug b1: c* {cb} e_aug_min {aug["e_aug_min"]:.6e} rel-weight ok for all c: {aug["rel_weight_ok_all"]}; DONE')
