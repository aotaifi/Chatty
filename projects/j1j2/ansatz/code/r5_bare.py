"""r5_bare.py: round-5 readout (PLAN_pfaffian.md "Two-PP class (d)") = r4_bare.py (sha 76c40780) with the arm loop of round 5.  K, Vm (sign s_G) built ONCE; then
  0. control: theta_4's table (pfaffian_r4/pdb_tabs/tab_t4/round_1/la_table.npy, theta_4 sha 31142501) must give e = 1.2184484428720862e-05 (rel 1e-8), else exit 2 (no other solve);
  1. for each arm in readout/arms.txt (written by r5_chain.sh; tokens ARM:pdb | ARM:th = the race winner's cluster; subset of pp2_s1 pp2_s2 ctl_s1 ctl_s2): theta bookkeeping (7 shas; theta_5 recomputed locally = np.mean of the 7
     iterates vs the cluster's theta_5.npy, recorded, no abort), then bare e of theta_5 (ARM/tab/tab_t5/round_1/la_table.npy) and r_5 = (1.2184484e-5 - e) / 0.7817e-5;
  2. for each arm: the augmented c scan (c = 0,1,2,3,3.5,4,4.5,5,6,7 + parabola refinement; f_J^G and aug() verbatim from r4_bare.py, E' = E_FN[theta_5 of that arm]).
Writes OUT/readout.json (resumable), readout.log.   usage: python3 r5_bare.py OUTDIR [--control-only]
Pre-registered (PLAN "Two-PP class (d)"): KILL (d): both pp2 seeds r_5 <= 0.05 AND e_aug >= 4.2e-6.  NOT ATTRIBUTABLE: r_5(pp2_sK) - r_5(ctl_sK) < 0.03 for K = 1 or 2.
Predictions: pp2 r_5 0.06-0.16 (bare 1.09-1.17e-5), aug 3.4-4.1e-6; control r_5 0.01-0.04 (bare 1.19-1.21e-5), aug 4.20-4.35e-6.  (r5_gate.py applies the rules.)"""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; CONTROL_ONLY = '--control-only' in sys.argv
os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
Q = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5'; R4 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4'
S1 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy'
CTRL = (f'{R4}/pdb_tabs/tab_t4/round_1/la_table.npy', 1.2184484428720862e-05)   # pfaffian_r4/readout/readout.json tables.theta_4.e (full digits)
CTRL_SHA = json.load(open(f'{R4}/readout/readout.json'))['tables']['theta_4']['sha256']
PRE_E = 1.2184484e-5; PRE_G = 0.7817e-5
ARMS_ALL = ('pp2_s1', 'pp2_s2', 'ctl_s1', 'ctl_s2')
tok = open(f'{OUT}/arms.txt').read().split() if not CONTROL_ONLY else []      # 'ARM:pdb' (Paderborn: ARM/fit, ARM/tab) or 'ARM:th' (theorie race copy: ARM/fit_th, ARM/tab_th)
arms = [t.split(':')[0] for t in tok]; CL = {t.split(':')[0]: (t.split(':') + ['pdb'])[1] for t in tok}
assert all(a in ARMS_ALL for a in arms) and all(c in ('pdb', 'th') for c in CL.values()), tok
DIRS = {a: ((f'{Q}/{a}/fit', f'{Q}/{a}/tab') if CL[a] == 'pdb' else (f'{Q}/{a}/fit_th', f'{Q}/{a}/tab_th')) for a in arms}
STEPS = (600, 750, 900, 1050, 1200, 1350)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RJ = f'{OUT}/readout.json'; res = json.load(open(RJ)) if os.path.exists(RJ) else {}
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
res.update(arms=arms, cluster=CL, dirs=DIRS, host=os.uname().nodename, s1=S1, s1_sha256=sha(S1), E0=BS.E0, pre_e=PRE_E, pre_g=PRE_G,
           code_sha256={f: sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in ('r5_bare.py', 'bond_split.py', 'delta0.py')},
           fit_code_sha256={a: json.load(open(f'{DIRS[a][0]}/spec_r1.json')).get('code_sha256') for a in arms})
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
s1 = np.load(S1).astype(np.float64); assert set(np.unique(s1)) <= {-1.0, 1.0}
K, Vm, D0 = BS.build(n, s1); log('built K,Vm (s_G) once')
def load(p):
    la = np.load(p).astype(np.float64); assert la.shape == (D,) and np.isfinite(la).all(), p; return la
tabs = res.setdefault('tables', {})
def bare(name, path):
    if name in tabs and 'E_FN' in tabs[name]: return load(path), tabs[name]['E_FN']
    la = load(path); t = time.time(); r, _ = D0M.efn_vec(K, Vm, D0, n, la)
    tabs.setdefault(name, {}).update(path=path, sha256=sha(path), e=r['dE_site'], E_FN=r); dump()
    log(f'{name}: e = {r["dE_site"]:.9e}  it {r["iters"]} res {r["res"]:.1e}  {time.time() - t:.0f}s'); return la, r
# 0. control
assert sha(CTRL[0]) == CTRL_SHA, 'theta_4 table sha'
la, r = bare('control_theta_4', CTRL[0]); del la; rel = (tabs['control_theta_4']['e'] - CTRL[1]) / CTRL[1]
res['control'] = dict(path=CTRL[0], e=tabs['control_theta_4']['e'], expect=CTRL[1], rel=rel, pass_=bool(abs(rel) <= 1e-8)); dump()
log(f'control: e {res["control"]["e"]:.9e} expect {CTRL[1]:.9e} rel {rel:.2e} pass {res["control"]["pass_"]}')
if not res['control']['pass_']: res['status'] = 'CONTROL FAIL'; dump(); log('CONTROL FAIL: STOP'); sys.exit(2)
if CONTROL_ONLY: res['status'] = 'CONTROL ONLY'; dump(); sys.exit(0)
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
def fJ_of(lF, EP):                                                                           # verbatim r4_bare.py (= r3avg_bare.py)
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
def aug(name, lF, r):                                                                        # verbatim r4_bare.py
    EP = r['E']; fJ, info = fJ_of(lF, EP); tabs[name]['fJ'] = info; log(f'{name} fJ {info}')
    cache = {float(k): v for k, v in tabs[name].get('scan', {}).items()}; cache[0.0] = tabs[name]['e']; tabs[name]['scan'] = {str(k): v for k, v in cache.items()}
    def solve(c):
        c = round(float(c), 6)
        if c in cache: return cache[c]
        rr, _ = D0M.efn_vec(K, Vm, D0, n, lF + c * fJ); cache[c] = rr['dE_site']; tabs[name]['scan'][str(c)] = rr['dE_site']; dump()
        log(f'{name} c={c:.3f} e {rr["dE_site"]:.6e} it {rr["iters"]} res {rr["res"]:.1e}'); return rr['dE_site']
    for c in (1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0): solve(c)
    status = None
    for it in range(8):
        b = min(cache, key=cache.get)
        if b >= 7.0: status = 'MIN AT UPPER EDGE'; break
        if b <= 0.0: status = 'MIN AT c=0'; break
        cs = sorted(cache); i = cs.index(b); trio = cs[i - 1:i + 2]
        p = np.polyfit(trio, [cache[c] for c in trio], 2)
        if p[0] <= 0: status = 'NOT CONVEX'; break
        cv = round(float(np.clip(-p[1] / (2 * p[0]), 0, 7)) / 0.1) * 0.1; log(f'{name} iter {it}: trio {trio} vertex-> {cv:.2f}')
        if round(cv, 6) in cache: break
        solve(cv)
    cst = min(cache, key=cache.get); tabs[name].update(cstar=cst, e_aug_min_site=cache[cst], scan_status=status or 'DONE'); dump()
    log(f'{name}: c* {cst} e_aug {cache[cst]:.6e} ({tabs[name]["scan_status"]})')
# 1. per arm: theta bookkeeping + bare e + r_5
names = [f'theta_step{k}.npy' for k in STEPS] + ['theta.npy']; vn = res.setdefault('verdict_numbers', {}); th_rec = res.setdefault('theta', {})
for a in arms:
    FD, TD = DIRS[a]
    th = [np.load(f'{FD}/round_1/{f}') for f in names]; m = np.mean(np.stack(th), axis=0); np.save(f'{OUT}/theta_5_{a}_local.npy', m); cl = np.load(f'{TD}/theta_5.npy')
    th_rec[a] = dict(npar=int(m.size), shas={f: sha(f'{FD}/round_1/{f}') for f in names}, theta_5_cluster_sha256=sha(f'{TD}/theta_5.npy'), theta_5_local_sha256=sha(f'{OUT}/theta_5_{a}_local.npy'),
                     max_abs_diff_cluster_vs_local=float(np.abs(cl - m).max()), cluster_shas_json=json.load(open(f'{TD}/theta_shas.json')))
    if m.size == 5334: th_rec[a]['lambda_theta5'] = [float(m[-2]), float(m[-1])]; th_rec[a]['lambda_final'] = [float(th[-1][-2]), float(th[-1][-1])]
    dump(); log(f'{a} theta_5: npar {m.size} cluster sha {th_rec[a]["theta_5_cluster_sha256"][:12]} local {th_rec[a]["theta_5_local_sha256"][:12]} max|diff| {th_rec[a]["max_abs_diff_cluster_vs_local"]:.2e}'
                + (f' lambda(theta_5) {th_rec[a]["lambda_theta5"]}' if 'lambda_theta5' in th_rec[a] else ''))
    la, r = bare(a, f'{TD}/tab_t5/round_1/la_table.npy'); del la; e = tabs[a]['e']
    vn[a] = dict(e=e, r_5=(PRE_E - e) / PRE_G, r_5_exact_e4=(CTRL[1] - e) / PRE_G, bare_drop=CTRL[1] - e); dump()
    log(f'VERDICT NUMBERS {a}: e(theta_5) {e:.6e}  r_5 {vn[a]["r_5"]:.4f}  drop {vn[a]["bare_drop"] * 1e6:.4f}e-6')
res['bare_done'] = True; dump()
# 2. per arm: augmented scan (pp2 arms first, as listed)
for a in arms:
    if tabs[a].get('scan_status'): continue
    la = load(f'{DIRS[a][1]}/tab_t5/round_1/la_table.npy'); aug(a, la, tabs[a]['E_FN']); del la
    vn[a]['e_aug'] = tabs[a]['e_aug_min_site']; vn[a]['cstar'] = tabs[a]['cstar']; dump()
res['status'] = 'DONE'; res['done'] = True; dump(); log('DONE')
