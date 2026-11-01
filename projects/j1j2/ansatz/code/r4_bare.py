"""r4_bare.py: round-4 readout (PLAN_pfaffian.md "Round 4").  K, Vm (sign s_G) built ONCE; then
  0. control: theta_bar's table (pfaffian_r3avg/tabs/tab_bar/round_1/la_table.npy, sha 3a3159fa) must give e = 1.2521612014025783e-05 (rel 1e-8), else exit 2 (no other solve);
  1. theta_4 (winner TABS/tab_t4): bare e, then the augmented c scan (c = 0,1,2,3,3.5,4,4.5,5,6,7 + parabola refinement; f_J^G as r3avg_bare.py / r3c_bare.py, E' = E_FN[theta_4]);
  2. bare e of the final iterate (winner FIT/round_1/la_table.npy) and of the checkpoints 600..1350 (missing checkpoint tables are skipped and recorded);
  3. theta_4 bookkeeping: the 7 theta shas, theta_4 recomputed locally = np.mean of the 7 iterates and compared with the cluster's theta_4.npy (recorded, no abort).
Winner = readout/winner.txt: 'pdb' -> pfaffian_r4/{pdb_fit,pdb_tabs}; 'a40' -> pfaffian_r4/{fit,tabs}.  Writes OUT/readout.json (resumable), readout.log.
Pre-registered: r_4 = (1.2522e-5 - e(theta_4)) / 0.7878e-5; r_4 >= 0.178 kills the decline; bare drop e(bar) - e(theta_4) <= 0.5e-6 kills progress.
usage: python3 r4_bare.py OUTDIR [--control-only]"""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; CONTROL_ONLY = '--control-only' in sys.argv
os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
Q = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4'; RA = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3avg'
S1 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy'
CTRL = (f'{RA}/tabs/tab_bar/round_1/la_table.npy', 1.2521612014025783e-05, '3a3159fa')   # pfaffian_r3avg/readout/readout.json theta_bar e (full digits)
E_BAR = 1.2521612014025783e-05; PRE_E = 1.2522e-5; PRE_G = 0.7878e-5; KILL_R = 0.178; KILL_DROP = 0.5e-6
winner = open(f'{OUT}/winner.txt').read().split()[0]; assert winner in ('pdb', 'a40'), winner
FD, TD = (f'{Q}/pdb_fit', f'{Q}/pdb_tabs') if winner == 'pdb' else (f'{Q}/fit', f'{Q}/tabs')
STEPS = (600, 750, 900, 1050, 1200, 1350)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RJ = f'{OUT}/readout.json'; res = json.load(open(RJ)) if os.path.exists(RJ) else {}
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
res.update(winner=winner, fit_dir=FD, tabs_dir=TD, host=os.uname().nodename, s1=S1, s1_sha256=sha(S1), E0=BS.E0,
           code_sha256={f: sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in ('r4_bare.py', 'bond_split.py', 'delta0.py')},
           fit_cx_r1fit_sha256=json.load(open(f'{FD}/spec_r1.json')).get('code_sha256', {}).get('cx_r1fit.py'))
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
assert sha(CTRL[0]).startswith(CTRL[2]), 'theta_bar table sha'
la, r = bare('control_theta_bar', CTRL[0]); del la; rel = (tabs['control_theta_bar']['e'] - CTRL[1]) / CTRL[1]
res['control'] = dict(path=CTRL[0], e=tabs['control_theta_bar']['e'], expect=CTRL[1], rel=rel, pass_=bool(abs(rel) <= 1e-8)); dump()
log(f'control: e {res["control"]["e"]:.9e} expect {CTRL[1]:.9e} rel {rel:.2e} pass {res["control"]["pass_"]}')
if not res['control']['pass_']: res['status'] = 'CONTROL FAIL'; dump(); log('CONTROL FAIL: STOP'); sys.exit(2)
if CONTROL_ONLY: res['status'] = 'CONTROL ONLY'; dump(); sys.exit(0)
# 3. theta bookkeeping (cheap, first)
names = [f'theta_step{k}.npy' for k in STEPS] + ['theta.npy']
th = [np.load(f'{FD}/round_1/{f}') for f in names]; m = np.mean(np.stack(th), axis=0); np.save(f'{OUT}/theta_4_local.npy', m)
cl = np.load(f'{TD}/theta_4.npy')
res['theta'] = dict(shas={f: sha(f'{FD}/round_1/{f}') for f in names}, theta_4_cluster_sha256=sha(f'{TD}/theta_4.npy'), theta_4_local_sha256=sha(f'{OUT}/theta_4_local.npy'),
                    max_abs_diff_cluster_vs_local=float(np.abs(cl - m).max()), cluster_shas_json=json.load(open(f'{TD}/theta_shas.json')))
dump(); log(f'theta_4: cluster sha {res["theta"]["theta_4_cluster_sha256"][:12]} local sha {res["theta"]["theta_4_local_sha256"][:12]} max|diff| {res["theta"]["max_abs_diff_cluster_vs_local"]:.2e}')
# 1. theta_4: bare + augmented scan
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
def fJ_of(lF, EP):                                                                           # verbatim r3avg_bare.py
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
def aug(name, lF, r):
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
la, r = bare('theta_4', f'{TD}/tab_t4/round_1/la_table.npy'); e4 = tabs['theta_4']['e']
res['verdict_numbers'] = dict(e_bar=E_BAR, e_theta4=e4, r_4=(PRE_E - e4) / PRE_G, r_4_exact_ebar=(E_BAR - e4) / PRE_G, bare_drop=E_BAR - e4,
                              kills_decline_r4_ge_0p178=bool((PRE_E - e4) / PRE_G >= KILL_R), kills_progress_drop_le_half_e6=bool(E_BAR - e4 <= KILL_DROP)); dump()
log(f'VERDICT NUMBERS: e(theta_4) {e4:.6e}  r_4 {res["verdict_numbers"]["r_4"]:.4f}  drop {res["verdict_numbers"]["bare_drop"] * 1e6:.4f}e-6  kill-decline {res["verdict_numbers"]["kills_decline_r4_ge_0p178"]}  kill-progress {res["verdict_numbers"]["kills_progress_drop_le_half_e6"]}')
aug('theta_4', la, r); del la
res['aug_done'] = True; dump()
# 2. final iterate and checkpoints (bare)
bare('final_1500', f'{FD}/round_1/la_table.npy')
for k in STEPS:
    p = f'{TD}/tab_{k}/round_1/la_table.npy'
    if os.path.exists(p): bare(f'ckpt_{k}', p)
    else: res.setdefault('skipped', []).append(f'ckpt_{k}'); log(f'ckpt_{k}: table missing, skipped')
rows = [(k, tabs[f'ckpt_{k}']['e']) for k in STEPS if f'ckpt_{k}' in tabs] + [(1500, tabs['final_1500']['e'])]
log('ITERATES (step: e): ' + '  '.join(f'{k}: {e:.6e}' for k, e in rows) + f'   | theta_4 {e4:.6e}  theta_bar {E_BAR:.6e}')
res['status'] = 'DONE'; res['done'] = True; dump(); log('DONE')
