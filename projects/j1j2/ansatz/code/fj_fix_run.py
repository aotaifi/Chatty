"""fj_fix_run.py: extension of fj_audit.py (PLAN "FIX").  Per guide: modes a1,a2 (guides with non-finite rows only; with none they equal the unfixed guide), b1,b2, each at c in 3,3.5,4,4.5,5;
unfixed ('old') on the same grid for the round-5 guides (two-PP gain before/after).  Unit check (first s_G guide): b1/b2 with f0=inf equal the old vector exactly; f0=1e-3 changes it and E.
Adds the round-5 controls ctl_s1/ctl_s2.   usage: python3 fj_fix_run.py OUTDIR"""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS, delta0 as D0M, fj_fix as FX
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
R = '/scratch/a/A.Otaifi/ansatz_runs'; S_1 = f'{R}/pfaffian_z/s1.npy'; S_G = f'{R}/pfaffian_lswo2/s_G.npy'; CG = (3.0, 3.5, 4.0, 4.5, 5.0)
GUIDES = [('lswo_r1', S_1, f'{R}/pfaffian_lswo/readout/readout.json', None, f'{R}/pfaffian_lswo/run/round_1/la_table.npy'),
 ('r2g_g1', S_G, f'{R}/pfaffian_r2g/readout/g1/readout.json', None, None), ('r2g_g2', S_G, f'{R}/pfaffian_r2g/readout/g2/readout.json', None, None),
 ('r3_1200', S_G, f'{R}/pfaffian_r3avg/readout/readout.json', 'r3_1200', None), ('r3_theta_bar', S_G, f'{R}/pfaffian_r3avg/readout/readout.json', 'theta_bar', None),
 ('r4_theta_4', S_G, f'{R}/pfaffian_r4/readout/readout.json', 'theta_4', None),
 ('r5_pp2_s1', S_G, f'{R}/pfaffian_r5/readout/readout.json', 'pp2_s1', None), ('r5_pp2_s2', S_G, f'{R}/pfaffian_r5/readout/readout.json', 'pp2_s2', None),
 ('r5_ctl_s1', S_G, f'{R}/pfaffian_r5/readout/readout.json', 'ctl_s1', None), ('r5_ctl_s2', S_G, f'{R}/pfaffian_r5/readout/readout.json', 'ctl_s2', None)]
R5 = ('r5_pp2_s1', 'r5_pp2_s2', 'r5_ctl_s1', 'r5_ctl_s2')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RJ = f'{OUT}/fj_fix.json'; res = json.load(open(RJ)) if os.path.exists(RJ) else {}
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); assert n.sum() == 9075135300
res['code_sha256'] = {f: sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in ('fj_fix.py', 'fj_fix_run.py', 'bond_split.py', 'delta0.py')}
unit_done = bool(res.get('unit'))
for sname, sfile in (('s1', S_1), ('sG', S_G)):
    s = np.load(sfile).astype(np.float64); K, Vm, D0 = BS.build(n, s); log(f'built K,Vm ({sname})')
    for name, sf, rj, key, tpath in [g for g in GUIDES if g[1] == sfile]:
        G = res.setdefault('guides', {}).setdefault(name, {})
        if G.get('done'): continue
        d = json.load(open(rj))
        if key is None: cst = d['cstar']; eaug = d['e_aug_min_site']; EP = d['E_FN_fit']['E']; tp = tpath or d['fit']; tsha = d['fit_sha256']
        else: rec = d['tables'][key]; cst = rec['cstar']; eaug = rec['e_aug_min_site']; EP = rec['E_FN']['E']; tp = rec['path']; tsha = rec['sha256']
        assert sha(tp) == tsha; lF = np.load(tp).astype(np.float64); G.update(sign=sname, table=tp, table_sha256=tsha, cstar=cst, e_aug_reported=eaug, EP=EP)
        raw, dmin = FX.raw_fJ(lF, EP, s); assert dmin > 0; nb = int((~np.isfinite(raw)).sum()); G['n_nonfinite'] = nb
        def solve(mode, c, f0=None):
            add, info = FX.apply_mode(raw, n, mode, c, f0); la = lF + add
            try: wc = FX.weight_check(la, n); wf = None
            except ValueError as e: wc = None; wf = str(e)
            r, _ = D0M.efn_vec(K, Vm, D0, n, la); return dict(e=r['dE_site'], iters=r['iters'], res=r['res'], weight_check=wc, weight_fail=wf, **{k: v for k, v in info.items() if k not in ('mode', 'c')})
        if not unit_done and name == 'r4_theta_4':
            a_old, _ = FX.apply_mode(raw, n, 'old', 4.0); u = {}
            for m in ('b1', 'b2'):
                a_inf, _ = FX.apply_mode(raw, n, m, 4.0, np.inf); u[m + '_f0inf_equal_old'] = bool((a_inf == a_old).all())
                a_t, _ = FX.apply_mode(raw, n, m, 4.0, 1e-3); u[m + '_f0tiny_changes_vector'] = bool(np.abs(a_t - a_old).max() > 1e-3)
            u['n_nonfinite'] = nb; r0, _ = D0M.efn_vec(K, Vm, D0, n, lF + a_old); rt, _ = D0M.efn_vec(K, Vm, D0, n, lF + FX.apply_mode(raw, n, 'b1', 4.0, 1e-3)[0])
            u['e_old_c4'] = r0['dE_site']; u['e_b1_f0tiny'] = rt['dE_site']; u['f0tiny_changes_E'] = bool(abs(rt['dE_site'] - r0['dE_site']) > 1e-9)
            u['pass_'] = bool(all(v for k, v in u.items() if k.endswith(('equal_old', 'changes_vector', 'changes_E')))); res['unit'] = u; unit_done = True; dump(); log(f'unit {u}')
            if not u['pass_']: res['status'] = 'UNIT FAIL'; dump(); sys.exit(2)
        modes = ['b1', 'b2'] + (['a1', 'a2'] if nb else []) + (['old'] if name in R5 else [])
        for m in modes:
            sc = G.setdefault('scan', {}).setdefault(m, {})
            for c in CG:
                if str(c) in sc: continue
                t = time.time(); sc[str(c)] = solve(m, c); dump(); log(f'{name} {m} c={c} e {sc[str(c)]["e"]:.6e} wf {sc[str(c)]["weight_fail"]} {time.time()-t:.0f}s')
            cb = min(sc, key=lambda k: sc[k]['e']); G.setdefault('min', {})[m] = dict(c=float(cb), e=sc[cb]['e'], diff_vs_reported=sc[cb]['e'] - eaug, f0=sc[cb].get('f0'))
        G['done'] = True; dump(); log(f'{name} min {G["min"]}')
    del K, Vm, D0
# two-PP gain before/after (control minus two-PP at each variant's own min over the grid)
gain = {}
try:
    for sd in ('s1', 's2'):
        p, c_ = res['guides'][f'r5_pp2_{sd}']['min'], res['guides'][f'r5_ctl_{sd}']['min']
        gain[sd] = {m: c_[m]['e'] - p[m]['e'] for m in ('old', 'b1', 'b2')}
        gain[sd].update({m + '_rel_change': gain[sd][m] / gain[sd]['old'] - 1 for m in ('b1', 'b2')})
except KeyError as e: gain['error'] = str(e)
res['gain'] = gain; res['status'] = 'DONE'; res['done'] = True; dump(); log(f'gain {gain}'); log('DONE')
