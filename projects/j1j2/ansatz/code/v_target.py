"""v_target.py: gate V's target T_5^b1 = norm_la(la(theta_4) + add_b1), add_b1 = fj_fix.apply_mode(raw f_J^G[theta_4], mode 'b1', c = 4.0) (PLAN "FIX", "V (narrowed)").
E' = E_FN[theta_4, s_G] as r5_prep.py solved it (pfaffian_r5/prep_result.json E_bar_fresh.E).  Controls: mode 'old' with the same raw f_J must reproduce
pfaffian_r5/target_T5_sG.npy (max |diff| < 1e-12).  Weight check RELATIVE (coordinator 2026-11-03): raise only if the augmented max single-orbit share exceeds
10x the base guide's (fj_fix.weight_check's absolute 1e-3 fires on a bulk 6x6 orbit for both, share 0.0093).   usage: python3 v_target.py OUTDIR"""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS, fj_fix as FX
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time(); sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
R4 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4'; R5 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5'; SG = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy'
TB = f'{R4}/pdb_tabs/tab_t4/round_1/la_table.npy'; pr = json.load(open(f'{R5}/prep_result.json')); assert sha(TB) == pr['tbar_table_sha256'] and sha(SG) == pr['sG_sha256']
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); lF = np.load(TB).astype(np.float64); s = np.load(SG).astype(np.float64); EP = pr['E_bar_fresh']['E']
def norm_la(la): la = la - la.max(); return la - 0.5 * np.log((n * np.exp(2 * la)).sum())     # verbatim r5_prep.py
def share(la):
    w = n * np.exp(2 * (la - la.max())); w /= w.sum(); i = int(np.argmax(w)); return float(w[i]), i
raw, dmin = FX.raw_fJ(lF, EP, s); assert dmin > 0
old, _ = FX.apply_mode(raw, n, 'old', 4.0); lold = norm_la(lF + old); T5old = np.load(f'{R5}/target_T5_sG.npy'); dold = float(np.max(np.abs(lold - T5old)))
add, info = FX.apply_mode(raw, n, 'b1', 4.0); lT = norm_la(lF + add)
sb, ib = share(lF); sa, ia = share(lT); rel_ok = sa <= 10 * sb
res = dict(EP=EP, den_min=dmin, info=info, control_old_vs_T5_max_abs=dold, control_pass=bool(dold < 1e-12), base_max_share=sb, base_argmax=ib, aug_max_share=sa, aug_argmax=ia,
           relative_weight_check_pass=bool(rel_ok), n_rows_capped=int(np.sum(np.isfinite(raw) & (np.abs(raw) > info['f0']))),
           max_abs_change_vs_old_T5=float(np.max(np.abs(lT - T5old))), code_sha256={f: sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in ('v_target.py', 'fj_fix.py')})
assert res['control_pass'], f'old-mode control {dold}'
if not rel_ok: json.dump(res, open(f'{OUT}/v_target.json', 'w'), indent=1); raise SystemExit(f'relative weight check: aug {sa:.3e} > 10 x base {sb:.3e}')
np.save(f'{OUT}/target_T5_b1_sG.npy', lT); res['T5_b1_sha256'] = sha(f'{OUT}/target_T5_b1_sG.npy'); res['sec'] = time.time() - T0
json.dump(res, open(f'{OUT}/v_target.json', 'w'), indent=1, default=float); print(json.dumps(res, default=float))
