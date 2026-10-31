"""r2g_readout.py: round-3 readout, arm r3_900 | r3_1200 (bare e only) | r3_1500 (bare + augmented scan). CPU numpy, one solve at a time. usage: python3 r2g_readout.py ARM(r3_900|r3_1200|r3_1500) OUTDIR
Solver/f_J as lswo_readout.py (bond_split.build + delta0.efn_vec)."""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
ARM = sys.argv[1]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
R2 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2'; R3 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3'
BASE = 1.4089759651645136e-05    # e(B_2) = pfaffian_r2g/readout/g2 e_fit_site (s_G)
FITC = dict(r3_900=[f'{R3}/tabs/tab_900/round_1/la_table.npy', f'{R3}/pdb_tabs/tab_900/round_1/la_table.npy'], r3_1200=[f'{R3}/tabs/tab_1200/round_1/la_table.npy', f'{R3}/pdb_tabs/tab_1200/round_1/la_table.npy'],
            r3_1500=[f'{R3}/fit/round_1/la_table.npy', f'{R3}/pdb_fit/round_1/la_table.npy'])
CFG = dict(r3_900=dict(FIT=f'{R3}/tabs/tab_900/round_1/la_table.npy', bare_only=True, label='round 3, theta at step 900 (table job)'),
           r3_1200=dict(FIT=f'{R3}/tabs/tab_1200/round_1/la_table.npy', bare_only=True, label='round 3, theta at step 1200 (table job)'),
           r3_1500=dict(FIT=f'{R3}/fit/round_1/la_table.npy', bare_only=False, label='round 3, final step (fit job, step count in fit/cx_r1.json)'))[ARM]
FIT_ = next((f for f in FITC[ARM] if os.path.exists(f)), FITC[ARM][0]); CFG['FIT'] = FIT_     # the winning cluster's copy (A40 dirs fit/tabs, or the Paderborn mirror pdb_fit/pdb_tabs)
CFG.update(S=f'{R2}/s_G.npy', TGT=f'{R3}/target_T3_sG.npy', base=BASE)
S1, FIT, TGT = CFG['S'], CFG['FIT'], CFG['TGT']
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RJ = f'{OUT}/readout.json'; res = json.load(open(RJ)) if os.path.exists(RJ) else {}
res.update(arm=ARM, host=os.uname().nodename, sign=ARM, s1=S1, s1_sha256=sha(S1), fit=FIT, fit_sha256=sha(FIT), target=TGT, target_sha256=sha(TGT), E0=BS.E0, base_e=CFG['base'], label=CFG['label'])
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
s1 = np.load(S1).astype(np.float64); assert set(np.unique(s1)) <= {-1.0, 1.0}
lF = np.load(FIT).astype(np.float64); lT = np.load(TGT).astype(np.float64); assert lF.shape == lT.shape == (D,) and np.isfinite(lF).all() and np.isfinite(lT).all()
K, Vm, D0 = BS.build(n, s1); log('built K,Vm (s1)')
# 1. gate
r = res['E_FN_fit'] if 'E_FN_fit' in res else D0M.efn_vec(K, Vm, D0, n, lF)[0]; eF = r['dE_site']; res['E_FN_fit'] = r; res['e_fit_site'] = eF
res['drop'] = CFG['base'] - eF; res['kill'] = 'STALL (drop <= 0.5e-6)' if res['drop'] <= 0.5e-6 else 'NOT STALLED'; dump()
log(f'1. e(fit) = {eF:.6e}  drop {res["drop"]:.4e} {res["kill"]}  it {r["iters"]} res {r["res"]:.1e}')
EP = r['E']
if CFG['bare_only']: res['status'] = 'BARE ONLY'; res['done'] = True; dump(); log(f'DONE bare only e {eF:.6e}'); sys.exit(0)
# 4. variational <H> of fit at s1 (E_guide_var = <a|H_FN|a> = <H>(s1 a), violating hops enter the diagonal with their true sign)
res['H_var_fit_site'] = (r['E_guide_var'] - BS.E0) / BS.NS; dump(); log(f'4. <H>(fit,s1) per site {res["H_var_fit_site"]:.6e}')
# 2. copy error vs T
cp = BS.cost_pass(n, s1, lT, lF); tot = cp['total_site']
res['bond_split'] = dict(cost_site=tot, npairs=cp['npairs'], J1=cp['by_class'][0] / cp['total'], J2=cp['by_class'][1] / cp['total'],
    hist_share=(cp['hist'] / cp['total']).tolist(), ratio_bins=BS.QB, dec_labels=BS.DEC.tolist(),
    col_share=(cp['hist'].sum(0) / cp['total']).tolist(), by_phi2_decade_share={int(d - 39): float(v / cp['total']) for d, v in enumerate(cp['by_phi2_decade']) if v > 0}); dump()
log(f'2. cost/site {tot:.4e}  J1 {res["bond_split"]["J1"]:.3f} J2 {res["bond_split"]["J2"]:.3f}  ratio-bin shares {res["bond_split"]["col_share"]}')
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
# 3. augmented: f_J[fit] with E' = E_FN[fit], s1 blocked bonds (as test_z.py (c))
fJ = np.empty(D); nbad = 0; dmin = np.inf
for i0 in range(0, D, 65536):
    i1 = min(D, i0 + 65536); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
    ss = s1[i0:i1, None] * s1[tt]; blk = valid & (ss > 0); kpt = valid & (ss < 0)
    ratio = np.exp(np.clip((lF - lF.max())[tt] - (lF - lF.max())[i0:i1, None], -700, 700))
    VA = (blk * BS.HW[None, :] * ratio).sum(1); SK = (kpt * BS.HW[None, :] * ratio).sum(1)
    den = 27.0 - 0.5 * (valid * BS.JW[None, :]).sum(1) + VA - EP; dmin = min(dmin, float(den.min())); nbad += int((den <= 0).sum() + (SK <= 0).sum())
    with np.errstate(divide='ignore', invalid='ignore'): fJ[i0:i1] = np.log(SK) - np.log(den)
bad = np.nonzero(~np.isfinite(fJ))[0]; res['fJ'] = dict(den_min=dmin, n_bad=nbad, bad_rows=[int(i) for i in bad]); assert len(bad) <= 3 and dmin > 0
fJ[bad] = 0.0; np.save(f'{OUT}/fJ_fit.npy', fJ); dump(); log(f'fJ[fit] {res["fJ"]}')
cache = {float(k): v for k, v in res.get('scan', {}).items()}; cache[0.0] = eF; res['scan'] = {str(k): v for k, v in cache.items()}      # resume: points already solved are kept (a killed run restarts from them)
def solve(c):
    c = round(float(c), 6)
    if c in cache: return cache[c]
    r, _ = D0M.efn_vec(K, Vm, D0, n, lF + c * fJ); cache[c] = r['dE_site']; res['scan'][str(c)] = r['dE_site']; dump()
    log(f'3. c={c:.3f} e {r["dE_site"]:.6e} it {r["iters"]} res {r["res"]:.1e}'); return r['dE_site']
for c in (1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0): solve(c)
for it in range(8):
    b = min(cache, key=cache.get)
    if b >= 7.0: res['status'] = 'MIN AT UPPER EDGE'; break
    if b <= 0.0: res['status'] = 'MIN AT c=0'; break
    cs = sorted(cache); i = cs.index(b); trio = cs[i - 1:i + 2]
    p = np.polyfit(trio, [cache[c] for c in trio], 2)
    if p[0] <= 0: res['status'] = 'NOT CONVEX'; break
    cv = round(float(np.clip(-p[1] / (2 * p[0]), 0, 7)) / 0.1) * 0.1; log(f'iter {it}: trio {trio} vertex-> {cv:.2f}')
    if round(cv, 6) in cache: break
    solve(cv)
cst = min(cache, key=cache.get); res['cstar'] = cst; res['e_aug_min_site'] = cache[cst]; res.setdefault('status', 'DONE'); res['done'] = True; dump()
log(f'3. c* {cst} e_aug {cache[cst]:.6e}  DONE')
