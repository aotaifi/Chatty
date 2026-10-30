"""r1_readout.py: R1 readout (PLAN_pfaffian.md "R1 readout AMENDED" + "CORRECTION 2026-10-30": complex PP, no backflow, RBM inactive), sign s1, CPU, one solve at a time.
Solver/f_J as test_z.py / lswo_readout.py. usage: python3 r1_readout.py OUTDIR"""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
S1 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_z/s1.npy'
FIT = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r1/run_a40/round_1/la_table.npy'
TGT = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r1/target_la_s1.npy'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RJ = f'{OUT}/readout.json'; res = json.load(open(RJ)) if os.path.exists(RJ) else {}
res.update(host=os.uname().nodename, sign='s1', s1=S1, s1_sha256=sha(S1), fit=FIT, fit_sha256=sha(FIT), target=TGT, target_sha256=sha(TGT), E0=BS.E0,
           note='fit NOT converged (935 steps budget-limited, held-out change 0.0426); class complex PP, no backflow, RBM inactive')
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
assert res['fit_sha256'].startswith('fae7af85')
E_COPY, E_PHI2, DELTA = 2.503377e-5, 1.607666e-5, 8.957e-6
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
s1 = np.load(S1).astype(np.float64); assert set(np.unique(s1)) <= {-1.0, 1.0}
lF = np.load(FIT).astype(np.float64); lT = np.load(TGT).astype(np.float64); assert lF.shape == lT.shape == (D,) and np.isfinite(lF).all() and np.isfinite(lT).all()
K, Vm, D0 = BS.build(n, s1); log('built K,Vm (s1)')
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
# 1. gate
r = res['E_FN_fit'] if 'E_FN_fit' in res else D0M.efn_vec(K, Vm, D0, n, lF)[0]; eB = r['dE_site']; res['E_FN_fit'] = r; res['e_B_site'] = eB
res['r'] = (E_COPY - eB) / (E_COPY - E_PHI2)
res['gate'] = 'PASS' if res['r'] >= 0.5 else ('KILL' if res['r'] <= 0 else 'BETWEEN'); dump()
log(f'1. e(B) = {eB:.6e}  r {res["r"]:.4f}  gate {res["gate"]}  it {r["iters"]} res {r["res"]:.1e}')
EP = r['E']
res['H_var_fit_site'] = (r['E_guide_var'] - BS.E0) / BS.NS; dump(); log(f'4. <H>(fit,s1) per site {res["H_var_fit_site"]:.6e}')
# 2. bond split of fit against phi''
cp = BS.cost_pass(n, s1, lT, lF); tot = cp['total_site']
res['bond_split'] = dict(cost_site=tot, npairs=cp['npairs'], J1=cp['by_class'][0] / cp['total'], J2=cp['by_class'][1] / cp['total'],
    hist_share=(cp['hist'] / cp['total']).tolist(), ratio_bins=BS.QB, dec_labels=BS.DEC.tolist(),
    col_share=(cp['hist'].sum(0) / cp['total']).tolist(), by_phi2_decade_share={int(d - 39): float(v / cp['total']) for d, v in enumerate(cp['by_phi2_decade']) if v > 0}); dump()
log(f'2. cost/site {tot:.4e}  J1 {res["bond_split"]["J1"]:.3f} J2 {res["bond_split"]["J2"]:.3f}  ratio-bin shares {res["bond_split"]["col_share"]}')
# 3. f_J (s1 blocked bonds, E' = E_FN of the base) and scans
def make_fJ(l, Ep, name):
    fJ = np.empty(D); nbad = 0; dmin = np.inf
    for i0 in range(0, D, 65536):
        i1 = min(D, i0 + 65536); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        ss = s1[i0:i1, None] * s1[tt]; blk = valid & (ss > 0); kpt = valid & (ss < 0)
        ratio = np.exp(np.clip((l - l.max())[tt] - (l - l.max())[i0:i1, None], -700, 700))
        VA = (blk * BS.HW[None, :] * ratio).sum(1); SK = (kpt * BS.HW[None, :] * ratio).sum(1)
        den = 27.0 - 0.5 * (valid * BS.JW[None, :]).sum(1) + VA - Ep; dmin = min(dmin, float(den.min())); nbad += int((den <= 0).sum() + (SK <= 0).sum())
        with np.errstate(divide='ignore', invalid='ignore'): fJ[i0:i1] = np.log(SK) - np.log(den)
    bad = np.nonzero(~np.isfinite(fJ))[0]; res['fJ_' + name] = dict(den_min=dmin, n_bad=nbad, bad_rows=[int(i) for i in bad]); assert len(bad) <= 3 and dmin > 0
    fJ[bad] = 0.0; np.save(f'{OUT}/fJ_{name}.npy', fJ); dump(); log(f'fJ[{name}] {res["fJ_" + name]}'); return fJ
def scan(l, fJ, e0, key):
    cache = {0.0: e0}; res[key] = {'0.0': e0}
    def solve(c):
        c = round(float(c), 6)
        if c in cache: return cache[c]
        rr, _ = D0M.efn_vec(K, Vm, D0, n, l + c * fJ); cache[c] = rr['dE_site']; res[key][str(c)] = rr['dE_site']; dump()
        log(f'{key} c={c:.3f} e {rr["dE_site"]:.6e} it {rr["iters"]} res {rr["res"]:.1e}'); return rr['dE_site']
    for c in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0): solve(c)
    status = 'DONE'
    for it in range(8):
        b = min(cache, key=cache.get)
        if b >= 6.0: status = 'MIN AT UPPER EDGE'; break
        if b <= 0.0: status = 'MIN AT c=0'; break
        cs = sorted(cache); i = cs.index(b); trio = cs[i - 1:i + 2]
        p = np.polyfit(trio, [cache[c] for c in trio], 2)
        if p[0] <= 0: status = 'NOT CONVEX'; break
        cv = round(float(np.clip(-p[1] / (2 * p[0]), 0, 6)) / 0.1) * 0.1; log(f'{key} iter {it}: trio {trio} vertex-> {cv:.2f}')
        if round(cv, 6) in cache: break
        solve(cv)
    cst = min(cache, key=cache.get); return dict(cstar=cst, e_min=cache[cst], status=status)
fJB = make_fJ(lF, EP, 'B'); res['aug_B'] = scan(lF, fJB, eB, 'scan_B'); dump(); log(f'3. augB {res["aug_B"]}')
# reference: phi'' (its own bare e solved here, to check the recorded 1.607666e-5)
rp, _ = D0M.efn_vec(K, Vm, D0, n, lT); res['E_FN_phi2'] = rp; ep = rp['dE_site']; dump(); log(f'e(phi\'\') recomputed {ep:.6e} (recorded {E_PHI2:.6e})')
fJP = make_fJ(lT, rp['E'], 'phi2'); res['aug_phi2'] = scan(lT, fJP, ep, 'scan_phi2'); dump(); log(f'3. augphi2 {res["aug_phi2"]}')
res['tau_over_Delta'] = (res['aug_B']['e_min'] - E_PHI2) / DELTA
res['tau_ref_over_Delta'] = (res['aug_B']['e_min'] - res['aug_phi2']['e_min']) / DELTA; res['done'] = True; dump()
log(f'tau/Delta {res["tau_over_Delta"]:.4f} tau_ref/Delta {res["tau_ref_over_Delta"]:.4f} DONE')
