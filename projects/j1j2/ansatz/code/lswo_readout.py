"""lswo_readout.py: L-SWO readout (PLAN_pfaffian.md "Label-free arm L-SWO"), sign s1 only, CPU numpy, one solve at a time.
Solver/f_J as test_z.py (bond_split.build + delta0.efn_vec). usage: python3 lswo_readout.py OUTDIR"""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
S1 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_z/s1.npy'
FIT = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo/run/round_1/la_table.npy'
TGT = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo/target_la_T.npy'
A_S1 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RJ = f'{OUT}/readout.json'; res = json.load(open(RJ)) if os.path.exists(RJ) else {}
res.update(host=os.uname().nodename, sign='s1', s1=S1, s1_sha256=sha(S1), fit=FIT, fit_sha256=sha(FIT), target=TGT, target_sha256=sha(TGT), E0=BS.E0)
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
assert res['fit_sha256'].startswith('b65ab273')
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
s1 = np.load(S1).astype(np.float64); assert set(np.unique(s1)) <= {-1.0, 1.0}
lF = np.load(FIT).astype(np.float64); lT = np.load(TGT).astype(np.float64); assert lF.shape == lT.shape == (D,) and np.isfinite(lF).all() and np.isfinite(lT).all()
K, Vm, D0 = BS.build(n, s1); log('built K,Vm (s1)')
# 1. gate
r = res['E_FN_fit'] if 'E_FN_fit' in res else D0M.efn_vec(K, Vm, D0, n, lF)[0]; eF = r['dE_site']; res['E_FN_fit'] = r; res['e_fit_site'] = eF
res['tau_over_Delta'] = (eF - 1.0185e-5) / 1.485e-5
res['gate'] = 'PASS' if eF <= 1.761e-5 else ('KILL' if eF >= 2.503e-5 else 'BETWEEN'); dump()
log(f'1. e(fit) = {eF:.6e}  tau/Delta {res["tau_over_Delta"]:.4f}  gate {res["gate"]}  it {r["iters"]} res {r["res"]:.1e}')
EP = r['E']
# 4. variational <H> of fit at s1 (E_guide_var = <a|H_FN|a> = <H>(s1 a), violating hops enter the diagonal with their true sign)
res['H_var_fit_site'] = (r['E_guide_var'] - BS.E0) / BS.NS; dump(); log(f'4. <H>(fit,s1) per site {res["H_var_fit_site"]:.6e}')
# 2. copy error vs T
cp = BS.cost_pass(n, s1, lT, lF); tot = cp['total_site']
res['bond_split'] = dict(cost_site=tot, npairs=cp['npairs'], J1=cp['by_class'][0] / cp['total'], J2=cp['by_class'][1] / cp['total'],
    hist_share=(cp['hist'] / cp['total']).tolist(), ratio_bins=BS.QB, dec_labels=BS.DEC.tolist(),
    col_share=(cp['hist'].sum(0) / cp['total']).tolist(), by_phi2_decade_share={int(d - 39): float(v / cp['total']) for d, v in enumerate(cp['by_phi2_decade']) if v > 0}); dump()
log(f'2. cost/site {tot:.4e}  J1 {res["bond_split"]["J1"]:.3f} J2 {res["bond_split"]["J2"]:.3f}  ratio-bin shares {res["bond_split"]["col_share"]}')
# deep band: e = ln fit - ln T (both norm'd), phi_T^2 = per-config T^2 (sum n T^2 = 1) < 1e-11; per-orbit cost c_x as mj_split
def norm_la(la): la = la - la.max(); return la - 0.5 * np.log((n * np.exp(2 * la)).sum())
lTn, lFn = norm_la(lT), norm_la(lF); e = lFn - lTn; p2 = np.exp(2 * lTn)
cx = np.zeros(D); t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r'); R = 65536
for i0 in range(0, D, R):
    i1 = min(D, i0 + R); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
    viol = valid & (s1[i0:i1, None] * s1[tt] > 0); rr, kk = np.nonzero(viol); x = rr + i0; y = tt[rr, kk]; H = BS.HW[kk]
    sw = lTn[x] > lTn[y]; ls_, ll_ = np.where(sw, lTn[y], lTn[x]), np.where(sw, lTn[x], lTn[y]); cs_, cl_ = np.where(sw, lFn[y], lFn[x]), np.where(sw, lFn[x], lFn[y])
    xs = np.where(sw, y, x); rs = np.exp(np.clip(ls_ - ll_, -300, 0)); r_ = np.exp(np.clip(cs_ - cl_, -300, 300))
    cx += np.bincount(xs, weights=H * np.exp(2 * ll_) * (r_ - rs) ** 2 / r_ * n[x] * 0.5, minlength=D)
def wmed(v, w):
    o = np.argsort(v); cw = np.cumsum(w[o]); return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])
deep = p2 < 1e-11; mid = (p2 >= 1e-11) & (p2 < 1e-9)
res['deep'] = dict(n_orb=int(deep.sum()), frac_orb=float(deep.mean()), cost_share=float(cx[deep].sum() / cx.sum()), median_e_orbit=float(np.median(e[deep])),
    median_e_weight_n=wmed(e[deep], n[deep]), median_e_cost_weighted=wmed(e[deep], cx[deep]), mean_e_cost_weighted=float((cx[deep] * e[deep]).sum() / cx[deep].sum()),
    cost_total_check_site=float(cx.sum() / BS.NS), cost_vs_cost_pass=float(cx.sum() / cp['total']),
    mid_median_e_orbit=float(np.median(e[mid])), mid_cost_share=float(cx[mid].sum() / cx.sum())); dump(); log(f'2. deep {res["deep"]}')
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
cache = {0.0: eF}; res['scan'] = {'0.0': eF}
def solve(c):
    c = round(float(c), 6)
    if c in cache: return cache[c]
    r, _ = D0M.efn_vec(K, Vm, D0, n, lF + c * fJ); cache[c] = r['dE_site']; res['scan'][str(c)] = r['dE_site']; dump()
    log(f'3. c={c:.3f} e {r["dE_site"]:.6e} it {r["iters"]} res {r["res"]:.1e}'); return r['dE_site']
for c in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0): solve(c)
for it in range(8):
    b = min(cache, key=cache.get)
    if b >= 6.0: res['status'] = 'MIN AT UPPER EDGE'; break
    if b <= 0.0: res['status'] = 'MIN AT c=0'; break
    cs = sorted(cache); i = cs.index(b); trio = cs[i - 1:i + 2]
    p = np.polyfit(trio, [cache[c] for c in trio], 2)
    if p[0] <= 0: res['status'] = 'NOT CONVEX'; break
    cv = round(float(np.clip(-p[1] / (2 * p[0]), 0, 6)) / 0.1) * 0.1; log(f'iter {it}: trio {trio} vertex-> {cv:.2f}')
    if round(cv, 6) in cache: break
    solve(cv)
cst = min(cache, key=cache.get); res['cstar'] = cst; res['e_aug_min_site'] = cache[cst]; res.setdefault('status', 'DONE'); res['done'] = True; dump()
log(f'3. c* {cst} e_aug {cache[cst]:.6e}  DONE')
