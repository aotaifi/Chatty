"""screen_s123.py: pre-registered screen S1-S3 (PLAN_pfaffian.md "Screen"), CPU, numpy. Sign s1 (oracle-free) = pfaffian_z/s1.npy throughout.
usage: python3 screen_s123.py OUTDIR. Reuses test_z.py logic (FN solves via delta0.efn_vec, f_J, c-scan). Stages cached in OUTDIR."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
RJ = f'{OUT}/screen_result.json'
res = json.load(open(RJ)) if os.path.exists(RJ) else {}
res['host'] = os.uname().nodename
def dump(): json.dump(res, open(RJ, 'w'), indent=1, default=float)
Z = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_z'
PHI1 = '/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/o2b_17431886/round_1/target_la.npy'
ASC = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy'
assert D0M.A0 == ASC, (D0M.A0, ASC)
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
HW = BS.HW; JW = BS.JW
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
s1 = np.load(f'{Z}/s1.npy').astype(np.float64)
def norm_la(la): la = la - la.max(); return la - 0.5 * np.log((n * np.exp(2 * la)).sum())
K, Vm, D0 = BS.build(n, s1); log('built K,Vm for s1 (Z/s1.npy)')
def efn(la, tag):
    t = time.time(); r, v = D0M.efn_vec(K, Vm, D0, n, la); log(f'{tag}: E_FN dE/site {r["dE_site"]:.6e} it {r["iters"]} res {r["res"]:.1e} {time.time()-t:.0f}s'); return r, v
def fJ_of(lg, EP):
    fJ = np.empty(D); nbad = 0; den_min = np.inf; l0 = lg - lg.max()
    for i0 in range(0, D, 65536):
        i1 = min(D, i0 + 65536); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        ss = s1[i0:i1, None] * s1[tt]; blk = valid & (ss > 0); kpt = valid & (ss < 0)
        ratio = np.exp(np.clip(l0[tt] - l0[i0:i1, None], -700, 700))
        VA = (blk * HW[None, :] * ratio).sum(1); SK = (kpt * HW[None, :] * ratio).sum(1)
        Hxx = 27.0 - 0.5 * (valid * JW[None, :]).sum(1); den = Hxx + VA - EP
        den_min = min(den_min, float(den.min())); nbad += int((den <= 0).sum())
        with np.errstate(divide='ignore', invalid='ignore'): fJ[i0:i1] = np.log(SK) - np.log(den)
    bad = np.nonzero(~np.isfinite(fJ))[0]
    info = dict(den_min=den_min, n_den_le0=nbad, n_nonfinite=len(bad), n_nonfinite_weight=float(n[bad].sum()), bad_rows_first=[int(i) for i in bad[:10]],
                fJ_max=float(np.nanmax(fJ[np.isfinite(fJ)])), fJ_min_finite=float(fJ[np.isfinite(fJ)].min()))
    assert den_min > 0 and len(bad) <= 50
    fJ[bad] = 0.0; return fJ, info
def scan(name, lg, fJ, E0site):
    cache = {0.0: E0site}; sc = res.setdefault(name, {}); sc['0.0'] = E0site
    for k, v in sc.items(): cache[float(k)] = v
    def solve(c):
        c = round(float(c), 4)
        if c in cache: return cache[c]
        r, _ = efn(lg + c * fJ, f'{name} c={c}'); cache[c] = r['dE_site']; sc[str(c)] = r['dE_site']; res.setdefault(name + '_meta', {})[str(c)] = dict(iters=r['iters'], res=r['res']); dump(); return cache[c]
    for c in (1.0, 2.0, 3.0, 4.0, 5.0): solve(c)
    for it in range(10):
        b = min(cache, key=cache.get); cs = sorted(cache); i = cs.index(b)
        trio = cs[max(i - 1, 0):i + 2] if i > 0 else cs[:3]
        if i == len(cs) - 1: trio = cs[-3:]
        p = np.polyfit(trio, [cache[c] for c in trio], 2)
        if p[0] <= 0: res[name + '_status'] = 'NOT CONVEX'; break
        cv = round(float(np.clip(-p[1] / (2 * p[0]), 0.0, 5.0)) * 10) / 10; log(f'{name} iter {it}: best {b} trio {trio} vertex(0.1) {cv}')
        if round(cv, 4) in cache: break
        solve(cv)
    # also neighbours +-0.1 of the best, to certify the 0.1 minimum
    for _ in range(10):
        b = min(cache, key=cache.get); new = [round(c, 4) for c in (b - 0.1, b + 0.1) if -1e-9 <= c <= 5.0001 and round(c, 4) not in cache]
        if not new: break
        for c in new: solve(c)
    b = min(cache, key=cache.get); res[name + '_min'] = dict(c=b, dE_site=cache[b]); dump(); log(f'{name} min c={b} {cache[b]:.6e}'); return cache
# ---- S1
lA = np.load(ASC).astype(np.float64)
if 'S1' not in res:
    r, _ = efn(lA, 'check E_FN[A_s1]'); zj = json.load(open(f'{Z}/z_result.json')); assert abs(r['dE_site'] - zj['b']['E_FN_A']['dE_site']) < 1e-9 * 1 + 1e-10
    fJA = np.load(f'{Z}/fJ.npy'); assert fJA.shape == (D,)
    lA1 = lA + 3.9 * fJA; np.save(f'{OUT}/ln_A1.npy', lA1)
    r1, v = efn(lA1, 'S1 FN(A_1) -> phi_2'); chi = np.abs(v) / np.sqrt(n); lphi2 = np.log(np.maximum(chi, 1e-300)); lphi2 = norm_la(lphi2); np.save(f'{OUT}/ln_phi2.npy', lphi2)
    Z1 = zj['dE_cstar_site']; rel = (r1['dE_site'] - Z1) / Z1
    r2, _ = efn(lphi2, 'S1 E_FN[phi_2 guide]')
    res['S1'] = dict(E_FN_A1=r1, Z_value=Z1, rel_vs_Z=rel, check_1em5=bool(abs(rel) < 1e-5), E_FN_phi2=r2, Delta_site=r1['dE_site'] - r2['dE_site'],
                     Delta_vs_formula=Z1 - r2['dE_site']); dump(); log(f'S1 rel {rel:.2e}, Delta {res["S1"]["Delta_site"]:.6e}')
lphi2 = np.load(f'{OUT}/ln_phi2.npy'); Ep2 = res['S1']['E_FN_phi2']['E']; dphi2 = res['S1']['E_FN_phi2']['dE_site']; Delta = res['S1']['Z_value'] - dphi2
# ---- S2
FJ2 = f'{OUT}/fJ_phi2.npy'
if not os.path.exists(FJ2):
    f, info = fJ_of(lphi2, Ep2); res['fJ_phi2'] = info; dump(); np.save(FJ2, f); log(f'fJ[phi_2] {info}')
fJ2 = np.load(FJ2); c2 = scan('S2_scan', lphi2, fJ2, dphi2)
# ---- S3
if 'S3' not in res:
    lA1n = norm_la(np.load(ASC).astype(np.float64)); lp1 = norm_la(np.load(PHI1).astype(np.float64)); e1 = lA1n - lp1
    np.save(f'{OUT}/e1.npy', e1); B2 = lphi2 + e1; np.save(f'{OUT}/ln_B2.npy', B2)
    rB, _ = efn(B2, 'S3 E_FN[B_2]'); res['S3'] = dict(E_FN_B2=rB, e1_std_n=float(np.sqrt((n * (e1 - (n * e1).sum() / n.sum()) ** 2).sum() / n.sum())),
                                                  B2_minus_phi2=rB['dE_site'] - dphi2); dump()
B2 = np.load(f'{OUT}/ln_B2.npy'); dB2 = res['S3']['E_FN_B2']['dE_site']
FJB = f'{OUT}/fJ_B2.npy'
if not os.path.exists(FJB):
    f, info = fJ_of(B2, res['S3']['E_FN_B2']['E']); res['fJ_B2'] = info; dump(); np.save(FJB, f); log(f'fJ[B_2] {info}')
fJB = np.load(FJB); cB = scan('S3_scan', B2, fJB, dB2)
mB = res['S3_scan_min']['dE_site']; res['tau_site'] = mB - dphi2; res['Delta_site'] = Delta; res['tau_over_Delta'] = (mB - dphi2) / Delta; res['status'] = 'DONE'; dump()
log(f'tau {res["tau_site"]:.6e} Delta {Delta:.6e} tau/Delta {res["tau_over_Delta"]:.3f} DONE')
