"""cstar_lanczos.py: Control S (PLAN_pfaffian.md; oracle s2). Scanned Lanczos factor on the s1 copy:
ln|A'| = ln|A_s1| + ln|1 + alpha (E_L - E)|, sign s2, psi = s2|A_s1|, E = <H>_psi (exact). usage: python3 cstar_lanczos.py OUTDIR
Same solver/copy/sign as cstar.py. Checks alpha=0 (2.092013e-5), then exact line search on E_FN over alpha in [-0.6,0] (extend to -1.2 once)."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
A0 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy'
E0 = -18.137147540071613; assert abs(BS.E0 - E0) < 1e-12
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s = np.load(BS.S2).astype(np.float64); D = len(n)
assert n.sum() == 9075135300
lA = np.load(A0).astype(np.float64)
res = dict(label='oracle s2', host=os.uname().nodename, A0=A0, S2=BS.S2, pairs=[]); RJ = f'{OUT}/cstar_lanczos.json'
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
K, Vm, D0 = BS.build(n, s); log('built'); sn = np.sqrt(n)
a = np.exp(lA - lA.max()); assert (a > 0).all()
def Hmag(b):  # (H u)(x)/s_x for u = s*b (psi sign s2 symmetric); full H, rep basis: D0 b + Vm b - K(sn b)/sn
    return D0 * b + Vm @ b - (K @ (sn * b)) / sn
Ha = Hmag(a); EL = Ha / a; w = n * a * a; Z = w.sum(); E = float((w * EL).sum() / Z)
assert np.isfinite(EL).all()
d = EL - E; mu2 = float((w * d * d).sum() / Z)
b = a * d; Hb = Hmag(b); mu3 = float((n * b * (Hb - E * b)).sum() / Z)   # <u1|(H-E)|u1>, u1=(H-E)psi
alG = (mu3 - np.sqrt(mu3 ** 2 + 4 * mu2 ** 3)) / (2 * mu2 ** 2)
res['moments'] = dict(E=E, dE_site=(E - E0) / BS.NS, mu2=mu2, mu3=mu3, alpha_moment=alG, EL_min=float(EL.min()), EL_max=float(EL.max()))
log(f'E<H>_psi {E:.10f} dE/site {(E - E0) / BS.NS:.6e} mu2 {mu2:.6e} mu3 {mu3:.6e} alpha_moment {alG:.5f}'); dump()
del Ha, Hb, b
cache = {}
def solve(al):
    al = round(float(al), 6)
    if al in cache: return cache[al]
    x = 1.0 + al * d; nz = int((np.abs(x) < 1e-6).sum()); n05 = int((np.abs(x) < 0.05).sum())
    la = lA + np.log(np.maximum(np.abs(x), np.exp(-60.0))) if al != 0 else lA.copy()
    t = time.time(); r, _ = D0M.efn_vec(K, Vm, D0, n, la)
    cache[al] = r['dE_site']; res['pairs'].append(dict(alpha=al, dE_site=r['dE_site'], E=r['E'], iters=r['iters'], res=r['res'], n_abs_lt_0p05=n05, n_abs_lt_1e6=nz, sec=time.time() - t)); dump()
    log(f'alpha={al:.4f} dE/site {r["dE_site"]:.6e} iters {r["iters"]} res {r["res"]:.1e} |x|<0.05: {n05} |x|<1e-6: {nz}'); return r['dE_site']
r0 = solve(0.0); res['check_alpha0'] = dict(got=r0, ref=2.092013e-5, rel=r0 / 2.092013e-5 - 1); dump()
log(f"check alpha=0 rel {res['check_alpha0']['rel']:.2e}")
if abs(res['check_alpha0']['rel']) > 1e-5: res['status'] = 'CHECK FAIL'; dump(); log('CHECK FAIL: STOP'); sys.exit(2)
if len(sys.argv) > 2:  # extra points around the minimum (no new search)
    for al in map(float, sys.argv[2].split(',')): solve(al)
    res['status'] = 'EXTRA DONE'; dump(); log('EXTRA DONE'); sys.exit(0)
for al in (-0.1, -0.2, -0.3, -0.45, -0.6): solve(al)
if min(cache, key=cache.get) == -0.6:
    log('minimum at -0.6: extending once to -1.2'); res['extended'] = True
    for al in (-0.8, -1.0, -1.2): solve(al)
for it in range(8):
    cs = sorted(cache); bst = min(cs, key=cache.get)
    near = sorted(sorted(cs, key=lambda c: abs(c - bst))[:3])
    p = np.polyfit(near, [cache[c] for c in near], 2)
    if p[0] <= 0: res['status'] = 'NOT CONVEX'; dump(); log('parabola not convex: STOP'); break
    lo = min(cs); v = float(np.clip(-p[1] / (2 * p[0]), lo, 0.0)); log(f'iter {it}: points {near} vertex {v:.4f}')
    if min(abs(v - c) for c in cache) < 0.01: break
    solve(v)
ast = min(cache, key=cache.get); res['alpha_star'] = ast; res['dE_alpha_star'] = cache[ast]
res['alpha_star_vertex'] = v if 'v' in dir() else None; res['status'] = res.get('status', 'DONE'); dump()
log(f'alpha* {ast} dE {cache[ast]:.6e}'); log('DONE')
