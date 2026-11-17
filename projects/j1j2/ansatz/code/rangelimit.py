"""rangelimit.py: RUNG 2 range-limit check (STORAGE_PROBLEM.md blocks 'RANGE-LIMIT BOUND FOR (d)' and 'RUNG 2 PLAN'). CPU, oracle s2.
Machinery copied from cstar_lanczos.py / cstar.py (A_0 = s1 copy, BS.build, D0M.efn_vec, Hmag, moments). usage:
  python3 rangelimit.py OUTDIR A   steps 1-4: reproduce alpha=-0.10 Lanczos guide of A_0 (dE 8.682592e-06), keep FN state p,
                                   r* = ln p - ln A_0, a_x, residual K_p = a p, range-limit bound per L. Saves p (amplitude per orbit).
  python3 rangelimit.py OUTDIR B   steps 5-6: N(chi) (E_FN of chi's own best Lanczos guide, same line search as cstar_lanczos.py) and
                                   E_var[chi] for chi_0, phi = s2 p, chi_L (L = 4.88, 3, 2, 1, 0); rho_L.
Rep basis: LOBPCG vector v = sqrt(n) * amplitude (as BS.guide / efn_vec: w = a sqrt(n)), so p = |v| / sqrt(n), sum n p^2 = sum v^2 = 1."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; PH = sys.argv[2]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
A0 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy'
REF = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_s/cstar_lanczos.json'
ALPHA1 = -0.1; REF_DE = 8.682592190013673e-06; HVAR0 = 3.0852017474695494e-05   # cstar_lanczos.json alpha=-0.1; cstar.json Hvar['0.0']
LS = [1.0, 2.0, 3.0, 4.88]
E0 = -18.137147540071613; assert abs(BS.E0 - E0) < 1e-12
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s = np.load(BS.S2).astype(np.float64); D = len(n)
assert n.sum() == 9075135300
lA = np.load(A0).astype(np.float64); assert np.isfinite(lA).all()
RJ = f'{OUT}/rangelimit_{PH}.json'; res = dict(label='oracle s2', host=os.uname().nodename, A0=A0, S2=BS.S2, phase=PH, solves=[])
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
K, Vm, D0 = BS.build(n, s); log('built'); sn = np.sqrt(n)
def Hmag(b): return D0 * b + Vm @ b - (K @ (sn * b)) / sn      # as cstar_lanczos.py
def moments(lc):
    a = np.exp(lc - lc.max()); assert (a > 0).all()
    EL = Hmag(a) / a; w = n * a * a; Z = w.sum(); E = float((w * EL).sum() / Z); assert np.isfinite(EL).all()
    return E, EL - E
def solve(tag, lc, d, al):
    x = 1.0 + al * d; la = lc + np.log(np.maximum(np.abs(x), np.exp(-60.0))) if al != 0 else lc.copy()
    t = time.time(); r, v = D0M.efn_vec(K, Vm, D0, n, la)
    res['solves'].append(dict(chi=tag, alpha=al, dE_site=r['dE_site'], E=r['E'], iters=r['iters'], res=r['res'], n_abs_lt_0p05=int((np.abs(x) < 0.05).sum()), sec=time.time() - t)); dump()
    log(f'{tag} alpha={al:.4f} dE/site {r["dE_site"]:.6e} iters {r["iters"]} res {r["res"]:.1e} {time.time() - t:.0f}s'); return r, v, la

if PH == 'A':
    E, d = moments(lA); res['Evar_A0_site'] = (E - E0) / BS.NS
    log(f'E_var[A_0]/site {(E - E0) / BS.NS:.12e} (cstar.json Hvar 0.0: {HVAR0:.12e})')
    assert abs((E - E0) / BS.NS / HVAR0 - 1) < 1e-9
    r, v, laG = solve('step1', lA, d, ALPHA1); rel = r['dE_site'] / REF_DE - 1
    res['step1'] = dict(got=r['dE_site'], ref=REF_DE, rel=rel, E_FN=r['E']); dump(); log(f'step1 rel {rel:.2e}')
    if abs(rel) > 1e-5: res['status'] = 'STEP1 FAIL'; dump(); log('STEP1 FAIL: STOP'); sys.exit(2)
    sg = np.sign(v.sum()); v = v * sg
    res['p_sign'] = dict(n_neg=int((v < 0).sum()), n_zero=int((v == 0).sum()), min_v=float(v.min()))
    p = np.abs(v) / sn; p = p / np.sqrt((n * p * p).sum()); np.save(f'{OUT}/p.npy', p)
    EF = r['E']; G = BS.guide(laG, n); VG = (Vm @ G) / G; a = D0 + VG - EF
    Kp = (K @ (sn * p)) / sn; ap = a * p; R = np.abs(Kp - ap) / np.maximum(np.abs(Kp), np.abs(ap))
    w = n * p * p; im = int(np.argmax(R))
    res['residual'] = dict(max_rel=float(R.max()), argmax=im, p_at_max=float(p[im]), np2_at_max=float(w[im]), wmean_rel=float((w * R).sum()),
                           q999=float(np.quantile(R, 0.999)), n_rel_gt_1e3=int((R > 1e-3).sum()), w_rel_gt_1e3=float(w[R > 1e-3].sum()),
                           a_min=float(a.min()), n_a_neg=int((a < 0).sum()))
    log(f"residual K_p=a p: {res['residual']}"); dump()
    rs = np.log(np.maximum(p, 1e-300)) - lA; m = float((w * rs).sum()); res['m'] = m; np.save(f'{OUT}/rstar.npy', rs)
    res['rstar_minus_m'] = dict(min=float((rs - m).min()), max=float((rs - m).max()), wstd=float(np.sqrt((w * (rs - m) ** 2).sum())))
    res['bound'] = {}
    for L in LS:
        z = rs - m; cz = np.clip(z, -L, L); e = cz - z; t = p * np.exp(e)
        bad = np.abs(z) > L; under = z > L; over = z < -L
        num = float((n * a * np.maximum(p * p, t * t))[bad].sum()); den = float(w[~bad].sum())
        res['bound'][str(L)] = dict(bound_site=num / den / BS.NS, bound_total=num / den, w_bad=float(w[bad].sum()), w_under=float(w[under].sum()), w_over=float(w[over].sum()),
                                    n_under=int(under.sum()), n_over=int(over.sum()), num_under=float((n * a * np.maximum(p * p, t * t))[under].sum()),
                                    num_over=float((n * a * np.maximum(p * p, t * t))[over].sum()))
        log(f"L={L}: {res['bound'][str(L)]}")
    res['status'] = 'A DONE'; dump(); log('A DONE')

elif PH == 'B':
    p = np.load(f'{OUT}/p.npy'); rs = np.load(f'{OUT}/rstar.npy'); m = json.load(open(f'{OUT}/rangelimit_A.json'))['m']
    assert (p > 0).all()
    def N(tag, lc):   # cstar_lanczos.py line search on chi's own Lanczos guide
        E, d = moments(lc); cache = {}
        def f(al):
            al = round(float(al), 6)
            if al not in cache: cache[al] = solve(tag, lc, d, al)[0]['dE_site']
            return cache[al]
        for al in (0.0, -0.1, -0.2, -0.3, -0.45, -0.6): f(al)
        ext = False
        if min(cache, key=cache.get) == -0.6:
            log(f'{tag}: minimum at -0.6: extending once to -1.2'); ext = True
            for al in (-0.8, -1.0, -1.2): f(al)
        st = 'DONE'
        for it in range(8):
            cs = sorted(cache); bst = min(cs, key=cache.get); near = sorted(sorted(cs, key=lambda c: abs(c - bst))[:3])
            pp = np.polyfit(near, [cache[c] for c in near], 2)
            if pp[0] <= 0: st = 'NOT CONVEX'; log(f'{tag}: parabola not convex'); break
            v = float(np.clip(-pp[1] / (2 * pp[0]), min(cs), 0.0)); log(f'{tag} iter {it}: points {near} vertex {v:.4f}')
            if min(abs(v - c) for c in cache) < 0.01: break
            f(v)
        ast = min(cache, key=cache.get)
        out = dict(N=cache[ast], alpha_star=ast, Evar_site=(E - E0) / BS.NS, extended=ext, status=st, nsolves=len(cache))
        res.setdefault('chi', {})[tag] = out; dump(); log(f'{tag}: {out}'); return out
    c0 = N('chi_0', lA.copy()); ph = N('phi', np.log(p))
    B = c0['N'] - ph['N']; res['B'] = B; res['B_half'] = B / 2
    for L in (4.88, 3.0, 2.0, 1.0, 0.0):
        cL = N(f'chi_L{L}', lA + np.clip(rs - m, -L, L))
        cL['rho'] = (c0['N'] - cL['N']) / B; dump(); log(f'rho_{L} = {cL["rho"]:.6f}')
    res['status'] = 'B DONE'; dump(); log('B DONE')
