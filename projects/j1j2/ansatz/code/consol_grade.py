"""consol_grade.py: grader of the 6x6 consolidation test (STORAGE_PROBLEM.md 'CONSOLIDATION TEST 6x6 SPEC'), CPU, exact 6x6 symmetric-sector tables.
usage: python3 consol_grade.py OUTDIR TAG=LA_TABLE.npy [TAG=LA_TABLE.npy ...]
For each table chi (log|chi| on the 15804956 orbit reps, the ordering of A_0's la_table.npy; sign s2 fixed, the oracle sign of refresh_17409419):
  E_var[s2 chi]  = (<chi|H|chi>/<chi|chi> - E0)/36      (rangelimit_rounds.moments)
  E_FN[s2 chi]   = FN ground energy of H_FN[chi, s2], per site above E0 (delta0.efn_vec / bond_split.guide, LOBPCG tol 1e-6, maxit 60)
  N(chi)         = min over alpha of E_FN[s2 chi |1 + alpha (E_L - E)|]   with the alpha search of rangelimit_rounds.py (grid 0, -0.1, -0.2, -0.3, -0.45,
                   -0.6 (+ -0.8, -1.0, -1.2 if the minimum is at -0.6), then parabola vertices through the 3 evaluated points nearest the best one until a
                   vertex is within 0.01 of an evaluated point, <= 8 iterations); N = the lowest EVALUATED value (as rangelimit_rounds alpha_k).
  retention_N    = (N(A_0) - N(chi)) / (N(A_0) - N(A_1)),  N(A_0) = 8.683e-6, N(A_1) = 3.950e-6 per site (the spec's constants, not changed)
  certificate    = E_var[s2 chi] < 8.683e-6
Writes OUTDIR/<TAG>.json per table and prints one line 'GRADE <TAG> N=.. retention_N=.. E_FN=.. E_var=.. certificate=..'."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
N_A0, N_A1 = 8.683e-6, 3.950e-6
OUT = sys.argv[1]; ITEMS = [a.split('=', 1) for a in sys.argv[2:]]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s = np.load(BS.S2).astype(np.float64); D = len(n); assert n.sum() == 9075135300
K, Vm, D0 = BS.build(n, s); log('built'); sn = np.sqrt(n)
def Hmag(b): return D0 * b + Vm @ b - (K @ (sn * b)) / sn
def moments(la):
    a = np.exp(la - la.max()); EL = Hmag(a) / a; w = n * a * a; E = float((w * EL).sum() / w.sum()); assert np.isfinite(EL).all()
    return E, EL - E
def lanc(la, d, al): return la + np.log(np.maximum(np.abs(1.0 + al * d), np.exp(-60.0))) if al != 0 else la.copy()
for tag, path in ITEMS:
    la = np.load(path).astype(np.float64); assert la.shape == (D,) and np.isfinite(la).all(), (path, la.shape)
    rec = dict(tag=tag, table=path, N_A0=N_A0, N_A1=N_A1); RJ = f'{OUT}/{tag}.json'
    def dump(): rec['sec'] = time.time() - T0; json.dump(rec, open(RJ, 'w'), indent=1, default=float)
    E, d = moments(la); rec['E_var_site'] = (E - BS.E0) / BS.NS; rec['certificate'] = bool(rec['E_var_site'] < N_A0); log(f'{tag}: E_var[s2 chi]/site {rec["E_var_site"]:.6e}'); dump()
    cache = {}
    def f(al):
        al = round(float(al), 6)
        if al not in cache:
            t = time.time(); r, v = D0M.efn_vec(K, Vm, D0, n, lanc(la, d, al)); del v; r['sec'] = time.time() - t; cache[al] = r
            rec.setdefault('solves', []).append(dict(alpha=al, **r)); dump(); log(f'{tag} alpha={al}: dE/site {r["dE_site"]:.6e} iters {r["iters"]} res {r["res"]:.1e} {r["sec"]:.0f}s')
        return cache[al]['dE_site']
    for al in (0.0, -0.1, -0.2, -0.3, -0.45, -0.6): f(al)
    if min(cache, key=lambda a_: cache[a_]['dE_site']) == -0.6:
        for al in (-0.8, -1.0, -1.2): f(al)
    vtx = None
    for it in range(8):
        cs = sorted(cache); bst = min(cs, key=lambda a_: cache[a_]['dE_site']); near = sorted(sorted(cs, key=lambda c: abs(c - bst))[:3])
        pp = np.polyfit(near, [cache[c]['dE_site'] for c in near], 2)
        if pp[0] <= 0: rec['line_search'] = 'NOT CONVEX'; break
        vtx = float(np.clip(-pp[1] / (2 * pp[0]), min(cs), 0.0)); log(f'  iter {it}: points {near} vertex {vtx:.4f}')
        if min(abs(vtx - c) for c in cache) < 0.01: break
        f(vtx)
    ak = min(cache, key=lambda a_: cache[a_]['dE_site'])
    rec.update(alpha=ak, alpha_vertex=vtx, E_FN_site=cache[0.0]['dE_site'], N=cache[ak]['dE_site'], grid={str(a_): cache[a_]['dE_site'] for a_ in sorted(cache)})
    rec['retention_N'] = (N_A0 - rec['N']) / (N_A0 - N_A1); rec['done'] = True; dump()
    log(f'GRADE {tag} N={rec["N"]:.6e} (alpha {ak}) retention_N={rec["retention_N"]:.4f} E_FN={rec["E_FN_site"]:.6e} E_var={rec["E_var_site"]:.6e} certificate={rec["certificate"]}')
log('DONE')
