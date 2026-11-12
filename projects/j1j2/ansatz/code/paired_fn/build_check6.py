#!/usr/bin/env python3
"""build_check6.py (paired_fn): build every 6x6 reference guide of guides6.GUIDES and check it against the saved deterministic solves.
Checks per guide: (1) variational energy sum n a^2 E_L / sum n a^2 from the paired_fn rows == stored E_guide_var (1e-9 total) where stored;
(2) for guides without a stored E_guide_var (the c-guides): one exact FN solve (delta0.efn_vec, tol 1e-9, maxit 150, the q0r.py solver) must
reproduce the stored F to 1e-10/site, and its E_guide_var must equal (1); (3) T0B0_c47's rebuilt la equals sign_fni/la_a.npy.
usage: build_check6.py OUT.json [NAME ...]      exit 0 only if every check passes"""
import os, sys, json, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import guides6 as G, providers as P
OUT = sys.argv[1]; names = sys.argv[2:] or list(G.GUIDES); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] PROGRESS {m}', flush=True)
res = dict(guides={}, failures=[]); dump = lambda: json.dump(res, open(OUT, 'w'), indent=1, default=float)
T = P.orbit_tables(); n = T['n']; BS = T['BS']
sys.path.insert(0, P.J1J2_CODE); import delta0 as D0M
for nm in names:
    rec = {}; res['guides'][nm] = rec
    if os.path.exists(f'{G.GUIDE_DIR}/{nm}.la.npy') and os.path.exists(f'{G.GUIDE_DIR}/{nm}.json'): la = np.load(f'{G.GUIDE_DIR}/{nm}.la.npy'); rec['built'] = 'existing'
    else: la, _ = G.build(nm, log); rec['built'] = 'now'
    if nm == 'T0B0_c47':
        ref = np.load(f'{G.FNI}/la_a.npy'); rec['la_vs_la_a_maxabs'] = float(np.max(np.abs((la - la.max()) - (ref - ref.max()))))   # an additive constant cancels in the normalised amplitude; del ref
        if rec['la_vs_la_a_maxabs'] > 1e-9: res['failures'].append(f'{nm}: la != la_a.npy')
    g = G.load(nm); t = time.time(); ev = g.E_var(); rec.update(E_var=ev, E_var_sec=time.time() - t, n_floor=g.n_floor)
    F, Evs = G.resolve_F(G.GUIDES[nm]); rec.update(F_stored=F, E_var_stored=Evs)
    if Evs is not None:
        rec['dE_var'] = ev - Evs
        if abs(ev - Evs) > 1e-9: res['failures'].append(f'{nm}: E_var {ev} != stored {Evs}')
        log(f'{nm}: E_var {ev:.12f} stored {Evs:.12f} diff {ev - Evs:.1e}')
    else:
        t = time.time(); K, Vm, D0 = BS.build(n, g.sig.astype(np.float64)); r, v = D0M.efn_vec(K, Vm, D0, n, la, tol=1e-9, maxit=150); del K, Vm, D0, v
        r['sec'] = time.time() - t; rec['exact'] = r; rec['dF'] = r['dE_site'] - F; rec['dE_var_vs_solver'] = ev - r['E_guide_var']
        if abs(rec['dF']) > 1e-10 or r['res'] > 1e-9: res['failures'].append(f'{nm}: exact solve F {r["dE_site"]} != stored {F}')
        if abs(rec['dE_var_vs_solver']) > 1e-9: res['failures'].append(f'{nm}: rows E_var != solver E_guide_var')
        log(f'{nm}: solve dE_site {r["dE_site"]:.12e} stored {F:.12e} diff {rec["dF"]:.1e} res {r["res"]:.1e}; E_var rows - solver {rec["dE_var_vs_solver"]:.1e}')
    del la, g; dump()
res['pairs'] = {k: dict(A=a, B=b, delta_exact_site=G.exact_delta(k)) for k, (a, b) in G.PAIRS.items()}
res['passed'] = not res['failures']; res['sec'] = time.time() - T0; dump()
log(f'VERDICT_BUILD6 passed={res["passed"]} failures={res["failures"]}'); sys.exit(0 if res['passed'] else 2)
