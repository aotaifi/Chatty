#!/usr/bin/env python3
"""copyloss_lam_grades.py RUN DAT : STORAGE_PROBLEM.md 'FIRST OR SECOND ORDER?, REGISTERED'. Grades only.
Tables T_1 + lam*(chi_1 - T_1 - c), lam = 0.5, -0.5, -1, c recomputed (weight n exp(2 T_1) mean of chi_1 - T_1, must match 1.0142e-5 to 1e-3 rel).
Grade = round_map.py 'start' N under S_R1 (as copyloss_grades.py). Order: chi_1 (known 8.2017896e-07), T_1 (known 5.9428352e-07), then the lam tables."""
import os, sys, json, time, hashlib, gc, traceback
sys.dont_write_bytecode = True
import numpy as np
RUN, DAT = sys.argv[1], sys.argv[2]
WT = '/scratch/a/A.Otaifi/tmp/wt_ratnet/projects/j1j2/ansatz/code'; sys.path.insert(0, WT)
import bond_split as BS
RMAP = f'{WT}/round_map.py'; RMAP_SHA = '30edb2fcc1c1b29b3133ab0244cb48798e2dc52f99ca1639bca493ae8d649ef1'
SC = '/scratch/a/A.Otaifi/ansatz_runs/consol_6x6'
CHI1 = (f'{SC}/cycle6f_store/fit0/fit/tables/t_R.npy', 'e40d974e2fe46db88e3f921b0f40ea0d'); T1 = (f'{SC}/cycle6_store/map0/lA_next.npy', '2e0985d35deef850d8ed7d391b5480b0')
SR1 = (f'{SC}/cycle5_r1/S_R1.npy', 'ecb4b8b5c2f6f08a56ac0c602685387a')
LAMS = (0.5, -0.5, -1.0); C_REG = 1.0142e-5; L_REF = 2.259e-7
T0 = time.time(); OUT = f'{RUN}/result.json'; res = dict(status='RUNNING', grades={}, job=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename)
def log(m): print(f'[{time.time() - T0:7.0f}s] PROGRESS {m}', flush=True)
def md5(p): return hashlib.md5(open(p, 'rb').read()).hexdigest()
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(OUT + '.tmp', 'w'), indent=1, default=float); os.replace(OUT + '.tmp', OUT)
def stop(msg, code): res['status'] = msg; dump(); log(f'VERDICT_C6L FAILED {msg}'); sys.stdout.flush(); os._exit(code)
try:
    assert hashlib.sha256(open(RMAP, 'rb').read()).hexdigest() == RMAP_SHA, 'round_map changed'
    for nm, (p, h) in (('SR1', SR1), ('chi1', CHI1), ('T1', T1)):
        m = md5(p); res.setdefault('md5', {})[nm] = m
        if m != h: stop(f'md5 mismatch {nm}', 4)
    chi1 = np.load(CHI1[0]).astype(np.float64); t1 = np.load(T1[0]).astype(np.float64)
    n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert D == len(chi1) == len(t1)
    w = n * np.exp(2 * (t1 - t1.max())); w /= w.sum(); d = chi1 - t1; c = float((w * d).sum()); res['c'] = c
    if abs(c / C_REG - 1) > 1e-3: stop(f'c recomputed {c:.5e} vs registered {C_REG:.4e}', 6)
    os.makedirs(f'{DAT}/tables', exist_ok=True); tabs = {}
    for lam in LAMS:
        p = f'{DAT}/tables/lam_{lam:+.1f}.npy'; np.save(p, t1 + lam * (d - c)); tabs[lam] = p
    del chi1, t1, n, w, d; gc.collect(); log(f'c = {c:.6e}; tables written')
    order = [('chi1', CHI1[0]), ('T1', T1[0])] + [(f'lam{lam:+.1f}', tabs[lam]) for lam in LAMS]
    src = open(RMAP).read(); head = src.split("ak, d, cache = search(lA0, 'start')")[0]; assert head != src
    BS.S2 = SR1[0]; argv = sys.argv; sys.argv = [RMAP, CHI1[0], f'{DAT}/rm_SR1']
    G = dict(__name__='rm_head', __file__=RMAP); exec(compile(head, RMAP, 'exec'), G); sys.argv = argv
    assert np.array_equal(G['s'], np.load(SR1[0]).astype(np.float64)); log(f'head built under SR1 md5 {md5(BS.S2)}')
    known = dict(chi1=8.201789605995676e-07, T1=5.942835229354337e-07)
    for name, path in order:
        t = time.time(); lA = np.load(path).astype(np.float64); assert np.isfinite(lA).all() and lA.shape == (D,)
        ak, _, _ = G['search'](lA, f'g_SR1_{name}'); rec = G['res'][f'g_SR1_{name}']
        res['grades'][name] = dict(table=path, md5=md5(path), N=rec['N'], alpha=ak, N_alpha0=rec['N_alpha0'], grid=rec['grid'], sec=time.time() - t); dump()
        log(f'GRADE SR1 {name} N {rec["N"]:.7e} alpha {ak} {time.time() - t:.0f}s')
        if name in known:
            rel = abs(rec['N'] / known[name] - 1); res['grades'][name]['known_rel'] = rel
            if rel > 1e-6: stop(f'KNOWN CASE FAILED {name} N {rec["N"]:.7e} rel {rel:.2e}', 2)
    g = {k: v['N'] for k, v in res['grades'].items()}; N0 = g['T1']
    lam = np.array([-1, -0.5, 0, 0.5, 1.0]); Nl = np.array([g['lam-1.0'], g['lam-0.5'], g['T1'], g['lam+0.5'], g['chi1']]); y = Nl - N0
    a, b = np.linalg.lstsq(np.stack([lam, lam ** 2], 1), y, rcond=None)[0]; L = g['chi1'] - N0
    # broken: quadratic through lam 0, 0.5, 1 predicts lam = -1: y = a3 lam + b3 lam^2 through (0.5, y.5), (1, y1)
    a3, b3 = np.linalg.solve([[0.5, 0.25], [1, 1]], [y[3], y[4]]); pred = -a3 + b3
    res['fit'] = dict(lam=lam.tolist(), N=Nl.tolist(), a=a, b=b, L=L, L_reg=L_REF, a_over_L=a / L, a_over_Lreg=a / L_REF, a_over_L_note='L = N(chi_1) - N(T_1) from this run',
                      broken=dict(a3=a3, b3=b3, predicted_lam_m1=pred + N0, measured_lam_m1=Nl[0], diff=Nl[0] - pred - N0, diff_over_L=(Nl[0] - pred - N0) / L))
    res['status'] = 'DONE'; dump(); log('VERDICT_C6L DONE ' + json.dumps(res['fit']))
except SystemExit: raise
except Exception:
    res['error'] = traceback.format_exc(); print(res['error'], flush=True); stop('ERROR', 3)
sys.stdout.flush(); os._exit(0)
