#!/usr/bin/env python3
"""copyloss_gamma_grades.py RUN DAT : coordinator addition to c6_lam. (1) numpy: cos(d, s), beta = <d,s>/<s,s>, d = chi_1 - T_1, s = T_1 - chi_0, each with its own weighted mean removed,
weight (a) n exp(2 T_1); (b) psi0^2 not available (no exact ground-state amplitude found where round_map/bond_split look: only csr/meta.json E0), so (a) only.
(2) grades under S_R1 (copyloss_grades.py route) of ln chi(gamma) = chi_0 + gamma (chi_1 - chi_0), gamma = 1 (known case 8.2017896e-07, chi_1 itself), 1.5, 2, 3."""
import os, sys, json, time, hashlib, gc, traceback
sys.dont_write_bytecode = True
import numpy as np
RUN, DAT = sys.argv[1], sys.argv[2]
WT = '/scratch/a/A.Otaifi/tmp/wt_ratnet/projects/j1j2/ansatz/code'; sys.path.insert(0, WT)
import bond_split as BS
RMAP = f'{WT}/round_map.py'; RMAP_SHA = '30edb2fcc1c1b29b3133ab0244cb48798e2dc52f99ca1639bca493ae8d649ef1'
SC = '/scratch/a/A.Otaifi/ansatz_runs/consol_6x6'
CHI1 = (f'{SC}/cycle6f_store/fit0/fit/tables/t_R.npy', 'e40d974e2fe46db88e3f921b0f40ea0d'); T1 = (f'{SC}/cycle6_store/map0/lA_next.npy', '2e0985d35deef850d8ed7d391b5480b0')
CHI0 = (f'{SC}/ref2hop/NNx/x1/x1fit/tables/t_R.npy', '4db4075d9b02720ca464b4fc3628c15e'); SR1 = (f'{SC}/cycle5_r1/S_R1.npy', 'ecb4b8b5c2f6f08a56ac0c602685387a')
GAMMAS = (1.5, 2.0, 3.0)
T0 = time.time(); OUT = f'{RUN}/result_gamma.json'; res = dict(status='RUNNING', grades={}, job=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename)
def log(m): print(f'[{time.time() - T0:7.0f}s] PROGRESS {m}', flush=True)
def md5(p): return hashlib.md5(open(p, 'rb').read()).hexdigest()
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(OUT + '.tmp', 'w'), indent=1, default=float); os.replace(OUT + '.tmp', OUT)
def stop(msg, code): res['status'] = msg; dump(); log(f'VERDICT_C6G FAILED {msg}'); sys.stdout.flush(); os._exit(code)
try:
    assert hashlib.sha256(open(RMAP, 'rb').read()).hexdigest() == RMAP_SHA, 'round_map changed'
    for nm, (p, h) in (('SR1', SR1), ('chi1', CHI1), ('T1', T1), ('chi0', CHI0)):
        m = md5(p); res.setdefault('md5', {})[nm] = m
        if m != h: stop(f'md5 mismatch {nm}', 4)
    chi1 = np.load(CHI1[0]).astype(np.float64); t1 = np.load(T1[0]).astype(np.float64); chi0 = np.load(CHI0[0]).astype(np.float64)
    n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert D == len(chi1) == len(t1) == len(chi0)
    w = n * np.exp(2 * (t1 - t1.max())); w /= w.sum()
    d = chi1 - t1; s = t1 - chi0; d = d - (w * d).sum(); s = s - (w * s).sum()
    ds, dd, ss = [float((w * x).sum()) for x in (d * s, d * d, s * s)]
    res['cos_a'] = dict(weight='n exp(2 T_1)', cos=ds / np.sqrt(dd * ss), beta=ds / ss, rms_d=np.sqrt(dd), rms_s=np.sqrt(ss))
    res['cos_b'] = 'psi0^2 weight: no exact ground-state amplitude on disk where round_map/bond_split look (csr/meta.json has E0 only); not computed'
    log(f'cos_a {json.dumps(res["cos_a"])}'); dump()
    os.makedirs(f'{DAT}/tables', exist_ok=True); tabs = {}
    for g in GAMMAS:
        p = f'{DAT}/tables/gamma_{g:.1f}.npy'; np.save(p, chi0 + g * (chi1 - chi0)); tabs[g] = p
    del chi1, t1, chi0, n, w, d, s; gc.collect()
    order = [('gamma1.0', CHI1[0])] + [(f'gamma{g:.1f}', tabs[g]) for g in GAMMAS]
    src = open(RMAP).read(); head = src.split("ak, d, cache = search(lA0, 'start')")[0]; assert head != src
    BS.S2 = SR1[0]; argv = sys.argv; sys.argv = [RMAP, CHI1[0], f'{DAT}/rm_SR1']
    G = dict(__name__='rm_head', __file__=RMAP); exec(compile(head, RMAP, 'exec'), G); sys.argv = argv
    assert np.array_equal(G['s'], np.load(SR1[0]).astype(np.float64)); log('head built under SR1')
    for name, path in order:
        t = time.time(); lA = np.load(path).astype(np.float64); assert np.isfinite(lA).all()
        ak, _, _ = G['search'](lA, f'g_SR1_{name}'); rec = G['res'][f'g_SR1_{name}']
        res['grades'][name] = dict(table=path, md5=md5(path), N=rec['N'], alpha=ak, N_alpha0=rec['N_alpha0'], grid=rec['grid'], sec=time.time() - t); dump()
        log(f'GRADE SR1 {name} N {rec["N"]:.7e} alpha {ak} {time.time() - t:.0f}s')
        if name == 'gamma1.0':
            rel = abs(rec['N'] / 8.201789605995676e-07 - 1); res['grades'][name]['known_rel'] = rel
            if rel > 1e-6: stop(f'KNOWN CASE FAILED chi_1 N {rec["N"]:.7e} rel {rel:.2e}', 2)
    res['status'] = 'DONE'; dump(); log('VERDICT_C6G DONE ' + json.dumps({k: v['N'] for k, v in res['grades'].items()}))
except SystemExit: raise
except Exception:
    res['error'] = traceback.format_exc(); print(res['error'], flush=True); stop('ERROR', 3)
sys.stdout.flush(); os._exit(0)
