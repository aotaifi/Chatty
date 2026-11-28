#!/usr/bin/env python3
"""copyloss_c2_grades.py RUN DAT : STORAGE_PROBLEM.md 'C2 AS A 13TH INPUT, REGISTERED'. Grades only.
Grade = round_map.py 'start' N (head of the unmodified source exec'd under a sign, then search(lA,'start')), as copyloss_grades.py.
Order: S_R1 chi_1 (known case 8.2017896e-07, stop if rel > 1e-6), then S_R1 C2.
L_0 = N - 5.9428352e-07 (N(T_1) under S_R1), L_s2 = N - 9.2773858e-07 (N(T_1) under s2), both from the registered numbers."""
import os, sys, json, time, hashlib, gc, traceback
sys.dont_write_bytecode = True
import numpy as np
RUN, DAT = sys.argv[1], sys.argv[2]
WT = '/scratch/a/A.Otaifi/tmp/wt_ratnet/projects/j1j2/ansatz/code'; sys.path.insert(0, WT)
import bond_split as BS
RMAP = f'{WT}/round_map.py'; RMAP_SHA = '30edb2fcc1c1b29b3133ab0244cb48798e2dc52f99ca1639bca493ae8d649ef1'
SC = '/scratch/a/A.Otaifi/ansatz_runs/consol_6x6'
SR1 = (f'{SC}/cycle5_r1/S_R1.npy', 'ecb4b8b5c2f6f08a56ac0c602685387a'); S2 = ('/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/refresh_17409419/s2.npy', None)
CHI1 = (f'{SC}/cycle6f_store/fit0/fit/tables/t_R.npy', 'e40d974e2fe46db88e3f921b0f40ea0d')
NT1_SR1, NT1_S2 = 5.9428352e-07, 9.2773858e-07
T = {'C2': f'{SC}/c6_c2/C2/fit/tables/t_R.npy'}
T0 = time.time(); OUT = f'{RUN}/result.json'; res = dict(status='RUNNING', grades={}, job=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename)
def log(m): print(f'[{time.time() - T0:7.0f}s] PROGRESS {m}', flush=True)
def md5(p): return hashlib.md5(open(p, 'rb').read()).hexdigest()
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(OUT + '.tmp', 'w'), indent=1, default=float); os.replace(OUT + '.tmp', OUT)
def stop(msg, code): res['status'] = msg; dump(); log(f'VERDICT_C6C2G FAILED {msg}'); sys.stdout.flush(); os._exit(code)
try:
    assert hashlib.sha256(open(RMAP, 'rb').read()).hexdigest() == RMAP_SHA, 'round_map changed'
    for nm, (p, h) in (('SR1', SR1), ('chi1', CHI1)):
        m = md5(p); res.setdefault('md5', {})[nm] = m
        if h and m != h: stop(f'md5 mismatch {nm}', 4)
    for nm, p in T.items():
        if not os.path.isfile(p): stop(f'missing table {nm}: {p}', 5)
        res['md5'][nm] = md5(p)
    assert res['md5']['C2'] != CHI1[1]
    S = np.load(SR1[0]).astype(np.float64); S2a = np.load(S2[0]).astype(np.float64)
    n = np.load(f'{BS.CSRD}/norb.npy'); D = len(n); del n
    order = [('SR1', 'chi1', CHI1[0]), ('SR1', 'C2', T['C2'])]
    src = open(RMAP).read(); head = src.split("ak, d, cache = search(lA0, 'start')")[0]; assert head != src; cur = None; G = None
    for sign, name, path in order:
        if sign != cur:
            G = None; gc.collect(); BS.S2 = SR1[0] if sign == 'SR1' else S2[0]; cur = sign
            argv = sys.argv; sys.argv = [RMAP, CHI1[0], f'{DAT}/rm_{sign}']
            G = dict(__name__='rm_head', __file__=RMAP); exec(compile(head, RMAP, 'exec'), G); sys.argv = argv
            assert np.array_equal(G['s'], S if sign == 'SR1' else S2a); log(f'head built under {sign} md5 {md5(BS.S2)}')
        t = time.time(); lA = np.load(path).astype(np.float64); assert np.isfinite(lA).all() and lA.shape == (D,)
        ak, _, _ = G['search'](lA, f'g_{sign}_{name}'); rec = G['res'][f'g_{sign}_{name}']
        ref = {'SR1': NT1_SR1, 's2': NT1_S2}[sign]
        res['grades'][f'{sign}/{name}'] = dict(table=path, md5=md5(path), N=rec['N'], alpha=ak, N_alpha0=rec['N_alpha0'], grid=rec['grid'], sec=time.time() - t)
        if name != 'chi1': res['grades'][f'{sign}/{name}']['L'] = rec['N'] - ref
        dump(); log(f'GRADE {sign} {name} N {rec["N"]:.7e} alpha {ak} {time.time() - t:.0f}s')
        if sign == 'SR1' and name == 'chi1':
            rel = abs(rec['N'] / 8.201789605995676e-07 - 1); res['grades']['SR1/chi1']['known_rel'] = rel
            if rel > 1e-6: stop(f'KNOWN CASE FAILED chi_1 N {rec["N"]:.7e} rel {rel:.2e}', 2)
    res['status'] = 'DONE'; dump(); log('VERDICT_C6C2G DONE ' + json.dumps({k: v.get('L') for k, v in res['grades'].items()}))
except SystemExit: raise
except Exception:
    res['error'] = traceback.format_exc(); print(res['error'], flush=True); stop('ERROR', 3)
sys.stdout.flush(); os._exit(0)
