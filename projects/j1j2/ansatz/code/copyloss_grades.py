#!/usr/bin/env python3
"""copyloss_grades.py RUN DAT : STORAGE_PROBLEM.md 'WHERE DOES ARM C's COPY LOSE?, REGISTERED' (+ G3b ring addition). Grades only, no training.
Grade = round_map.py's 'start' N (head of the unmodified source exec'd under a sign, then search(lA, 'start'), as stage.py K2 execs the head).
Order: numpy stats + splice tables; S_R1: chi_1, T_1 (known case, stop if rel > 1e-6), G3, G3b, G4_1pct, G4_10pct, G3_broken; then s2: chi_1, T_1."""
import os, sys, json, time, hashlib, gc, traceback
sys.dont_write_bytecode = True
import numpy as np
RUN, DAT = sys.argv[1], sys.argv[2]
WT = '/scratch/a/A.Otaifi/tmp/wt_ratnet/projects/j1j2/ansatz/code'; sys.path.insert(0, WT)
import bond_split as BS
RMAP = f'{WT}/round_map.py'; RMAP_SHA = '30edb2fcc1c1b29b3133ab0244cb48798e2dc52f99ca1639bca493ae8d649ef1'
SC = '/scratch/a/A.Otaifi/ansatz_runs/consol_6x6'
IN = dict(chi1=(f'{SC}/cycle6f_store/fit0/fit/tables/t_R.npy', 'e40d974e2fe46db88e3f921b0f40ea0d'), T1=(f'{SC}/cycle6_store/map0/lA_next.npy', '2e0985d35deef850d8ed7d391b5480b0'),
          chi0=(f'{SC}/ref2hop/NNx/x1/x1fit/tables/t_R.npy', '4db4075d9b02720ca464b4fc3628c15e'), c10=(f'{SC}/cycle6_store/map0/l_part.npy', 'ce35da4855b1604b70f8e3f3af382aea'),
          SR1=(f'{SC}/cycle5_r1/S_R1.npy', 'ecb4b8b5c2f6f08a56ac0c602685387a'), s2=('/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/refresh_17409419/s2.npy', None))
T0 = time.time(); OUT = f'{RUN}/result.json'; res = dict(status='RUNNING', grades={}, stats={}, job=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename)
def log(m): print(f'[{time.time() - T0:7.0f}s] PROGRESS {m}', flush=True)
def md5(p): return hashlib.md5(open(p, 'rb').read()).hexdigest()
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(OUT + '.tmp', 'w'), indent=1, default=float); os.replace(OUT + '.tmp', OUT)
def stop(msg, code):
    res['status'] = msg; dump(); log(f'VERDICT_C6W FAILED {msg}'); sys.stdout.flush(); os._exit(code)
try:
    assert hashlib.sha256(open(RMAP, 'rb').read()).hexdigest() == RMAP_SHA, 'round_map changed'
    for k, (p, h) in IN.items():
        m = md5(p); res.setdefault('md5', {})[k] = m
        if h and m != h: stop(f'md5 mismatch {k}', 4)
    L = {k: np.load(p) for k, (p, h) in IN.items()}
    chi1, T1, chi0, c10 = [L[k].astype(np.float64) for k in ('chi1', 'T1', 'chi0', 'c10')]; S = L['SR1'].astype(np.float64); S2 = L['s2'].astype(np.float64)
    n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert D == len(chi1) == 15804956
    fl = S != S2; assert int(fl.sum()) == 359588, int(fl.sum()); res['n_flipped'] = int(fl.sum())
    w = n * np.exp(2 * (T1 - T1.max())); w /= w.sum()
    d = chi1 - T1; c = float((w * d).sum()); dc = d - c; res['c'] = c
    # neighbour ring: rows joined to a flipped row by a nonzero off-diagonal element (nb_J1/nb_J2 tables as round_map's fJ_and_VsK)
    t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r'); RS = 32768
    out_ring = np.zeros(D, bool); in_ring = np.zeros(D, bool)
    for i0 in range(0, D, RS):
        i1 = min(D, i0 + RS); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); va = t >= 0
        hit_in = (va & fl[np.where(va, t, 0)]).any(1); in_ring[i0:i1] = hit_in            # row j has a neighbour that is flipped
        jj, cc = np.nonzero(va & fl[i0:i1, None]); out_ring[t[jj, cc]] = True                # neighbours of flipped rows
    res['ring_sym'] = dict(out_count=int(out_ring.sum()), in_count=int(in_ring.sum()), equal=bool(np.array_equal(out_ring, in_ring)))
    nbr = out_ring | in_ring; ring = nbr & ~fl; G3b = fl | nbr; del out_ring, in_ring
    def share(m): return float(w[m].sum())
    def rms(x, m): return float(np.sqrt((w[m] * x[m] ** 2).sum() / w[m].sum()))
    r = T1 - chi0 - c10; wr = lambda m: float((w[m] * r[m]).sum() / w[m].sum())
    st = res['stats']; oth = ~fl
    st['counts'] = dict(flipped=int(fl.sum()), G3b=int(G3b.sum()), ring_only=int(ring.sum()))
    st['weight_share'] = dict(flipped=share(fl), G3b=share(G3b), ring_only=share(ring), other=share(oth))
    st['rms_d_minus_c'] = dict(flipped=rms(dc, fl), other=rms(dc, oth), ring_only=rms(dc, ring), outside_G3b=rms(dc, ~G3b), all=rms(dc, np.ones(D, bool)))
    st['rms_r_about0'] = dict(flipped=rms(r, fl), other=rms(r, oth), ring_only=rms(r, ring), outside_G3b=rms(r, ~G3b))
    st['rms_r_about_group_mean'] = {k: float(np.sqrt((w[m] * (r[m] - wr(m)) ** 2).sum() / w[m].sum())) for k, m in dict(flipped=fl, other=oth, ring_only=ring).items()}
    st['mean_r'] = dict(flipped=wr(fl), other=wr(oth), ring_only=wr(ring))
    st['x1_target'] = 'NOT on disk: NNc/map6/{lA_next,p_out}.npy deleted 2026-11-26 (README_DELETED); no other copy of T6 found'
    ad = np.abs(dc); R = {}
    R['G3'] = fl; R['G3b'] = G3b
    for nm, f in (('G4_1pct', 0.01), ('G4_10pct', 0.10)):
        k = int(round(f * D)); idx = np.argpartition(-ad, k)[:k]; m = np.zeros(D, bool); m[idx] = True; R[nm] = m
        st[nm] = dict(count=int(m.sum()), weight_share=share(m), flipped_in=int((m & fl).sum()), ring_in=int((m & ring).sum()), threshold_absdc=float(ad[idx].min()))
    R['G3_broken'] = fl
    dump(); log(f'stats done: {json.dumps(st)[:600]}')
    if os.environ.get('DRY'): os._exit(0)
    # splice tables (aligned: T_1 + c on rows R, chi_1 elsewhere); broken: T_1 unshifted
    SPL = f'{DAT}/splice'; os.makedirs(SPL, exist_ok=True); tabs = {}
    for nm in ('G3', 'G3b', 'G4_1pct', 'G4_10pct'):
        np.save(f'{SPL}/{nm}.npy', np.where(R[nm], T1 + c, chi1)); tabs[nm] = f'{SPL}/{nm}.npy'
    np.save(f'{SPL}/G3_broken.npy', np.where(fl, T1, chi1)); tabs['G3_broken'] = f'{SPL}/G3_broken.npy'
    np.save(f'{SPL}/G3_shift.npy', np.where(fl, T1 + c + 0.05, chi1)); tabs['G3_shift'] = f'{SPL}/G3_shift.npy'
    del R, ad, w, r, d, dc, ring, nbr, G3b, L; gc.collect()
    order = [('SR1', 'chi1', IN['chi1'][0]), ('SR1', 'T1', IN['T1'][0])] + [('SR1', nm, tabs[nm]) for nm in ('G3', 'G3b', 'G4_1pct', 'G4_10pct', 'G3_broken', 'G3_shift')] + [('s2', 'chi1', IN['chi1'][0]), ('s2', 'T1', IN['T1'][0])]
    src = open(RMAP).read(); head = src.split("ak, d, cache = search(lA0, 'start')")[0]; assert head != src; cur = None; G = None
    for sign, name, path in order:
        if sign != cur:
            G = None; gc.collect(); BS.S2 = IN['SR1'][0] if sign == 'SR1' else IN['s2'][0]; cur = sign
            argv = sys.argv; sys.argv = [RMAP, IN['chi1'][0], f'{DAT}/rm_{sign}']
            G = dict(__name__='rm_head', __file__=RMAP); exec(compile(head, RMAP, 'exec'), G); sys.argv = argv
            assert np.array_equal(G['s'], S if sign == 'SR1' else S2); log(f'head built under {sign} md5 {md5(BS.S2)}')
        t = time.time(); lA = np.load(path).astype(np.float64); assert np.isfinite(lA).all() and lA.shape == (D,)
        ak, _, _ = G['search'](lA, f'g_{sign}_{name}'); rec = G['res'][f'g_{sign}_{name}']
        res['grades'][f'{sign}/{name}'] = dict(table=path, md5=md5(path), N=rec['N'], alpha=ak, N_alpha0=rec['N_alpha0'], grid=rec['grid'], sec=time.time() - t); dump()
        log(f'GRADE {sign} {name} N {rec["N"]:.7e} alpha {ak} {time.time() - t:.0f}s')
        if sign == 'SR1' and name in ('chi1', 'T1'):
            ref = 8.201789605995676e-07 if name == 'chi1' else 5.942835229354337e-07;  # cycle6f_store/result.json (registered 8.2018e-07 / 5.9428e-07)
            rel = abs(rec['N'] / ref - 1); res['grades'][f'{sign}/{name}']['known_rel'] = rel
            if rel > 1e-6: stop(f'KNOWN CASE FAILED {name} N {rec["N"]:.7e} vs {ref:.4e} rel {rel:.2e}', 2)
    g = {k: v['N'] for k, v in res['grades'].items()}; a, b = g['SR1/chi1'], g['SR1/T1']
    res['L_0_SR1'] = a - b; res['L_s2'] = g['s2/chi1'] - g['s2/T1']
    res['L0_share'] = {nm: (a - g[f'SR1/{nm}']) / (a - b) for nm in ('G3', 'G3b', 'G4_1pct', 'G4_10pct', 'G3_broken', 'G3_shift')}
    res['broken_diff_vs_G3'] = dict(G3_broken=g['SR1/G3_broken'] - g['SR1/G3'], G3_shift=g['SR1/G3_shift'] - g['SR1/G3'])
    res['status'] = 'DONE'; dump(); log('VERDICT_C6W DONE ' + json.dumps(res['L0_share']))
except SystemExit: raise
except Exception:
    res['error'] = traceback.format_exc(); print(res['error'], flush=True); stop('ERROR', 3)
sys.stdout.flush(); os._exit(0)
