#!/usr/bin/env python3
"""test_cnn_sign.py OUT.json [--only ctl,rows,labels,time] [--n 1000] [--M 64] : CPU controls of cnn_sign (StoredCNNSign, ExactStepSign).  Run with JAX_PLATFORMS=cpu G8_ALLOW_CPU=1.
 ctl    A = B in the coupled engine: StoredCNNSign(n_3) vs itself: Delta == 0 exactly, same_post 1; broken: A vs shuffled-weight n_3: Delta != 0.
 rows   StoredCNNSign rows == rows of the T-BAD FN code (sign_opts.SignOpt 'net', as r1_grade.build) on n configurations: sign sx, sy, rate, dFN (E_L = dFN - sum rate), 1e-10; broken: shuffled weights.
 labels ExactStepSign(n_2) == stored labels K[n_2] (r3/labels/table_r3.npz: y relative to Marshall, EL2 = E_L(|B0| n_2)) on n states, 0 mismatches; broken: alpha_flip, shuffled net.
 time   sec/block of each provider alone (single mode, M states) on CPU, cold and warm, with evaluation counts."""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--only', default='ctl,rows,labels,time'); ap.add_argument('--n', type=int, default=1000)
ap.add_argument('--M', type=int, default=64); ap.add_argument('--ctl_blocks', type=int, default=10); ap.add_argument('--time_blocks', type=int, default=4); ap.add_argument('--seed', type=int, default=5)
a = ap.parse_args(); only = set(a.only.split(','))
import cnn_sign as CS, providers as PV, engine as E
R = '/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad/'; X0F = '/scratch/a/A.Otaifi/ansatz_runs/sign_node/inputs/X0_B0_3.npy'
res = json.load(open(a.out)) if os.path.exists(a.out) else {}
def dump(): json.dump(res, open(a.out, 'w'), indent=1, default=float)
amp = CS.AmpCache(); n3 = CS.NetSign(R + 'r3/fit/net_8192.pkl', 'n3'); n2 = CS.NetSign(R + 'r2/fit/net_8192.pkl', 'n2')
res['nets'] = dict(n2=(n2.path, n2.sha), n3=(n3.path, n3.sha)); res['alpha_E'] = (CS.ALPHA, CS.E_REF)
X0 = np.load(X0F).astype(np.int8)[:a.M]
if 'ctl' in only:
    t0 = time.time(); g1, _ = CS.build('stored', amp, n3); g2, _ = CS.build('stored', amp, n3)
    P1 = PV.ConfigGuide(g1, guide_id='n3'); P2 = PV.ConfigGuide(g2, guide_id='n3')
    o = E.run(P1, P2, X0, X0, mode='coupled', dtau=0.1, measure_blocks=a.ctl_blocks, pair_seed=a.seed, NS=64)
    s = o['series']; c = dict(blocks=len(s['delta']), max_abs_delta=float(np.max(np.abs(s['delta']))), same_post_min=float(np.min(s['same_post'])),
        A_only=float(np.nansum(s['A_only'])), B_only=float(np.nansum(s['B_only'])), shared=float(np.nansum(s['shared'])), sec=time.time() - t0)
    c['PASS'] = bool(c['max_abs_delta'] == 0 and c['same_post_min'] == 1 and c['A_only'] + c['B_only'] == 0 and c['shared'] > 0); res['ctl_A_eq_B'] = c; dump(); print('CTL', c, flush=True)
    shuf = CS.shuffle_net(n3, 1); g3, _ = CS.build('stored', amp, shuf); g4, _ = CS.build('stored', amp, n3)
    o = E.run(PV.ConfigGuide(g4, guide_id='n3'), PV.ConfigGuide(g3, guide_id='shuf'), X0, X0, mode='coupled', dtau=0.1, measure_blocks=3, pair_seed=a.seed, NS=64)
    s = o['series']; b = dict(max_abs_delta=float(np.max(np.abs(s['delta']))), same_post_min=float(np.min(s['same_post'])), A_only=float(np.nansum(s['A_only'])), B_only=float(np.nansum(s['B_only'])))
    b['MISSED_BY_CONTROL'] = bool(b['max_abs_delta'] == 0 and b['same_post_min'] == 1); res['ctl_broken_shuffled'] = b; dump(); print('CTL broken', b, flush=True)
rg = np.random.default_rng(a.seed); T = np.load(R + 'r3/labels/table_r3.npz'); idx = np.sort(rg.choice(len(T['keys']), a.n, replace=False)); keys = T['keys'][idx]
if 'rows' in only:
    import grade_bare, r1_grade as RG, gfmc_aug8 as G8
    t0 = time.time(); ref, _ = RG.build('net', 8, 0.0, netpath=n3.path); X = CS.spins_of(keys).astype(np.int8)
    gs, _ = CS.build('stored', amp, n3, wchunk=8); shuf = CS.shuffle_net(n3, 2); gb, _ = CS.build('stored', amp, shuf, wchunk=8)
    def run(g):
        out = []; sx = []; sy = []
        for i in range(0, len(X), 100):
            r, d = g.rows(X[i:i + 100]); out.append((np.asarray(r), np.asarray(d))); sx.append(g.last_sx); sy.append(g.last_sy)
        return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]), np.concatenate(sx), np.concatenate(sy)
    rr, dr, sxr, syr = run(ref); rs, ds, sxs, sys_ = run(gs); rb, db, sxb, syb = run(gb)
    c = dict(n=int(len(X)), sx_mismatch=int((sxr != sxs).sum()), sy_mismatch=int((syr != sys_).sum()), max_abs_rate=float(np.max(np.abs(rr - rs))), max_abs_dFN=float(np.max(np.abs(dr - ds))),
             max_abs_EL=float(np.max(np.abs((dr - rr.sum(1)) - (ds - rs.sum(1))))), n_sy=int(len(syr)))
    c['PASS'] = bool(c['sx_mismatch'] == 0 and c['sy_mismatch'] == 0 and max(c['max_abs_rate'], c['max_abs_dFN'], c['max_abs_EL']) < 1e-10)
    c['broken_shuffled'] = dict(sx_mismatch=int((sxr != sxb).sum()), max_abs_dFN=float(np.max(np.abs(dr - db))), max_abs_EL=float(np.max(np.abs((dr - rr.sum(1)) - (db - rb.sum(1))))))
    c['sec'] = time.time() - t0; res['rows_vs_TBAD_code'] = c; dump(); print('ROWS', c, flush=True)
if 'labels' in only:
    t0 = time.time(); M = CS.marshall_pm(keys); Ks = (M * T['y'][idx]).astype(np.int8)
    c = dict(n=int(len(keys)), true_flips=int((T['y'][idx] != n2.sign(keys) * M).sum()))
    ge, _ = CS.build('exact', amp, n2); K, EL, nz = ge.K_of(keys)
    c['mismatch'] = int((K != Ks).sum()); c['max_abs_dEL_vs_table'] = float(np.max(np.abs(EL - T['EL2'][idx]))); c['n_flipped_vs_n2'] = int((K != nz).sum())
    gw, _ = CS.build('exact', amp, n2, broken='alpha_flip'); Kw, _, _ = gw.K_of(keys); c['broken_alpha_flip_mismatch'] = int((Kw != Ks).sum())
    gs, _ = CS.build('exact', amp, CS.shuffle_net(n2, 3)); Kq, _, _ = gs.K_of(keys); c['broken_shuffled_mismatch'] = int((Kq != Ks).sum())
    # table tolerance 1e-8 (lead 2026-11-15): measured 5.1e-9 on |E_L| ~ 30 is ~2e-10 relative, harmless for labels whose threshold margin is ~J; the rows test vs the FN code stays at 1e-10
    c['PASS'] = bool(c['mismatch'] == 0 and c['max_abs_dEL_vs_table'] < 1e-8 and c['broken_alpha_flip_mismatch'] > 0 and c['broken_shuffled_mismatch'] > 0)
    c['sec'] = time.time() - t0; res['labels_vs_K_n2'] = c; dump(); print('LABELS', c, flush=True)
if 'time' in only:
    out = {}
    for mode in ('stored', 'exact'):
        amp.d.clear(); amp.reset_counts(); n3.d.clear(); n3.n_eval = 0; n3.sec = 0.0; g, _ = CS.build(mode, amp, n3); ts = []
        t_start = time.time()
        o = E.run(PV.ConfigGuide(g, guide_id=mode), None, X0, mode='single', dtau=0.1, measure_blocks=a.time_blocks, pair_seed=1, NS=64, progress_every=1, log=lambda m: ts.append(time.time()))
        t = np.diff(np.array([t_start] + ts)); out[mode] = dict(M=a.M, block_sec=t.tolist(), amp_miss=amp.miss, amp_hit=amp.hit, amp_sec=amp.sec, net_eval=n3.n_eval, net_sec=n3.sec,
            rows_calls=g.n_calls, rows=g.n_rows, st={k: float(v) for k, v in g.st.items()})
        print('TIME', mode, out[mode], flush=True); res['timing_cpu'] = out; dump()
