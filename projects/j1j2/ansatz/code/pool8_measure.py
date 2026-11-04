"""pool8_measure.py: test (b)/(c) readout of pool8.py runs at 8x8 (or any size).
  python pool8_measure.py OUT.json P_REAL F1.npz F2.npz ...     (files from `pool8.py run`; all must share betas / mode / sweep_len; every ensemble is an independent chain)
Per rung: tau_int (Sokal, ACFs averaged over the independent ensembles) of ln|B|^2, of the decade index d = clip(floor((2 lnB - med)/ln10)) and of every decade indicator,
in SWEEPS of N proposals (informed runs with sweep_len < N are converted);  PT swap and move acceptance;  per-decade occupancy (relative decades: med = median of 2 lnB at the
beta = 1 rung, as pool_fit.pool_weights);  CPU s per sweep per walker (one core);  thin = ceil(2 max(tau_lnB2, tau_dec)).
DERIVED sizing (labelled): for S draws split in equal parts over the output rungs, every decade d in +2..-6 has ESS_d(S) = 0.8 * sum_q (S/nq) f_{d,q} eps_{d,q}   (0.8 = training share after
the 20% held-out hash; eps = 1/(1 + CV^2 of the in-cell weights exp(-2 beta lnB), from the sampled series); f_{d,q} = measured occupancy), thinned draws taken as independent.
S_needed(d) = ratio * P / ESS_d(1) ;  the optimum allocation over rungs is the LP  min sum S_q  s.t. 0.8 sum_q S_q f eps >= ratio P  for every decade.
Core-h = (max_q S_q) * thin * R * s_per_sweep_per_walker / 3600  (every rung of the ladder runs, one draw per rung per thin sweeps)."""
import sys, os, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pool8 as P8
LN10 = math.log(10.0); DECS = list(range(2, -7, -1))


def main(outp, P_real, files, ratio=3.0, out_betas=(1.0, 0.5, 0.25)):
    out_betas = tuple(float(b) for b in os.environ.get('OUT_BETAS', ','.join(map(str, out_betas))).split(','))
    Z = [np.load(f) for f in files]; betas = [float(b) for b in Z[0]['betas']]; N = 64 if int(Z[0]['problem']) == 8 else 36
    sl = int(Z[0]['sweep_len']) if 'sweep_len' in Z[0].files else N; conv = sl / N                     # one recorded 'sweep' = sl steps = conv sweeps of N proposals
    ser = np.concatenate([z['lnb_series'].astype(np.float64) for z in Z], axis=2); T, R, Et = ser.shape
    q1 = betas.index(1.0); x = 2.0 * ser; med = float(np.median(x[:, q1, :])); dec = np.floor((x - med) / LN10)
    res = dict(files=files, T_recorded=int(T), sweep_len=sl, sweeps_N_per_record=conv, E_tot=int(Et), betas=betas, med2lnB_beta1=med, mode=str(Z[0]['mode']), P_real=P_real, rungs=[])
    s_w = float(np.mean([float(z['s_per_sweep_per_walker']) for z in Z])) / conv                   # s per sweep-of-N-proposals per walker
    res['s_per_sweep_per_walker_core'] = s_w
    f_tab = np.zeros((R, len(DECS))); eps_tab = np.zeros((R, len(DECS)))
    for q, b in enumerate(betas):
        xq = x[:, q, :]; dq = np.clip(dec[:, q, :], DECS[-1] - 1, DECS[0] + 1)
        t1, M1 = P8.tau_int(xq.T); t2, M2 = P8.tau_int(np.clip(dec[:, q, :], -8, 4).T)
        occ = {}; taus = {}
        for g, d in enumerate(DECS):
            ind = (dec[:, q, :] == d).astype(np.float64); fm = ind.mean(0); f = fm.mean()
            tau_i = P8.tau_int(ind.T)[0] if ind.var() > 0 else float('nan'); taus[str(d)] = tau_i
            occ[str(d)] = dict(frac=float(f), se=float(fm.std(ddof=1) / math.sqrt(Et)) if Et > 1 else float('nan'), n=int(ind.sum()), tau=tau_i * conv)
            f_tab[q, g] = f; m = ind.astype(bool)
            if m.any():
                w = np.exp(-2.0 * b * (ser[:, q, :][m] - ser[:, q, :][m].max())); eps_tab[q, g] = float(w.sum() ** 2 / (w.size * (w ** 2).sum()))
        below = float((dec[:, q, :] < DECS[-1]).mean()); above = float((dec[:, q, :] > DECS[0]).mean())
        res['rungs'].append(dict(beta=b, tau_lnB2=t1 * conv, tau_dec=t2 * conv, window_lnB2=M1, window_dec=M2, tau_max_indicator=float(np.nanmax(list(taus.values()))) * conv,
                                 mean2lnB=float(xq.mean()), std2lnB=float(xq.std()), above=above, below=below, occupancy=occ,
                                 acc_move=float(np.mean([z['acc'][q] for z in Z])), cpu_s_per_sweep_walker=float(np.mean([float(z['s_per_sweep_per_walker']) for z in Z])) / conv))
    res['swap_acc'] = [float(np.mean([z['swap_acc'][i] for z in Z])) for i in range(R - 1)]
    tm = max(max(r['tau_lnB2'], r['tau_dec']) for r in res['rungs']); res['tau_slowest_lnB2_dec'] = tm; res['thin_sweeps'] = int(math.ceil(2.0 * tm))
    res['tau_slowest_with_indicators'] = max(r['tau_max_indicator'] for r in res['rungs'])
    # ---- sizing (derived)
    qs = [betas.index(b) for b in out_betas if b in betas]; nq = len(qs); need = ratio * P_real
    perS = np.array([0.8 * sum(f_tab[q, g] * eps_tab[q, g] for q in qs) / nq for g in range(len(DECS))])      # ESS_d per unit S, equal split
    S_d = np.where(perS > 0, need / np.maximum(perS, 1e-300), np.inf)
    res['sizing'] = dict(note='DERIVED from the measured occupancy; thinned draws taken as independent', ratio=ratio, out_betas=list(out_betas),
                         S_needed_per_decade_equal_split={str(d): float(s) for d, s in zip(DECS, S_d)}, S_needed_equal_split=float(np.max(S_d)),
                         limiting_decade=int(DECS[int(np.argmax(S_d))]), frac_eps={str(betas[q]): {str(d): [float(f_tab[q, g]), float(eps_tab[q, g])] for g, d in enumerate(DECS)} for q in range(R)},
                         tau_per_rung={str(betas[q]): max(res['rungs'][q]['tau_lnB2'], res['rungs'][q]['tau_dec']) for q in range(R)})
    try:
        from scipy.optimize import linprog
        A = -0.8 * np.array([[f_tab[q, g] * eps_tab[q, g] for q in qs] for g in range(len(DECS))]); keep = [g for g in range(len(DECS)) if np.any(A[g] < 0)]
        lp = linprog(np.ones(nq), A_ub=A[keep], b_ub=-need * np.ones(len(keep)), bounds=[(0, None)] * nq)
        if lp.success:
            Sq = lp.x; res['sizing']['lp_alloc'] = {str(betas[q]): float(s) for q, s in zip(qs, Sq)}; res['sizing']['S_needed_lp_total'] = float(Sq.sum())
            res['sizing']['decades_unreachable'] = [int(DECS[g]) for g in range(len(DECS)) if g not in keep]
            res['sizing']['core_h_lp'] = float(Sq.max() * res['thin_sweeps'] * R * s_w / 3600)
    except Exception as e:
        res['sizing']['lp_error'] = repr(e)
    res['sizing']['core_h_equal_split'] = float(res['sizing']['S_needed_equal_split'] / nq * res['thin_sweeps'] * R * s_w / 3600)
    json.dump(res, open(outp, 'w'), indent=1, default=float)
    print(f"mode {res['mode']} chains {Et} records {T} (sweep_len {sl}); med 2lnB(beta1) {med:.2f}; swap acc {np.round(res['swap_acc'], 3).tolist()}; thin {res['thin_sweeps']} sweeps; "
          f"cpu s/sweep/walker {s_w:.3f}")
    for r in res['rungs']:
        print(f"  beta {r['beta']:.3f}: acc {r['acc_move']:.3f} tau_lnB2 {r['tau_lnB2']:.1f} tau_dec {r['tau_dec']:.1f} tau_ind_max {r['tau_max_indicator']:.1f}  mean/std 2lnB {r['mean2lnB']:.2f}/{r['std2lnB']:.2f}  above {r['above']:.2e} below {r['below']:.2e}  occ " +
              ' '.join(f"{r['occupancy'][str(d)]['frac']:.1e}" for d in DECS))
    print('  sizing (derived):', {k: v for k, v in res['sizing'].items() if k in ('S_needed_equal_split', 'limiting_decade', 'S_needed_lp_total', 'core_h_equal_split', 'core_h_lp', 'decades_unreachable')})
    return res


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3:])
