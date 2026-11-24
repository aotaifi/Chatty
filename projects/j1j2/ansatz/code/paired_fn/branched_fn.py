"""branched_fn.py (paired_fn, NEW 2026-11-24): branched short-pair mode for the paired fixed-node energy difference.

Design (lead, PLAN_sign.md 'grade4 COUPLING vs TIME' + LEAD decision; fixed, nothing adaptive):
  1. ONE population is equilibrated under guide A (engine mode 'single', burn_A blocks) and kept running.
  2. K branch points at spaced times t_k = burn_A + k * spacing: snapshot S_k = the A population after block t_k (post-resampling, equal weights).
  3. Pair k = engine mode 'coupled' (guide A and guide B, common random numbers, the existing coupled engine, unchanged) started from
     X0_A = X0_B = S_k, burn_pair blocks (20) discarded, then a FIXED measure_pair blocks (100).  The length is set before the run and never
     depends on any outcome (no adaptive stop; same_post is recorded, never used to end a pair).
  4. Delta_k = mean over the measured blocks of (E_B - E_A) (total energy; /NS per site).  Estimate = mean over pairs, error = scatter over pairs
     / sqrt(K).  Pairs are statistically independent only if the spacing exceeds the A-population autocorrelation time; summarize() reports
     the lag-1 correlation of the pair Deltas in branch order, and tau_A (2 tau_int of the single-chain E_A series) when the series is long enough.
The A chain has its own RNG (seed) and is not touched by the pairs (pair RNGs are separate), so the branch snapshots are the same with or without pairs.
Cost: the A chain costs spacing single blocks per pair on top of the burn_pair + measure_pair coupled blocks.  (Alternative not implemented: take the
snapshots from a coupled run's A arm; its marginal law is exactly the single-chain law.)
Wall-time guard (not outcome-dependent): maxsec stops the loop before a pair that is projected to overrun; K_done is recorded.
Controls: control_blocks > 0 runs, on every snapshot, a short A = A coupled run (B = a SEPARATE provider instance of A, gB_ctl): Delta must be exactly 0,
same_post exactly 1, no A_only/B_only hops; otherwise ControlFailure.  Broken control for the check itself: pass a gB_ctl that differs from gA.
"""
import time
import numpy as np
import engine as E
import analysis as AN


class ControlFailure(RuntimeError):
    pass


def control_check(gA, gA2, S, *, dtau, blocks, seed, NS):
    """A = A coupled run from snapshot S: exactly zero Delta."""
    o = E.run(gA, gA2, S, S, mode='coupled', dtau=dtau, burn_blocks=0, measure_blocks=blocks, pair_seed=seed, NS=NS)
    s = o['series']
    c = dict(blocks=int(len(s['delta'])), max_abs_delta=float(np.max(np.abs(s['delta']))), same_post_min=float(np.min(s['same_post'])),
             A_only=float(np.nansum(s['A_only'])), B_only=float(np.nansum(s['B_only'])), shared=float(np.nansum(s['shared'])), sec=o['wall'])
    c['passed'] = bool(c['max_abs_delta'] == 0 and c['same_post_min'] == 1 and c['A_only'] + c['B_only'] == 0 and c['shared'] > 0 and c['blocks'] == blocks)
    return c


def run_branched(gA, gB, X0, *, K, burn_A, spacing, burn_pair=20, measure_pair=100, dtau=0.1, seed=1, NS=1, gB_ctl=None, control_blocks=0,
                 maxsec=None, log=print, on_pair=None, ref_energy=0.0):
    """gA, gB: providers.  X0: initial population (M states) for the A chain.  Returns dict(pairs=[...], A_series=..., meta=...).
    on_pair(k, pair_dict, series_dict) is called after each pair (checkpointing).  gB_ctl: separate provider instance equal to gA (control)."""
    t0 = time.time(); M = len(X0)
    engA = E.PairedFN(gA, None, X0, None, mode='single', dtau=dtau, seed=seed, reference_energy=ref_energy)
    eA = []; tA = [0.0]
    def A_block():
        t = time.time(); r = engA.block(); tA[0] += time.time() - t; eA.append(r['emix_A'])
        if len(eA) % 10 == 0: log(f'PROGRESS A chain block {len(eA)} {time.time() - t0:.0f}s')
    for i in range(burn_A): A_block()
    log(f'PROGRESS branched A chain burned {burn_A} blocks emix_A/site {np.mean(eA[-10:]) / NS:.6f} {time.time() - t0:.0f}s')
    pairs = []; pair_wall = None; t_branch = []
    for k in range(K):
        if k > 0:
            for i in range(spacing): A_block()
        if maxsec and pair_wall is not None and time.time() - t0 + pair_wall > maxsec:
            log(f'PROGRESS maxsec {maxsec}: no new pair after {k} (projected pair {pair_wall:.0f}s)'); break
        S = np.array(engA.A.S, copy=True); tb = burn_A + k * spacing; ctl = None
        if control_blocks:
            ctl = control_check(gA, gB_ctl if gB_ctl is not None else gA, S, dtau=dtau, blocks=control_blocks, seed=900000 + k, NS=NS)
            log(f'PROGRESS pair {k} control A=A passed={ctl["passed"]} max|delta| {ctl["max_abs_delta"]} same_post_min {ctl["same_post_min"]} A_only+B_only {ctl["A_only"] + ctl["B_only"]}')
            if not ctl['passed']: raise ControlFailure(f'pair {k}: control A = A failed {ctl}')
        tp = time.time()
        o = E.run(gA, gB, S, S, mode='coupled', dtau=dtau, burn_blocks=burn_pair, measure_blocks=measure_pair, pair_seed=1000 * seed + k + 1, NS=NS, progress_every=10, log=lambda m, k=k: log(f'pair {k} {m}'))
        s = o['series']; m = ~s['burn']; eAm, eBm = s['emix_A'][m], s['emix_B'][m]
        ps = AN.pair_stats(eAm, eBm, NS=NS); d = (s['emix_B'] - s['emix_A']) / NS
        win = dict(burn_0_20=float(d[:burn_pair].mean()), w1=float(d[burn_pair:burn_pair + measure_pair // 2].mean()), w2=float(d[burn_pair + measure_pair // 2:].mean()))
        pr = dict(k=k, branch_block=tb, delta_site=ps['delta_site'], sigma_site_within=ps['sigma_site'], rho=ps['rho'], same_post_mean=float(s['same_post'][m].mean()),
                  same_post_min=float(s['same_post'][m].min()), same_post_burn_mean=float(s['same_post'][~m].mean()) if (~m).any() else None, windows=win,
                  alpha_mean=float(s['alpha'][m].mean()), wall=time.time() - tp, control=ctl)
        pair_wall = pr['wall']
        pairs.append(pr); t_branch.append(tb)
        log(f'PROGRESS PAIR_DONE {k} branch_block {tb} delta/site {pr["delta_site"]:+.3e} sigma_within {pr["sigma_site_within"]:.2e} same_post {pr["same_post_mean"]:.3f} wall {pr["wall"]:.0f}s')
        if on_pair: on_pair(k, pr, {kk: s[kk] for kk in ('emix_A', 'emix_B', 'delta', 'same_post', 'alpha', 'burn')})
    return dict(pairs=pairs, A_series=np.array(eA), meta=dict(K=K, K_done=len(pairs), M=M, burn_A=burn_A, spacing=spacing, burn_pair=burn_pair, measure_pair=measure_pair,
                dtau=dtau, seed=seed, NS=NS, control_blocks=control_blocks, wall=time.time() - t0, A_chain_sec=tA[0], A_chain_blocks=len(eA),
                pair_sec=float(sum(p['wall'] for p in pairs)), pair_blocks=len(pairs) * (burn_pair + measure_pair)))


def tau_int_A(eA_series, burn_A):
    """2 tau_int (blocks) of the post-burn single-chain E_A block series = long-run variance / variance (binning plateau); nan if < 32 blocks."""
    x = np.asarray(eA_series, float)[burn_A:]
    if len(x) < 64: return float('nan')
    return float(AN.long_run_var(x) / x.var(ddof=1))


def summarize(pairs, NS=1, spacing=None, tau2_A=None):
    """estimate and error from the scatter over pairs (per-site Delta)."""
    d = np.array([p['delta_site'] for p in pairs]); K = len(d)
    out = dict(K=K, delta_site=float(d.mean()), sd_pair=float(d.std(ddof=1)) if K > 1 else None, err_scatter=float(d.std(ddof=1) / np.sqrt(K)) if K > 1 else None,
               per_pair=[float(x) for x in d], sigma_within_rms=float(np.sqrt(np.mean([p['sigma_site_within'] ** 2 for p in pairs]))))
    if K > 2:
        c = d - d.mean(); out['lag1_corr_pairs'] = float((c[:-1] * c[1:]).sum() / (c * c).sum())
    if tau2_A is not None and spacing is not None:
        out['tau2_A_blocks'] = tau2_A; out['spacing_over_tau2_A'] = float(spacing / tau2_A) if tau2_A == tau2_A else None
    return out


def projected_cost(sd_pair, target, sec_pair_block, sec_A_block, spacing, burn_pair=20, measure_pair=100, var_scale=1.0):
    """pairs needed N = (sd_pair * sqrt(var_scale) / target)^2 and A40-h = N (pair blocks * c_pair + spacing * c_A) / 3600.  var_scale rescales the
    per-pair variance when the cost numbers refer to a different population size (assumption stated by the caller, e.g. M_probe / M_prod)."""
    N = (sd_pair * np.sqrt(var_scale) / target) ** 2
    sec = N * ((burn_pair + measure_pair) * sec_pair_block + spacing * sec_A_block)
    return dict(target=target, pairs=float(N), A40_h=float(sec / 3600))
