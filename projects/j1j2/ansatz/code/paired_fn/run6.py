#!/usr/bin/env python3
"""run6.py (paired_fn): one exact-6x6 ensemble (a pair in coupled/union/reweight_A mode, or one standalone arm).
usage: run6.py OUTDIR --pair NAME | --single GUIDE  [--mode coupled] [--M 1024] [--dtau 0.1] [--burn 500] [--blocks 30000] [--seed s]
               [--start guide2|uniform] [--maxsec T] [--brokenB no_vsf|inverse_ratio] [--wallsB_from GUIDE] [--engine_broken inverse_hop_ratio]
Writes OUTDIR/blocks.npz, run_meta.json, E_A.npy / E_B.npy / E_D.npy (measured blocks, per site: E_D = E_B - E_A), summary.json.
Initial states: one draw of M orbits from the A guide's |psi|^2 ('guide2') or from n_r ('uniform', a random configuration), shared by both arms."""
import argparse, json, os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import guides6 as G, engine as E, analysis as AN
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--pair'); ap.add_argument('--single'); ap.add_argument('--mode', default='coupled')
ap.add_argument('--M', type=int, default=1024); ap.add_argument('--dtau', type=float, default=0.1); ap.add_argument('--burn', type=int, default=500)
ap.add_argument('--blocks', type=int, default=30000); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--start', default='guide2')
ap.add_argument('--maxsec', type=float, default=None); ap.add_argument('--brokenB', default=None); ap.add_argument('--wallsB_from', default=None)
ap.add_argument('--engine_broken', default=None); ap.add_argument('--memory_tau', type=int, default=None)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T0 = time.time(); NS = 36
prog = open(f'{a.out}/progress.txt', 'a')
def log(m):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {m}'; print(line, flush=True); prog.write(line + '\n'); prog.flush()
if a.single:
    gA = G.load(a.single); gB = None; ex = None; nameA, nameB = a.single, None
else:
    nameA, nameB = G.PAIRS[a.pair]; gA = G.load(nameA); gB = G.load(nameB, broken=a.brokenB, walls_from=a.wallsB_from); ex = G.exact_delta(a.pair)
rng = np.random.default_rng([a.seed, 6]); X0 = G.sample_X0(gA, a.M, rng, a.start)
log(f'PROGRESS start pair={a.pair} single={a.single} mode={a.mode} M={a.M} dtau={a.dtau} seed={a.seed} start={a.start} load {time.time() - T0:.0f}s')
o = E.run(gA, gB, X0, X0, mode='single' if a.single else a.mode, dtau=a.dtau, burn_blocks=a.burn, measure_blocks=a.blocks, pair_seed=[a.seed, 7],
          NS=NS, progress_every=1000, log=log, output=a.out, maxsec=a.maxsec, broken=a.engine_broken, memory_tau=a.memory_tau,
          meta=dict(pair=a.pair, single=a.single, guide_A_name=nameA, guide_B_name=nameB, start=a.start, brokenB=a.brokenB, wallsB_from=a.wallsB_from))
s = o['series']; m = ~s['burn']; nb = int(m.sum())
eA = s['emix_A'][m]; np.save(f'{a.out}/E_A.npy', eA / NS)
summ = dict(args=vars(a), blocks=nb, sec=o['wall'], sec_per_block=o['wall'] / len(s['emix_A']), EA_site=float(eA.mean() / NS),
            EA_err_site=AN.mean_err(eA)[1] / NS, F_exact_A=G.resolve_F(G.GUIDES[nameA])[0], E0=G.P.orbit_tables()['E0'])
summ['EA_minus_exact_site'] = summ['EA_site'] - (summ['E0'] / NS + summ['F_exact_A'])
if gB is not None:
    eB = s['emix_B'][m]; np.save(f'{a.out}/E_B.npy', eB / NS); np.save(f'{a.out}/E_D.npy', (eB - eA) / NS)
    st = AN.pair_stats(eA, eB, NS=NS); summ.update(pair=st, delta_exact_site=ex, z=(st['delta_site'] - ex) / st['sigma_site'],
        EB_site=float(eB.mean() / NS), EB_minus_exact_site=float(eB.mean() / NS - (summ['E0'] / NS + G.resolve_F(G.GUIDES[nameB])[0])),
        fifths_D_site=[x / NS for x in AN.fifths(eB - eA)], same_post=float(np.nanmean(s['same_post'][m])) if a.mode == 'coupled' else None,
        alpha=float(np.nanmean(s['alpha'][m])) if a.mode == 'coupled' else None,
        sep_rate=float(np.nansum(s['sep_flux'][m]) / max(np.nansum(s['matched_time'][m]), 1e-300)) if a.mode == 'coupled' else None,
        cut_rate=float(np.nansum(s['cut_flux'][m]) / max(np.nansum(s['matched_time'][m]), 1e-300)) if a.mode == 'coupled' else None,
        F_weight_A=float(np.nanmean(s['F_weight_A'][m])), F_ess_frac_A=float(np.nanmean(s['F_ess_frac_A'][m])), ess_B=float(np.nanmean(s['B_ess'][m])),
        mmax_B=float(np.nanmax(s['B_mmax'][m])))
json.dump(summ, open(f'{a.out}/summary.json', 'w'), indent=1, default=float)
msg = f'RUN6_DONE blocks {nb} {summ["sec_per_block"]:.3f} s/block EA-exact {summ["EA_minus_exact_site"]:+.2e}'
if gB is not None: msg += f' delta {summ["pair"]["delta_site"]:+.3e} +- {summ["pair"]["sigma_site"]:.2e} exact {ex:+.3e} z {summ["z"]:+.2f} rho {summ["pair"]["rho"]:.3f}'
log(msg)
