#!/usr/bin/env python3
"""probe8.py (paired_fn): 8x8 rho probe, coupled mode (PROPOSAL; lead's amendment 2026-11-12; submit only with the lead's OK).

Arms: A = s306 bare c = 0 FN guide sgn(Re w B)|B| (base_s306_p0/code/grade_bare.build_guide, BareM1Sign 'abs', params sha 45fa54e6, unchanged import),
      B = A with its sign flipped on a random sparse configuration set F = {x : splitmix64(key(x) ^ salt) < p_F 2^64} (fixed salt; a stand-in for residual
      flips: same amplitude, sign differs on a rare set).  Rows: GuideBase.finish with rr multiplied by m(x) m(y), m = -1 on F.
      Control: A = B (no mask): Delta == 0 in every block and same_post == 1 (rho = 1 by construction).
Shared amplitude work: both arms read one_swap_ratios of s306 through one per-configuration cache (key -> lnB, ratios), so B's rows at A's configuration
cost no GPU call and the A = B control is bitwise exact.
Measures (engine.py coupled block diagnostics): same_pre / same_post (matched fraction before / after resampling), alpha (recoupling probability at
resampling), sep_flux / matched_time (separation rate sum_b |r_A - r_B| while matched), cut_flux (exclusive-support rate h_cut, eq 14), shared /
A_only / B_only hop counts, F occupancy and F-stratum ESS; then long-run S_A, S_B, S_D, rho (analysis.pair_stats) over the measured blocks.
usage: probe8.py OUT --pF 1e-3 --control_blocks 40 --burn 30 --maxsec 700 [--M 1024] [--X0 file]
"""
import argparse, json, os, sys, time, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
BASE = os.environ.get('S306_CODE', '/project/theorie/a/A.Otaifi/runs/j1j2/base_s306_p0/code'); sys.path.insert(0, BASE)
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--pF', type=float, default=1e-3); ap.add_argument('--salt', type=int, default=0x5EED306)
ap.add_argument('--M', type=int, default=1024); ap.add_argument('--dtau', type=float, default=0.1); ap.add_argument('--control_blocks', type=int, default=40)
ap.add_argument('--burn', type=int, default=30); ap.add_argument('--blocks', type=int, default=100000); ap.add_argument('--maxsec', type=float, default=700)
ap.add_argument('--seed', type=int, default=306); ap.add_argument('--X0', default='/project/theorie/a/A.Otaifi/runs/j1j2/base_s306_p0/inputs/X0_B0s_0.npy')
ap.add_argument('--wchunk', type=int, default=128); ap.add_argument('--sa_sec', type=float, default=240)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T0 = time.time(); prog = open(f'{a.out}/progress.txt', 'a')
def P(m):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {m}'; print(line, flush=True); prog.write(line + '\n'); prog.flush()

import jax
if 'cuda' not in str(jax.devices()[0]).lower() and not os.environ.get('G8_ALLOW_CPU'): sys.exit(f'probe8: no CUDA device ({jax.devices()[0]})')
import grade_bare as GB                     # unchanged base code: build_guide (sha-pinned s306), GuideRec(BareM1Sign)
import gfmc_b2 as G2
import providers as PV, engine as E, analysis as AN

U = np.uint64
_PW = (U(1) << np.arange(64, dtype=U))
def cfg_keys(X): return ((np.asarray(X) > 0).astype(U) * _PW).sum(-1, dtype=U)
def splitmix(k):
    with np.errstate(over='ignore'):
        z = (np.asarray(k, U) + U(0x9E3779B97F4A7C15)); z = (z ^ (z >> U(30))) * U(0xBF58476D1CE4E5B9); z = (z ^ (z >> U(27))) * U(0x94D049BB133111EB)
        return z ^ (z >> U(31))
THR = U(min(int(a.pF * 2.0 ** 64), 2 ** 64 - 1))
def mask(keys): return np.where(splitmix(np.asarray(keys, U) ^ U(a.salt)) < THR, -1.0, 1.0)

CACHE = {}; CST = dict(hit=0, miss=0)
class Cached(GB.GuideRec):
    """BareM1Sign with a per-configuration cache of one_swap_ratios (shared by both arms) and an optional sign mask."""
    use_mask = False
    def ratios(self, X):
        if len(CACHE) > 150000: CACHE.clear()                            # clear BEFORE the lookup, never mid-call
        k = cfg_keys(X); need = [i for i, kk in enumerate(k) if int(kk) not in CACHE]
        CST['hit'] += len(k) - len(need); CST['miss'] += len(need)
        if need:
            Xn = X[need]; q = len(need); W = self.wc; pad = (-q) % W          # pad to whole chunks of wchunk: one jit shape, no per-size recompiles
            if pad: Xn = np.concatenate([Xn, np.repeat(Xn[:1], pad, 0)])
            lB, r, _, _ = G2.BareM1Sign.ratios(self, Xn); lB = np.asarray(lB)[:q]; r = np.asarray(r)[:q]
            for j, i in enumerate(need): CACHE[int(k[i])] = (lB[j], r[j])
        lB = np.array([CACHE[int(kk)][0] for kk in k]); r = np.stack([CACHE[int(kk)][1] for kk in k])
        self.last_lb = 2.0 * np.real(lB); return lB, r, lB, r
    def finish(self, X, anti, wi, bi, rr):
        if self.use_mask:
            kx = cfg_keys(X); ky = kx[wi] ^ _PW[self.BI[bi]] ^ _PW[self.BJ[bi]]
            rr = rr * mask(kx)[wi] * mask(ky)
        return super().finish(X, anti, wi, bi, rr)

gA, info = GB.build_guide('B0s', a.wchunk); gA.__class__ = Cached
gB, _ = GB.build_guide('B0s', a.wchunk); gB.__class__ = Cached; gB.use_mask = True
gAA, _ = GB.build_guide('B0s', a.wchunk); gAA.__class__ = Cached
ones = lambda X: np.ones(len(X))
pA = PV.ConfigGuide(gA, guide_id='s306_bare', sign_fn=ones)
pB = PV.ConfigGuide(gB, guide_id=f's306_bare_mask_pF{a.pF:g}_salt{a.salt}', sign_fn=lambda X: mask(cfg_keys(X)))
pAA = PV.ConfigGuide(gAA, guide_id='s306_bare', sign_fn=ones)
X0 = np.load(a.X0).astype(np.int8)[:a.M]; assert X0.shape == (a.M, 64)
res = dict(args=vars(a), guide=info, x0=a.X0, x0_sha256=hashlib.sha256(open(a.X0, 'rb').read()).hexdigest(),
           code_sha256={f: hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest() for f in ('engine.py', 'providers.py', 'analysis.py', 'probe8.py')},
           F_fraction_X0=float((mask(cfg_keys(X0)) < 0).mean()), device=str(jax.devices()[0]), job=os.environ.get('SLURM_JOB_ID'))
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(f'{a.out}/result.json', 'w'), indent=1, default=float)
dump()
# ---- control A = B
o = E.run(pA, pAA, X0, X0, mode='coupled', dtau=a.dtau, measure_blocks=a.control_blocks, pair_seed=a.seed, NS=64, progress_every=10, log=P)
s = o['series']; ctl = dict(blocks=len(s['delta']), max_abs_delta=float(np.max(np.abs(s['delta']))), same_post_min=float(np.min(s['same_post'])),
     A_only=float(np.nansum(s['A_only'])), B_only=float(np.nansum(s['B_only'])), shared=float(np.nansum(s['shared'])), sec_per_block=o['wall'] / len(s['delta']),
     cache=dict(CST)); ctl['passed'] = bool(ctl['max_abs_delta'] == 0 and ctl['same_post_min'] == 1 and ctl['A_only'] + ctl['B_only'] == 0 and ctl['shared'] > 0)
res['control_A_eq_B'] = ctl; dump(); P(f'CONTROL A=B passed={ctl["passed"]} max|delta| {ctl["max_abs_delta"]} sec/block {ctl["sec_per_block"]:.2f}')
np.savez(f'{a.out}/control_blocks.npz', **s)
# ---- probe A vs A x mask
CACHE.clear(); CST.update(hit=0, miss=0)
left = a.maxsec - (time.time() - T0)
o = E.run(pA, pB, X0, X0, mode='coupled', dtau=a.dtau, burn_blocks=a.burn, measure_blocks=a.blocks, pair_seed=a.seed + 1, NS=64, progress_every=10, log=P,
          maxsec=max(left, 60), output=a.out)
s = o['series']; m = ~s['burn']; nb = int(m.sum())
eA, eB = s['emix_A'][m], s['emix_B'][m]
probe = dict(blocks_measured=nb, sec_per_block=o['wall'] / len(s['delta']), cache=dict(CST),
             same_pre_mean=float(np.mean(s['same_pre'][m])), same_post_mean=float(np.mean(s['same_post'][m])), alpha_mean=float(np.mean(s['alpha'][m])),
             sep_rate=float(np.sum(s['sep_flux'][m]) / max(np.sum(s['matched_time'][m]), 1e-300)),
             cut_rate=float(np.sum(s['cut_flux'][m]) / max(np.sum(s['matched_time'][m]), 1e-300)),
             hops=dict(shared=float(np.sum(s['shared'][m])), A_only=float(np.sum(s['A_only'][m])), B_only=float(np.sum(s['B_only'][m]))),
             F_count_A=float(np.nanmean(s['F_count_A'][m])), F_weight_A=float(np.nanmean(s['F_weight_A'][m])), ess_A=float(np.nanmean(s['A_ess'][m])),
             ess_B=float(np.nanmean(s['B_ess'][m])), mmax_B=float(np.nanmax(s['B_mmax'][m])),
             F_count_B=float(np.nanmean(s['F_count_B'][m])), F_weight_B=float(np.nanmean(s['F_weight_B'][m])),
             F_ess_frac_A=float(np.nanmean(s['F_ess_frac_A'][m])), F_ess_frac_B=float(np.nanmean(s['F_ess_frac_B'][m])),
             blocks_with_F_A=int(np.sum(s['F_count_A'][m] > 0)))
probe['separation_time'] = 1.0 / probe['sep_rate'] if probe['sep_rate'] > 0 else float('inf')
for g in ('A', 'B'): np.save(f'{a.out}/E_{g}.npy', s[f'emix_{g}'][m] / 64)
np.save(f'{a.out}/E_D.npy', (eB - eA) / 64); res['probe'] = probe; dump()
P(f'PROBE coupled done: {nb} measured blocks, {probe["sec_per_block"]:.2f} s/block')
# ---- standalone arms (multinomial, same engine, mode single) for kappa_work: S_A^sa, S_B^sa and their costs
sa = {}
for nm, g, sd in (('A', pA, a.seed + 2), ('B', pB, a.seed + 3)):
    CACHE.clear()
    o2 = E.run(g, None, X0, mode='single', dtau=a.dtau, burn_blocks=a.burn, measure_blocks=a.blocks, pair_seed=sd, NS=64, progress_every=20, log=P, maxsec=a.sa_sec)
    s2 = o2['series']; m2 = ~s2['burn']; sa[nm] = dict(E=s2['emix_A'][m2], c=o2['wall'] / len(s2['emix_A']), n=int(m2.sum()))
    np.save(f'{a.out}/E_{nm}_standalone.npy', s2['emix_A'][m2] / 64)
    probe[f'standalone_{nm}'] = dict(blocks=sa[nm]['n'], sec_per_block=sa[nm]['c'], mean_site=float(np.mean(s2['emix_A'][m2]) / 64)); dump()
if nb >= 64 and min(sa['A']['n'], sa['B']['n']) >= 64:
    probe['pair'] = AN.pair_stats(eA, eB, probe['sec_per_block'], sa['A']['E'], sa['B']['E'], sa['A']['c'], sa['B']['c'], NS=64)
    for nbins in (16, 32):
        if nb >= 4 * nbins: probe[f'rho_binned_{nbins}'] = AN.rho_binned(eA, eB, nbins)
res['probe'] = probe; res['status'] = 'DONE'; dump()
P(f'PROBE8_DONE control_passed={ctl["passed"]} blocks {nb} same_post {probe["same_post_mean"]:.3f} alpha {probe["alpha_mean"]:.3f} '
  f'sep_time {probe["separation_time"]:.3g} rho {probe.get("pair", {}).get("rho", float("nan")):.3f}')
