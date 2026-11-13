#!/usr/bin/env python3
"""gated_probe.py OUT [--gate -5] [--blocks_plain 6] [--blocks_ctl 20] [--blocks_broken 3] [--burn 4] [--maxsec 780]: timing + controls of the gated exact one-step sign inside the paired FN engine (coupled mode).
Arm A = probe8 s306 guide (GatedGuide gate -inf == probe8 Cached BareM1Sign), arm B = GatedGuide(gate); one shared per-configuration cache; same bond IDs, M = 1024, dtau 0.1, X0 = base_s306_p0 X0_B0s_0.
Stages (cache cleared before 1 and 4; stage 2 reuses the warm cache of stage 1):
  1 plain A = A control (gate -inf both), blocks_plain blocks: sec/block, cold cache.
  2 CONTROL 1: A vs B(gate -inf), blocks_ctl blocks: Delta must be exactly 0 (also same_post == 1, no A_only/B_only hops).
  3 broken control: A vs B(gate -inf, 'hashmask' p 1e-3 extra sign flips), blocks_broken blocks: Delta must be != 0 (the control can fail).
  4 paired run A vs B(gate), cache cleared (cold): burn + measured blocks until maxsec; per-block times, gate-hit fractions, separation rate per tau.
NOT the production paired measurement."""
import argparse, hashlib, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
BASE = os.environ.get('S306_CODE', '/project/theorie/a/A.Otaifi/runs/j1j2/base_s306_p0/code'); sys.path.insert(0, BASE)
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--gate', type=float, default=-5); ap.add_argument('--M', type=int, default=1024)
ap.add_argument('--dtau', type=float, default=0.1); ap.add_argument('--blocks_plain', type=int, default=6); ap.add_argument('--blocks_ctl', type=int, default=20)
ap.add_argument('--blocks_broken', type=int, default=3); ap.add_argument('--burn', type=int, default=4); ap.add_argument('--blocks', type=int, default=100000)
ap.add_argument('--maxsec', type=float, default=780); ap.add_argument('--seed', type=int, default=3060); ap.add_argument('--wchunk', type=int, default=128)
ap.add_argument('--X0', default='/project/theorie/a/A.Otaifi/runs/j1j2/base_s306_p0/inputs/X0_B0s_0.npy'); ap.add_argument('--skip_stages', default='')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T0 = time.time(); prog = open(f'{a.out}/progress.txt', 'a')
def P(m):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {m}'; print(line, flush=True); prog.write(line + '\n'); prog.flush()
import jax
if 'cuda' not in str(jax.devices()[0]).lower() and not os.environ.get('G8_ALLOW_CPU'): sys.exit(f'gated_probe: no CUDA device ({jax.devices()[0]})')
import gated_sign as GS, providers as PV, engine as E, analysis as AN
cache = GS.Cache(cap=300000)
def mk(gate, broken=None, hash_p=1e-3):
    g, info = GS.build(gate, cache, broken=broken, wchunk=a.wchunk, hash_p=hash_p); return g, info
def prov(g, tag, rel=False):
    return PV.ConfigGuide(g, guide_id=tag, sign_fn=(g.rel_sign if rel else (lambda X: np.ones(len(X)))))
X0 = np.load(a.X0).astype(np.int8)[:a.M]; assert X0.shape == (a.M, 64)
res = dict(args=vars(a), x0=a.X0, x0_sha256=hashlib.sha256(open(a.X0, 'rb').read()).hexdigest(), device=str(jax.devices()[0]), job=os.environ.get('SLURM_JOB_ID'),
           alpha_E=GS.read_sign_constants(), consts_sha=GS.CONSTS_SHA, MREF=GS.MREF,
           code_sha256={f: hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest() for f in ('engine.py', 'providers.py', 'gated_sign.py', 'gated_probe.py')})
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(f'{a.out}/result.json', 'w'), indent=1, default=float)
dump()
def timed(label):
    ts = []
    def log(m):
        ts.append(time.time()); P(f'{label} {m}')
    return ts, log
def per_block(ts, t_start):
    t = np.array([t_start] + list(ts)); return np.diff(t).tolist()
def stage(label, gA, gB, nblocks, burn=0, maxsec=None, seed=0, rel=False, keep=False):
    cache.reset_counts()
    pA, pB = prov(gA, 'A_s306'), prov(gB, f'B_{label}', rel=rel)
    ts, log = timed(label); t_start = time.time()
    o = E.run(pA, pB, X0, X0, mode='coupled', dtau=a.dtau, burn_blocks=burn, measure_blocks=nblocks, pair_seed=seed, NS=64, progress_every=1, log=log, maxsec=maxsec,
              output=(a.out if keep else None))
    s = o['series']; m = ~s['burn']; pb = per_block(ts, t_start)
    r = dict(blocks=int(len(s['delta'])), measured=int(m.sum()), wall=o['wall'], t_init=o['t_init'], sec_per_block_all=o['wall'] / len(s['delta']), block_sec=pb,
             sec_per_block_first3=float(np.mean(pb[:3])), sec_per_block_after3=float(np.mean(pb[3:])) if len(pb) > 3 else None,
             sec_per_block_last_half=float(np.mean(pb[len(pb) // 2:])), cache=dict(hit=cache.hit, miss=cache.miss, gpu_sec=cache.sec, gpu_calls=cache.calls, size=len(cache.d)),
             max_abs_delta=float(np.max(np.abs(s['delta']))), same_post_min=float(np.min(s['same_post'])), A_only=float(np.nansum(s['A_only'])),
             B_only=float(np.nansum(s['B_only'])), shared=float(np.nansum(s['shared'])))
    return o, s, m, r
skip = set(a.skip_stages.split(','))
# warm-up: compile the one_swap_ratios kernel on one chunk, then drop the cache
gw, _ = mk(-math.inf); gw.rows(X0[:8]); cache.d.clear(); P('warm-up done, compile excluded from all stages')
# ---- stage 1: plain A = A
gA1, info = mk(-math.inf); gA2, _ = mk(-math.inf); res['guide'] = info
o, s, m, r = stage('plainAA', gA1, gA2, a.blocks_plain, seed=a.seed); res['plain_AA'] = r; dump(); P(f'PLAIN A=A: {r["sec_per_block_all"]:.2f} s/block (first3 {r["sec_per_block_first3"]:.2f}) max|delta| {r["max_abs_delta"]}')
# ---- stage 2: control 1 (gated gate = -inf vs A), warm cache from stage 1
gA, _ = mk(-math.inf); gBm, _ = mk(-math.inf)
o, s, m, r = stage('ctl1', gA, gBm, a.blocks_ctl, seed=a.seed + 1)
r['passed'] = bool(r['max_abs_delta'] == 0 and r['same_post_min'] == 1 and r['A_only'] + r['B_only'] == 0 and r['shared'] > 0 and r['blocks'] >= 20)
r['gate_stats'] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in gBm.st.items()}; res['control1_gate_minf'] = r; dump(); P(f'CONTROL1 passed={r["passed"]} blocks {r["blocks"]} max|delta| {r["max_abs_delta"]} {r["sec_per_block_all"]:.2f} s/block')
# ---- stage 3: broken control (must NOT be 0)
gA, _ = mk(-math.inf); gBb, _ = mk(-math.inf, 'hashmask', 1e-3)
o, s, m, r = stage('broken', gA, gBb, a.blocks_broken, seed=a.seed + 2)
r['detected'] = bool(r['max_abs_delta'] > 0); res['control1_broken_hashmask'] = r; dump(); P(f'BROKEN hashmask 1e-3 detected={r["detected"]} max|delta| {r["max_abs_delta"]}')
# ---- stage 4: paired A vs B(gate), cold cache
cache.d.clear(); gA, _ = mk(-math.inf); gB, infoB = mk(a.gate); res['guide_B'] = infoB
left = a.maxsec
o, s, m, r = stage('paired', gA, gB, a.blocks, burn=a.burn, maxsec=left, seed=a.seed + 3, rel=True, keep=True)
st = gB.st; nbd = st['nb_dec_hist']; hist_cum = {int(d): float(nbd[d - GS.DEC0]) for d in range(GS.DEC0, GS.DEC1 + 1) if nbd[d - GS.DEC0] > 0}
r['gate'] = a.gate
r['gate_stats'] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in st.items()}
r['gate_hit_fraction_neighbours'] = st['nb_gated'] / max(st['nb'], 1); r['gate_hit_fraction_candidates'] = st['nb_cand'] / max(st['nb'], 1)
r['gate_hit_fraction_x'] = st['x_gated'] / max(st['x'], 1); r['flip_fraction_of_gated_neighbours'] = st['nb_flip'] / max(st['nb_gated'], 1)
r['neighbour_decade_counts'] = hist_cum
eA, eB = s['emix_A'][m], s['emix_B'][m]; mt = np.sum(s['matched_time'][m])
r['sep_rate_per_tau'] = float(np.sum(s['sep_flux'][m]) / max(mt, 1e-300)); r['cut_rate_per_tau'] = float(np.sum(s['cut_flux'][m]) / max(mt, 1e-300))
r['same_post_mean'] = float(np.mean(s['same_post'][m])); r['alpha_mean'] = float(np.mean(s['alpha'][m]))
r['hops'] = dict(shared=float(np.sum(s['shared'][m])), A_only=float(np.sum(s['A_only'][m])), B_only=float(np.sum(s['B_only'][m])))
r['F_count_A'] = float(np.nanmean(s['F_count_A'][m])); r['F_count_B'] = float(np.nanmean(s['F_count_B'][m]))
r['delta_per_site_mean'] = float(np.mean((eB - eA)) / 64); r['delta_per_site_note'] = 'short run, not a measurement (blocks too few, not equilibrated)'
r['matched_tau'] = float(mt); r['tau_measured'] = float(r['measured'] * a.dtau)
res['paired_gate'] = r; res['status'] = 'DONE'; dump()
np.save(f'{a.out}/E_A.npy', eA / 64); np.save(f'{a.out}/E_B.npy', eB / 64)
P(f'GATED_PROBE_DONE control1={res["control1_gate_minf"]["passed"]} broken_detected={res["control1_broken_hashmask"]["detected"]} paired blocks {r["measured"]} {r["sec_per_block_all"]:.2f} s/block '
  f'gate_hit_nb {r["gate_hit_fraction_neighbours"]:.4f} sep_rate {r["sep_rate_per_tau"]:.3g}')
