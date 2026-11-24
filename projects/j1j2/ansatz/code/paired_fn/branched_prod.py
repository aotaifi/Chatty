#!/usr/bin/env python3
"""branched_prod.py OUT [--M --K --burn_A --spacing --burn_pair --measure_pair --maxsec --net4 --x0 --seed] : branched short-pair grader at 8x8 (T-BAD amplitude |B0|).
A = StoredCNNSign(n_3 = sign_bad/r3/fit/net_8192.pkl), B = StoredCNNSign(n_4 = r4/fit/net_8192.pkl), same amplitude/dtau/guides as grade4.sbatch (cnn_sign.build 'stored').
One A population is equilibrated (burn_A blocks, start X0_B0_0 first M rows, as grade4 chain 0), K pairs branched at spacing blocks, each 20 burn + 100 measured (fixed).
Every snapshot first runs the A = A control (control_blocks, default 5): Delta == 0 exactly, same_post 1, else exit 3.  maxsec: no new pair is started if projected to overrun
(wall-time guard, not outcome dependent).  Writes pair_<k>.npz, progress.txt, result.json (per pair Delta/site, within-pair sigma, same_post, windows; scatter estimate; costs)."""
import argparse, hashlib, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--x0dir', default='/scratch/a/A.Otaifi/ansatz_runs/sign_node/inputs'); ap.add_argument('--x0', default='X0_B0_0.npy')
ap.add_argument('--net4', default=None); ap.add_argument('--chunk', type=int, default=128); ap.add_argument('--M', type=int, default=256); ap.add_argument('--K', type=int, default=4)
ap.add_argument('--burn_A', type=int, default=150); ap.add_argument('--spacing', type=int, default=25); ap.add_argument('--burn_pair', type=int, default=20); ap.add_argument('--measure_pair', type=int, default=100)
ap.add_argument('--control_blocks', type=int, default=5); ap.add_argument('--maxsec', type=float, default=None); ap.add_argument('--seed', type=int, default=7001)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T0 = time.time(); prog = open(f'{a.out}/progress.txt', 'a')
def P(m):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {m}'; print(line, flush=True); prog.write(line + '\n'); prog.flush()
R = '/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad/'; N3SHA = '5b8b6dc437de0178379895b531ef92209ca141426469a32755f50eb7fa5ca8f1'; N4SHA = '6ebf24474e2341e79783fd045c47956f93a3596f0a73c5ee943d7325b4f6e34a'
import jax
if 'cuda' not in str(jax.devices()[0]).lower() and not os.environ.get('G8_ALLOW_CPU'): sys.exit(f'branched_prod: no CUDA device ({jax.devices()[0]})')
import cnn_sign as CS, providers as PV, engine as E, analysis as AN, branched_fn as BF
amp = CS.AmpCache(chunk=a.chunk, cap=300000); n3 = CS.NetSign(R + 'r3/fit/net_8192.pkl', 'n3'); assert n3.sha == N3SHA, 'n3 sha'
p4 = a.net4 or R + 'r4/fit/net_8192.pkl'; n4 = CS.NetSign(p4, 'n4'); assert n4.sha == N4SHA, 'n4 sha'
def mk(nets):
    g, _ = CS.build('stored', amp, nets, wchunk=a.chunk); return PV.ConfigGuide(g, guide_id=f'stored:{nets.name}'), g
gA, _ = mk(n3); gA2, _ = mk(n3); gB, gBraw = mk(n4)
f = os.path.join(a.x0dir, a.x0); X0 = np.load(f).astype(np.int8)[:a.M]; assert X0.shape == (a.M, 64)
hdr = dict(args=vars(a), job=os.environ.get('SLURM_JOB_ID'), device=str(jax.devices()[0]), x0_sha256=hashlib.sha256(open(f, 'rb').read()).hexdigest(),
           code_sha256={fn: hashlib.sha256(open(os.path.join(HERE, fn), 'rb').read()).hexdigest() for fn in ('engine.py', 'providers.py', 'analysis.py', 'cnn_sign.py', 'branched_fn.py', 'branched_prod.py')})
P(f'branched_prod start M {a.M} K {a.K} burn_A {a.burn_A} spacing {a.spacing} pair {a.burn_pair}+{a.measure_pair} control {a.control_blocks} maxsec {a.maxsec} device {jax.devices()[0]}')
saved = []
def on_pair(k, pr, s):
    np.savez(f'{a.out}/pair_{k}.npz', **s); saved.append(pr)
    json.dump(dict(**hdr, pairs=saved, done=False), open(f'{a.out}/result.partial.json', 'w'), indent=1, default=float)
try:
    out = BF.run_branched(gA, gB, X0, K=a.K, burn_A=a.burn_A, spacing=a.spacing, burn_pair=a.burn_pair, measure_pair=a.measure_pair, dtau=0.1, seed=a.seed, NS=64, gB_ctl=gA2,
                          control_blocks=a.control_blocks, maxsec=a.maxsec, log=P, on_pair=on_pair)
except BF.ControlFailure as e:
    P(f'CONTROL FAILED ABORT {e}'); json.dump(dict(**hdr, aborted=str(e), done=False), open(f'{a.out}/result.json', 'w'), indent=1, default=float); sys.exit(3)
m = out['meta']; tau2 = BF.tau_int_A(out['A_series'], a.burn_A)
sm = BF.summarize(out['pairs'], NS=64, spacing=a.spacing, tau2_A=tau2)
cA = m['A_chain_sec'] / m['A_chain_blocks']; cP = m['pair_sec'] / m['pair_blocks']
res = dict(**hdr, meta=m, summary=sm, pairs=out['pairs'], sec_per_A_block=cA, sec_per_pair_block=cP, done=True, sec=time.time() - T0, cache=dict(hit=amp.hit, miss=amp.miss), stats=dict(gB.st))
json.dump(res, open(f'{a.out}/result.json', 'w'), indent=1, default=float); np.save(f'{a.out}/A_series.npy', out['A_series'])
P(f'BRANCHED_DONE K_done {sm["K"]} delta/site {sm["delta_site"]:+.3e} sd_pair {sm["sd_pair"]} err {sm["err_scatter"]} cA {cA:.2f} cP {cP:.2f} wall {time.time() - T0:.0f}s')
