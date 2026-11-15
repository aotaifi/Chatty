#!/usr/bin/env python3
"""sep_probe.py OUT --arm exact|stored4 [--net4 PATH] [--M 256] [--blocks N] [--burn b] [--maxsec S]: T-SEP rho probe (GPU slice; CPU only with G8_ALLOW_CPU=1).
A = StoredCNNSign(n_3);  B1 (--arm exact) = ExactStepSign(n_3) (K[n_3] at walker time);  B2 (--arm stored4) = StoredCNNSign(n_4), n_4 from --net4 (default sign_bad/r4/final_net.txt).
Amplitude |B0|; one shared amplitude cache; coupled engine; X0 = T-NODE start X0_B0_3.  Stage 1: control A = A (Delta == 0, same_post 1).  Stage 2: paired run, then
pair_stats (rho, S_A, S_B, S_D) when >= 64 measured blocks, plus same_post, alpha, separation rate per tau, s/block, cache counters.  NOT the production measurement."""
import argparse, hashlib, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
R = '/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad/'
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--arm', required=True, choices=['exact', 'stored4']); ap.add_argument('--net4', default=None)
ap.add_argument('--M', type=int, default=256); ap.add_argument('--dtau', type=float, default=0.1); ap.add_argument('--ctl_blocks', type=int, default=10)
ap.add_argument('--burn', type=int, default=10); ap.add_argument('--blocks', type=int, default=100000); ap.add_argument('--maxsec', type=float, default=1500)
ap.add_argument('--seed', type=int, default=4001); ap.add_argument('--chunk', type=int, default=128)
ap.add_argument('--X0', default='/scratch/a/A.Otaifi/ansatz_runs/sign_node/inputs/X0_B0_3.npy'); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); T0 = time.time(); prog = open(f'{a.out}/progress.txt', 'a')
def P(m):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {m}'; print(line, flush=True); prog.write(line + '\n'); prog.flush()
import jax
if 'cuda' not in str(jax.devices()[0]).lower() and not os.environ.get('G8_ALLOW_CPU'): sys.exit(f'sep_probe: no CUDA device ({jax.devices()[0]})')
import cnn_sign as CS, providers as PV, engine as E, analysis as AN
amp = CS.AmpCache(chunk=a.chunk, cap=150000); n3 = CS.NetSign(R + 'r3/fit/net_8192.pkl', 'n3')
if a.arm == 'stored4':
    p4 = a.net4 or open('/scratch/a/A.Otaifi/ansatz_runs/sign_bad/r4/final_net.txt').read().split()[0]; n4 = CS.NetSign(p4, 'n4')
else: n4 = None
X0 = np.load(a.X0).astype(np.int8)[:a.M]; assert X0.shape == (a.M, 64)
res = dict(args=vars(a), n3=(n3.path, n3.sha), n4=(n4.path, n4.sha) if n4 else None, alpha_E=(CS.ALPHA, CS.E_REF), device=str(jax.devices()[0]), job=os.environ.get('SLURM_JOB_ID'),
           x0_sha256=hashlib.sha256(open(a.X0, 'rb').read()).hexdigest(),
           code_sha256={f: hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest() for f in ('engine.py', 'providers.py', 'analysis.py', 'cnn_sign.py', 'sep_probe.py')})
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(f'{a.out}/result.json', 'w'), indent=1, default=float)
dump()
def mk(mode, nets, broken=None):
    g, _ = CS.build(mode, amp, nets, broken=broken, wchunk=a.chunk); return PV.ConfigGuide(g, guide_id=f'{mode}:{nets.name}:{broken}'), g
pA, gA = mk('stored', n3)
# ---- stage 1: control A = A
pA2, _ = mk('stored', n3)
ts = []; o = E.run(pA, pA2, X0, X0, mode='coupled', dtau=a.dtau, measure_blocks=a.ctl_blocks, pair_seed=a.seed, NS=64, progress_every=1, log=lambda m: (ts.append(time.time()), P('ctl ' + m)))
s = o['series']; c = dict(blocks=len(s['delta']), max_abs_delta=float(np.max(np.abs(s['delta']))), same_post_min=float(np.min(s['same_post'])), A_only=float(np.nansum(s['A_only'])),
     B_only=float(np.nansum(s['B_only'])), shared=float(np.nansum(s['shared'])), sec_per_block=o['wall'] / len(s['delta']))
c['PASS'] = bool(c['max_abs_delta'] == 0 and c['same_post_min'] == 1 and c['A_only'] + c['B_only'] == 0 and c['shared'] > 0); res['control_A_eq_A'] = c; dump(); P(f'CONTROL A=A passed={c["PASS"]} {c["sec_per_block"]:.2f} s/block')
if not c['PASS']: sys.exit('control failed')
# ---- stage 2: paired
amp.reset_counts(); n3.n_eval = 0; n3.sec = 0.0
pB, gB = mk('exact', n3) if a.arm == 'exact' else mk('stored', n4)
ts2 = []; t_start = time.time()
o = E.run(pA, pB, X0, X0, mode='coupled', dtau=a.dtau, burn_blocks=a.burn, measure_blocks=a.blocks, pair_seed=a.seed + 1, NS=64, progress_every=1, maxsec=max(a.maxsec - (time.time() - T0), 60),
          log=lambda m: (ts2.append(time.time()), P('paired ' + m)), output=a.out)
s = o['series']; m = ~s['burn']; nb = int(m.sum()); pb = np.diff(np.array([t_start] + ts2))
eA, eB = s['emix_A'][m], s['emix_B'][m]; mt = float(np.sum(s['matched_time'][m]))
r = dict(arm=a.arm, blocks_measured=nb, sec_per_block=o['wall'] / len(s['delta']), sec_per_block_last_half=float(np.mean(pb[len(pb) // 2:])), same_post_mean=float(np.mean(s['same_post'][m])),
         same_pre_mean=float(np.mean(s['same_pre'][m])), alpha_mean=float(np.mean(s['alpha'][m])), sep_rate_per_tau=float(np.sum(s['sep_flux'][m]) / max(mt, 1e-300)),
         cut_rate_per_tau=float(np.sum(s['cut_flux'][m]) / max(mt, 1e-300)), hops=dict(shared=float(np.sum(s['shared'][m])), A_only=float(np.sum(s['A_only'][m])), B_only=float(np.sum(s['B_only'][m]))),
         ess_A=float(np.nanmean(s['A_ess'][m])), ess_B=float(np.nanmean(s['B_ess'][m])), delta_site_short=float(np.mean(eB - eA) / 64), note='short run, not a measurement',
         amp=dict(hit=amp.hit, miss=amp.miss, sec=amp.sec), n3_net=dict(eval=n3.n_eval, sec=n3.sec), n4_net=(dict(eval=n4.n_eval, sec=n4.sec) if n4 else None),
         st={k: float(v) for k, v in gB.st.items()}, hops_total_per_block=float((np.sum(s['shared'][m]) + np.sum(s['A_only'][m]) + np.sum(s['B_only'][m])) / max(nb, 1)))
r['separation_time'] = 1.0 / r['sep_rate_per_tau'] if r['sep_rate_per_tau'] > 0 else float('inf')
if nb >= 64:
    r['pair'] = AN.pair_stats(eA, eB, r['sec_per_block'], NS=64)
    for nbins in (16, 32):
        if nb >= 4 * nbins: r[f'rho_binned_{nbins}'] = AN.rho_binned(eA, eB, nbins)
np.save(f'{a.out}/E_A.npy', eA / 64); np.save(f'{a.out}/E_B.npy', eB / 64)
res['probe'] = r; res['status'] = 'DONE'; dump()
P(f'SEP_PROBE_DONE arm={a.arm} blocks {nb} same_post {r["same_post_mean"]:.3f} alpha {r["alpha_mean"]:.3f} sep_time {r["separation_time"]:.3g} rho {r.get("pair", {}).get("rho", float("nan")):.3f} {r["sec_per_block"]:.2f} s/block')
