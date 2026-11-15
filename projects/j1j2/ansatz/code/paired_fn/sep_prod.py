#!/usr/bin/env python3
"""sep_prod.py OUT --table chains_sep.json [--only i,j] [--M --burn --measure --control_blocks --ctl_broken] : T-SEP production driver (PLAN_sign.md "T-SEP DESIGN").
Paired FN on the 8x8 T-BAD amplitude |B0| (rb8_prod): A = StoredCNNSign(n_3);  arm 'exact': B1 = ExactStepSign(n_3) (K[n_3] at walker time);  arm 'stored4': B2 = StoredCNNSign(n_4).
G_exact = E[A] - E[B1], G_stored = E[A] - E[B2], f = G_stored / G_exact (per-site energies; Delta/site = (E_B - E_A)/64 = -G).
Table: {"dtau":0.1,"M":1024,"burn":..,"measure":..,"control_blocks":..,"control_seed":..,"chains":[{"id":"exact_3","arm":"exact","x0_file":"X0_B0_3.npy","x0_sha256":..,"seed":..},..]}
Start (every start): control A = A (StoredCNNSign(n_3) vs itself): Delta == 0 exactly, same_post 1, no A_only/B_only, else exit 3.  Per chain chain_<id>.npz/.json/.done (resume = chain granularity).
result.json: per chain pair_stats; per arm mean Delta/site, error = larger of pooled long-run and chain scatter; G_exact, G_stored, f with propagated error; 'done' true when all chains are done.
Overrides (test only: result.json then smoke=true): --M --burn --measure --control_blocks --only --ctl_broken shuffle (control with a shuffled-weight B must abort, exit 3)."""
import argparse, hashlib, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--table', required=True); ap.add_argument('--x0dir', default='/scratch/a/A.Otaifi/ansatz_runs/sign_node/inputs')
ap.add_argument('--net4', default=None); ap.add_argument('--chunk', type=int, default=128); ap.add_argument('--M', type=int); ap.add_argument('--burn', type=int); ap.add_argument('--measure', type=int)
ap.add_argument('--control_blocks', type=int); ap.add_argument('--only', default=None); ap.add_argument('--ctl_broken', default=None)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T0 = time.time(); prog = open(f'{a.out}/progress.txt', 'a')
def P(m):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {m}'; print(line, flush=True); prog.write(line + '\n'); prog.flush()
R = '/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad/'
tab = json.load(open(a.table)); tab_sha = hashlib.sha256(open(a.table, 'rb').read()).hexdigest()
M = a.M or int(tab['M']); BURN = a.burn if a.burn is not None else int(tab['burn']); MEAS = a.measure if a.measure is not None else int(tab['measure'])
NCTL = a.control_blocks if a.control_blocks is not None else int(tab['control_blocks']); DTAU = float(tab['dtau'])
smoke = any(v is not None for v in (a.M, a.burn, a.measure, a.control_blocks, a.only, a.ctl_broken))
chains = [c for c in tab['chains'] if a.only is None or c['id'] in a.only.split(',')]; assert chains, 'no chain selected'
import jax
if 'cuda' not in str(jax.devices()[0]).lower() and not os.environ.get('G8_ALLOW_CPU'): sys.exit(f'sep_prod: no CUDA device ({jax.devices()[0]})')
import cnn_sign as CS, providers as PV, engine as E, analysis as AN
amp = CS.AmpCache(chunk=a.chunk, cap=300000); n3 = CS.NetSign(R + 'r3/fit/net_8192.pkl', 'n3'); assert n3.sha == tab['n3_sha256'], 'n3 sha mismatch'
n4 = None
if any(c['arm'] == 'stored4' for c in chains):
    p4 = a.net4 or open('/scratch/a/A.Otaifi/ansatz_runs/sign_bad/r4/final_net.txt').read().split()[0]; n4 = CS.NetSign(p4, 'n4'); assert smoke or n4.sha == tab['n4_sha256'], 'n4 sha mismatch'
def mk(mode, nets, broken=None):
    g, _ = CS.build(mode, amp, nets, broken=broken, wchunk=a.chunk); return PV.ConfigGuide(g, guide_id=f'{mode}:{nets.name}:{broken}'), g
def load_x0(c):
    f = os.path.join(a.x0dir, c['x0_file']); assert hashlib.sha256(open(f, 'rb').read()).hexdigest() == c['x0_sha256'], f'X0 sha mismatch {f}'
    X = np.load(f).astype(np.int8)[:M]; assert X.shape == (M, 64); return X
hdr = dict(table=a.table, table_sha256=tab_sha, M=M, burn=BURN, measure=MEAS, dtau=DTAU, smoke=smoke, device=str(jax.devices()[0]), job=os.environ.get('SLURM_JOB_ID'), n3=(n3.path, n3.sha),
           n4=(n4.path, n4.sha) if n4 else None, code_sha256={f: hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest() for f in ('engine.py', 'providers.py', 'analysis.py', 'cnn_sign.py', 'sep_prod.py')})
P(f'sep_prod start table {tab_sha[:12]} chains {[c["id"] for c in chains]} burn {BURN} measure {MEAS} M {M} control_blocks {NCTL} smoke={smoke} device {jax.devices()[0]}')
# ---- control A = A (every start; also the warm-up)
X0c = load_x0(chains[0]); pA, _ = mk('stored', n3)
if a.ctl_broken == 'shuffle':
    g_, _ = CS.build('stored', amp, CS.shuffle_net(n3, 1), wchunk=a.chunk); pA2 = PV.ConfigGuide(g_, guide_id='stored:shuffled')   # broken control: shuffled-weight n_3 as B; must abort (exit 3)
else: pA2, _ = mk('stored', n3)
amp.reset_counts()
o = E.run(pA, pA2, X0c, X0c, mode='coupled', dtau=DTAU, measure_blocks=NCTL, pair_seed=int(tab['control_seed']), NS=64, progress_every=1, log=lambda m: P('ctl ' + m))
s = o['series']; ctl = dict(blocks=int(len(s['delta'])), max_abs_delta=float(np.max(np.abs(s['delta']))), same_post_min=float(np.min(s['same_post'])), A_only=float(np.nansum(s['A_only'])),
                            B_only=float(np.nansum(s['B_only'])), shared=float(np.nansum(s['shared'])), sec_per_block=o['wall'] / len(s['delta']), broken=a.ctl_broken)
ctl['passed'] = bool(ctl['max_abs_delta'] == 0 and ctl['same_post_min'] == 1 and ctl['A_only'] + ctl['B_only'] == 0 and ctl['shared'] > 0 and ctl['blocks'] == NCTL)
json.dump(dict(**hdr, control=ctl), open(f'{a.out}/control.json', 'w'), indent=1, default=float)
P(f'CONTROL A=A passed={ctl["passed"]} blocks {ctl["blocks"]} max|delta| {ctl["max_abs_delta"]} same_post_min {ctl["same_post_min"]} A_only+B_only {ctl["A_only"] + ctl["B_only"]}')
if not ctl['passed']:
    P('CONTROL FAILED: Delta != 0 for A = A, ABORT'); json.dump(dict(**hdr, control=ctl, done=False, aborted='control failed'), open(f'{a.out}/result.json', 'w'), indent=1, default=float); sys.exit(3)
# ---- chains
def chain_stats(c):
    z = np.load(f'{a.out}/chain_{c["id"]}.npz'); m = ~z['burn']; eA, eB = z['emix_A'][m], z['emix_B'][m]
    ps = AN.pair_stats(eA, eB, NS=64); ps.update(id=c['id'], arm=c['arm'], seed=c['seed'], x0_file=c['x0_file'], blocks_measured=int(m.sum()), matched_time=float(z['matched_time'][m].sum()),
        sep_rate_per_tau=float(z['sep_flux'][m].sum() / max(z['matched_time'][m].sum(), 1e-300)), alpha_mean=float(z['alpha'][m].mean()), same_post_mean=float(z['same_post'][m].mean()))
    return ps
def arm_est(ch):
    d = np.array([c['delta_site'] for c in ch]); sg = np.array([c['sigma_site'] for c in ch]); n = len(d)
    e_pool = float(np.sqrt((sg ** 2).sum()) / n); e_sc = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return dict(n_chains=n, delta_site=float(d.mean()), G=float(-d.mean()), err_pooled_longrun=e_pool, err_scatter=e_sc, err=float(max(e_pool, e_sc or 0.0)), rho=[c['rho'] for c in ch], per_chain_G=[float(-x) for x in d],
                chi2_vs_mean_with_sigma_c=float((((d - d.mean()) / sg) ** 2).sum()))
def write_result(done):
    ids = [c['id'] for c in chains if os.path.exists(f'{a.out}/chain_{c["id"]}.done')]; ch = [chain_stats(c) for c in chains if c['id'] in ids]
    r = dict(**hdr, control=ctl, chains_done=ids, chains_total=len(chains), per_chain=ch, done=bool(done), sec=time.time() - T0, arms={})
    for arm in ('exact', 'stored4'):
        cc = [c for c in ch if c['arm'] == arm]
        if cc: r['arms'][arm] = arm_est(cc)
    if 'exact' in r['arms'] and 'stored4' in r['arms']:
        ge, gs = r['arms']['exact'], r['arms']['stored4']; f = gs['G'] / ge['G'] if ge['G'] != 0 else float('nan')
        r['f'] = dict(value=float(f), err=float(abs(f) * np.hypot(gs['err'] / gs['G'], ge['err'] / ge['G'])) if ge['G'] and gs['G'] else None, G_exact=ge['G'], err_exact=ge['err'], G_stored=gs['G'], err_stored=gs['err'],
                      G_exact_sigma=float(ge['G'] / ge['err']), G_diff=float(ge['G'] - gs['G']), G_diff_err=float(np.hypot(ge['err'], gs['err'])), note='independent arms; f error by linear propagation (unreliable if G_stored/err is small)')
    tmp = f'{a.out}/result.json.tmp'; json.dump(r, open(tmp, 'w'), indent=1, default=float); os.replace(tmp, f'{a.out}/result.json'); return r
for c in chains:
    cid = c['id']; base = f'{a.out}/chain_{cid}'
    if os.path.exists(base + '.done'): P(f'chain {cid} already .done, skipped'); continue
    X0 = load_x0(c); amp.reset_counts(); pA, _ = mk('stored', n3); pB, gB = mk('exact', n3) if c['arm'] == 'exact' else mk('stored', n4)
    P(f'chain {cid} arm {c["arm"]} start x0 {c["x0_file"]} seed {c["seed"]}'); t1 = time.time()
    o = E.run(pA, pB, X0, X0, mode='coupled', dtau=DTAU, burn_blocks=BURN, measure_blocks=MEAS, pair_seed=int(c['seed']), NS=64, progress_every=1, log=lambda m, cid=cid: P(f'chain {cid} {m}'))
    s = o['series']; np.savez(base + '.tmp.npz', **{k: s[k] for k in ('emix_A', 'emix_B', 'delta', 'alpha', 'sep_flux', 'matched_time', 'same_post', 'burn', 'shared', 'A_only', 'B_only')}); os.replace(base + '.tmp.npz', base + '.npz')
    json.dump(dict(id=cid, arm=c['arm'], seed=c['seed'], x0_file=c['x0_file'], wall=time.time() - t1, st={k: float(v) for k, v in gB.st.items()}, cache=dict(hit=amp.hit, miss=amp.miss)), open(base + '.json', 'w'), default=float)
    open(base + '.done', 'w').write(time.strftime('%Y-%m-%dT%H:%M:%S') + '\n')
    r = write_result(False); cs = r['per_chain'][-1]; P(f'CHAIN_DONE {cid} delta/site {cs["delta_site"]:+.3e} sigma {cs["sigma_site"]:.2e} rho {cs["rho"]:.3f} wall {time.time() - t1:.0f}s')
r = write_result(len(chains) == sum(os.path.exists(f'{a.out}/chain_{c["id"]}.done') for c in chains))
P(f'SEP_PROD_DONE done={r["done"]} chains {len(r["chains_done"])}/{r["chains_total"]} ' + ' '.join(f'G_{k} {v["G"]:+.3e} err {v["err"]:.2e}' for k, v in r['arms'].items()))
