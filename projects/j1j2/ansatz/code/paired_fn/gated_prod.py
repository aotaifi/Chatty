#!/usr/bin/env python3
"""gated_prod.py OUT --table chains.json --x0dir DIR --consts consts_s306_p1.json   (G(-5) production driver, 2026-11-13; PLAN_sign.md "G(-5) PRODUCTION")
One coupled paired run per chain: A = s306 (probe8 guide == GatedGuide gate -inf), B = GatedGuide(gate -5), shared per-configuration cache, same engine as gated_probe.py.
Chain table (JSON): {"gate": -5, "dtau": 0.1, "M": 1024, "burn": 20, "measure": 150, "control_blocks": 5, "chains": [{"id": 0, "x0_file": "X0_B0s_0.npy", "x0_sha256": ..., "seed": ...}, ...]}
(X0 is looked up as x0dir/x0_file and its sha256 is checked, so the same table serves both clusters; all M walkers of the file are used.)
Start: d_gate = -inf A = B control for control_blocks blocks (Delta must be exactly 0, same_post 1, no A_only/B_only), else abort (exit 3).
Per chain: chain_<id>.npz (+ chain_<id>.json gate stats, then chain_<id>.done); chains with .done are skipped on restart (resume = chain granularity).
At the end result.json (done=true): per-chain Delta/site + long-run error (analysis.pair_stats), 4-chain mean, error from chain scatter and pooled long-run error (both; 'err' = larger), rho per chain.
Overrides for tests only (result.json then carries smoke=true): --burn --measure --M --only ID --control_blocks --ctl_broken (control run with a broken B; must abort)."""
import argparse, hashlib, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
BASE = os.environ.get('S306_CODE', '/project/theorie/a/A.Otaifi/runs/j1j2/base_s306_p0/code'); sys.path.insert(0, BASE)
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--table', required=True); ap.add_argument('--x0dir', required=True); ap.add_argument('--consts', required=True)
ap.add_argument('--wchunk', type=int, default=128); ap.add_argument('--burn', type=int); ap.add_argument('--measure', type=int); ap.add_argument('--M', type=int)
ap.add_argument('--control_blocks', type=int); ap.add_argument('--only', type=int); ap.add_argument('--ctl_broken', default=None); ap.add_argument('--ctl_hash_p', type=float, default=1e-3)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T0 = time.time(); prog = open(f'{a.out}/progress.txt', 'a')
def P(m):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {m}'; print(line, flush=True); prog.write(line + '\n'); prog.flush()
tab = json.load(open(a.table)); tab_sha = hashlib.sha256(open(a.table, 'rb').read()).hexdigest()
GATE = float(tab['gate']); DTAU = float(tab['dtau']); M = a.M or int(tab['M']); BURN = a.burn if a.burn is not None else int(tab['burn'])
MEAS = a.measure if a.measure is not None else int(tab['measure']); NCTL = a.control_blocks if a.control_blocks is not None else int(tab['control_blocks'])
smoke = any(v is not None for v in (a.burn, a.measure, a.M, a.only, a.control_blocks, a.ctl_broken))
chains = [c for c in tab['chains'] if a.only is None or c['id'] == a.only]; assert chains, 'no chain selected'
import jax
if 'cuda' not in str(jax.devices()[0]).lower() and not os.environ.get('G8_ALLOW_CPU'): sys.exit(f'gated_prod: no CUDA device ({jax.devices()[0]})')
import gated_sign as GS, providers as PV, engine as E, analysis as AN
GS.CONSTS = a.consts                                    # consts_s306_p1.json path differs per cluster; sha is still checked by read_sign_constants
cache = GS.Cache(cap=300000)
def mk(gate, broken=None, hash_p=1e-3): return GS.build(gate, cache, broken=broken, wchunk=a.wchunk, hash_p=hash_p)
def prov(g, tag, rel=False): return PV.ConfigGuide(g, guide_id=tag, sign_fn=(g.rel_sign if rel else (lambda X: np.ones(len(X)))))
def load_x0(c):
    f = os.path.join(a.x0dir, c['x0_file']); assert hashlib.sha256(open(f, 'rb').read()).hexdigest() == c['x0_sha256'], f'X0 sha mismatch {f}'
    X = np.load(f).astype(np.int8)[:M]; assert X.shape == (M, 64); return X
info_hdr = dict(table=a.table, table_sha256=tab_sha, gate=GATE, dtau=DTAU, M=M, burn=BURN, measure=MEAS, smoke=smoke, device=str(jax.devices()[0]), job=os.environ.get('SLURM_JOB_ID'),
                code_sha256={f: hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest() for f in ('engine.py', 'providers.py', 'analysis.py', 'gated_sign.py', 'gated_prod.py')})
P(f'gated_prod start table {tab_sha[:12]} gate {GATE} chains {[c["id"] for c in chains]} burn {BURN} measure {MEAS} M {M} control_blocks {NCTL} smoke={smoke} device {jax.devices()[0]}')
# ---- warm-up (compile) and control d_gate = -inf: A = B  (run at EVERY start, also on resume: it is the JIT warm-up and the integrity check)
gw, _ = mk(-math.inf); X0c = load_x0(chains[0]); gw.rows(X0c[:8]); cache.d.clear(); P('warm-up done')
gA, _ = mk(-math.inf); gB, _ = mk(-math.inf, a.ctl_broken, a.ctl_hash_p); cache.reset_counts()
o = E.run(prov(gA, 'A_s306'), prov(gB, 'B_ctl'), X0c, X0c, mode='coupled', dtau=DTAU, measure_blocks=NCTL, pair_seed=int(tab['control_seed']), NS=64, progress_every=1, log=lambda m: P('ctl ' + m))
s = o['series']; ctl = dict(blocks=int(len(s['delta'])), max_abs_delta=float(np.max(np.abs(s['delta']))), same_post_min=float(np.min(s['same_post'])), A_only=float(np.nansum(s['A_only'])),
                            B_only=float(np.nansum(s['B_only'])), shared=float(np.nansum(s['shared'])), sec_per_block=o['wall'] / len(s['delta']), broken=a.ctl_broken)
ctl['passed'] = bool(ctl['max_abs_delta'] == 0 and ctl['same_post_min'] == 1 and ctl['A_only'] + ctl['B_only'] == 0 and ctl['shared'] > 0 and ctl['blocks'] == NCTL)
json.dump(dict(**info_hdr, control=ctl), open(f'{a.out}/control.json', 'w'), indent=1, default=float)
P(f'CONTROL gate -inf A=B passed={ctl["passed"]} blocks {ctl["blocks"]} max|delta| {ctl["max_abs_delta"]} same_post_min {ctl["same_post_min"]} A_only+B_only {ctl["A_only"] + ctl["B_only"]}')
if not ctl['passed']:
    P('CONTROL FAILED: Delta != 0 at d_gate = -inf, ABORT'); json.dump(dict(**info_hdr, control=ctl, done=False, aborted='control failed'), open(f'{a.out}/result.json', 'w'), indent=1, default=float); sys.exit(3)
# ---- chains
def chain_stats(c):
    z = np.load(f'{a.out}/chain_{c["id"]}.npz'); m = ~z['burn']; eA, eB = z['emix_A'][m], z['emix_B'][m]
    ps = AN.pair_stats(eA, eB, NS=64); ps.update(id=c['id'], seed=c['seed'], x0_file=c['x0_file'], blocks_measured=int(m.sum()), matched_time=float(z['matched_time'][m].sum()),
        sep_rate_per_tau=float(z['sep_flux'][m].sum() / max(z['matched_time'][m].sum(), 1e-300)), alpha_mean=float(z['alpha'][m].mean()), delta_site_binning_levels=AN.binning(eB - eA))
    return ps
def write_result(done):
    ids = [c['id'] for c in chains if os.path.exists(f'{a.out}/chain_{c["id"]}.done')]; ch = [chain_stats(c) for c in chains if c['id'] in ids]
    r = dict(**info_hdr, control=ctl, chains_done=ids, chains_total=len(chains), per_chain=ch, done=bool(done), sec=time.time() - T0)
    if ch:
        d = np.array([c['delta_site'] for c in ch]); sg = np.array([c['sigma_site'] for c in ch]); n = len(d)
        r['mean_delta_site'] = float(d.mean()); r['err_pooled_longrun'] = float(np.sqrt((sg ** 2).sum()) / n)
        r['err_scatter'] = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else None
        r['err'] = float(max(r['err_pooled_longrun'], r['err_scatter'] or 0.0)); r['err_note'] = 'err = larger of err_pooled_longrun (sqrt(sum sigma_c^2)/n) and err_scatter (std(chain Delta)/sqrt(n))'
        r['chi2_vs_mean_with_sigma_c'] = float((((d - d.mean()) / sg) ** 2).sum()); r['rho_per_chain'] = [c['rho'] for c in ch]
    tmp = f'{a.out}/result.json.tmp'; json.dump(r, open(tmp, 'w'), indent=1, default=float); os.replace(tmp, f'{a.out}/result.json'); return r
for c in chains:
    cid = c['id']; base = f'{a.out}/chain_{cid}'
    if os.path.exists(base + '.done'): P(f'chain {cid} already .done, skipped'); continue
    X0 = load_x0(c); cache.d.clear(); cache.reset_counts(); gA, _ = mk(-math.inf); gB, infoB = mk(GATE)
    P(f'chain {cid} start x0 {c["x0_file"]} seed {c["seed"]}'); t1 = time.time()
    o = E.run(prov(gA, 'A_s306'), prov(gB, f'B_gate{GATE:g}', rel=True), X0, X0, mode='coupled', dtau=DTAU, burn_blocks=BURN, measure_blocks=MEAS, pair_seed=int(c['seed']), NS=64,
              progress_every=1, log=lambda m, cid=cid: P(f'chain {cid} {m}'))
    s = o['series']; np.savez(base + '.tmp.npz', **{k: s[k] for k in ('emix_A', 'emix_B', 'delta', 'alpha', 'sep_flux', 'matched_time', 'same_post', 'burn')})
    os.replace(base + '.tmp.npz', base + '.npz')
    json.dump(dict(id=cid, seed=c['seed'], x0_file=c['x0_file'], wall=time.time() - t1, gate_stats={k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in gB.st.items()},
                   cache=dict(hit=cache.hit, miss=cache.miss)), open(base + '.json', 'w'), default=float)
    open(base + '.done', 'w').write(time.strftime('%Y-%m-%dT%H:%M:%S') + '\n')
    r = write_result(False); cs = r['per_chain'][-1]; P(f'CHAIN_DONE {cid} delta/site {cs["delta_site"]:+.3e} sigma {cs["sigma_site"]:.2e} rho {cs["rho"]:.3f} wall {time.time() - t1:.0f}s')
r = write_result(len(chains) == sum(os.path.exists(f'{a.out}/chain_{c["id"]}.done') for c in chains))
P(f'GATED_PROD_DONE done={r["done"]} chains {len(r["chains_done"])}/{r["chains_total"]} mean {r.get("mean_delta_site", float("nan")):+.3e} err {r.get("err", float("nan")):.2e}')
