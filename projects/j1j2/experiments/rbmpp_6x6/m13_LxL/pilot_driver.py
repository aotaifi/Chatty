"""Timing pilot for the L x L M1 measurement (RBM+PP translations x spin flip): SR training steps, eval timing, VMC moments, GFMC p = 0 / 1.
Same script on every machine (Paderborn H100, theorie A40); every stage has its own wall cap, a stage that fails does not stop the rest.

  python pilot_driver.py OUT L --stages train,timing,vmc,gfmc0,gfmc1 [--params P.npz] [--sr 2048:40,8192:6] [--guide_steps 40]
                         [--vmc_sec 150] [--g0_sec 120] [--g1_sec 240] [--g1_M 32] [--g0_M 256]

OUT/summary.json collects: SR s/step (median, sample part) per chain count; ms per eval (chunk scan); VMC s per sweep, Var(E_loc)/N,
E/N, alpha; GFMC s per walker-tau for p = 0 and p = 1 with sigma_walker/N (std of the walker-weighted E_L estimator per block times
sqrt(walkers)/N is NOT used: sigma_walker is taken from the std over blocks of emix times sqrt(M dtau tau_int-free) -- see 'sigw').
Code files are expected next to this script's copy in OUT/code (the sbatch copies them).
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('L', type=int)
ap.add_argument('--stages', default='train,timing,vmc,gfmc0,gfmc1')
ap.add_argument('--params', default=None)
ap.add_argument('--sr', default='2048:40,8192:6'); ap.add_argument('--guide_steps', type=int, default=40)
ap.add_argument('--vmc_sec', type=float, default=150); ap.add_argument('--vmc_chains', type=int, default=1024)
ap.add_argument('--g0_sec', type=float, default=120); ap.add_argument('--g0_M', type=int, default=256)
ap.add_argument('--g1_sec', type=float, default=240); ap.add_argument('--g1_M', type=int, default=32)
ap.add_argument('--cap_vmc_extra', type=float, default=1500, help='wall allowance for the VMC stage beyond --vmc_sec (stage A and B0 are not capped by maxsec_seed)')
ap.add_argument('--chunk', type=int, default=256); ap.add_argument('--train_chunk', type=int, default=512)
ap.add_argument('--timing_chunks', default='64,128,256,512'); ap.add_argument('--stageA_chains', type=int, default=256); ap.add_argument('--vmc_seed_sec', type=float, default=None)
ap.add_argument('--cap_train', type=float, default=1e9); ap.add_argument('--therm', type=int, default=20, help='thermalisation sweeps before SR (timing runs: few)')
a = ap.parse_args()
OUT = os.path.abspath(a.out); os.makedirs(OUT, exist_ok=True)
CODE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
SUMF = os.path.join(OUT, 'summary.json')
S = json.load(open(SUMF)) if os.path.exists(SUMF) else {}
S['L'] = a.L


def dump():
    json.dump(S, open(SUMF, 'w'), indent=1, default=float)


def log(m):
    print(time.strftime('[%H:%M:%S]'), m, flush=True)


def run(cmd, cap, logf):
    t0 = time.time()
    try:
        with open(logf, 'a') as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=cap, cwd=CODE)
        return r.returncode, time.time() - t0
    except subprocess.TimeoutExpired:
        return 'timeout', time.time() - t0


params = a.params
stages = a.stages.split(',')
# --------------------------------------------------------------------------- SR training steps
if 'train' in stages:
    S['train'] = {}
    for item in a.sr.split(','):
        C, steps = [int(x) for x in item.split(':')]
        d = os.path.join(OUT, f'train_c{C}')
        spec = dict(L=a.L, J2=0.5, proj='translations', rbm=True, M=16, seed=7, steps=steps, chains=C, lr=0.05, lr_end=0.05, shift=1e-3,
                    max_dnorm=0.3, chunk=a.train_chunk, ckpt_every=steps, therm=a.therm, eval_samples=0)
        os.makedirs(d, exist_ok=True); json.dump(spec, open(os.path.join(d, 'spec.json'), 'w'))
        log(f'train chains {C} steps {steps}')
        rc, sec = run([PY, 'rbmpp_train.py', d, os.path.join(d, 'spec.json')], a.cap_train, os.path.join(d, 'stdout.txt'))
        try:
            h = json.load(open(os.path.join(d, 'train.json')))['hist']
        except Exception:
            h = []
        r = dict(rc=rc, wall=sec, chains=C, steps_done=len(h))
        if h:
            ts = np.array([x['t_step'] for x in h]); tm = np.array([x['t_sample'] for x in h])
            r.update(t_step_first=float(ts[0]), t_step_median=float(np.median(ts[1:])) if len(ts) > 1 else float(ts[0]),
                     t_sample_median=float(np.median(tm[1:])) if len(tm) > 1 else float(tm[0]), E_last=h[-1]['E'], var_last=h[-1]['var'],
                     acc_last=h[-1]['acc'])
        S['train'][str(C)] = r; dump(); log(json.dumps(r))
        if params is None and rc == 0 and os.path.exists(os.path.join(d, 'params_final.npz')) and C == int(a.sr.split(',')[0].split(':')[0]):
            params = os.path.join(d, 'params_final.npz')
if params is None:
    cand = os.path.join(OUT, 'train_c%s' % a.sr.split(',')[0].split(':')[0], 'params_final.npz')
    params = cand if os.path.exists(cand) else None
S['params'] = params; dump()
if params is None:
    log('no params, stop'); sys.exit(0)
# --------------------------------------------------------------------------- eval timing
if 'timing' in stages:
    rc, sec = run([PY, 'timing8.py', os.path.join(OUT, 'timing.json'), params, str(a.L), 'translations'] + a.timing_chunks.split(','), 900, os.path.join(OUT, 'timing.txt'))
    try: S['timing'] = json.load(open(os.path.join(OUT, 'timing.json')))
    except Exception: S['timing'] = dict(rc=rc)
    dump(); log('timing ' + json.dumps(S['timing']))
# --------------------------------------------------------------------------- VMC moments
if 'vmc' in stages:
    d = os.path.join(OUT, 'vmc')
    rc, sec = run([PY, 'vmc_moments.py', d, '--params', params, '--L', str(a.L), '--proj', 'translations', '--chains', str(a.vmc_chains), '--sweeps', '1000',
                   '--seeds', '1', '--stageA_sweeps', '6', '--stageA_chains', str(a.stageA_chains), '--therm', '8', '--chunk', str(a.chunk), '--maxsec_seed', str(a.vmc_seed_sec if a.vmc_seed_sec is not None else a.vmc_sec)],
                  a.vmc_sec + a.cap_vmc_extra, os.path.join(d + '.txt'))
    r = dict(rc=rc, wall=sec)
    try:
        v = json.load(open(os.path.join(d, 'vmc_s1.json'))); r.update(info=v['info'], per_site=v['per_site'], alpha=v['alpha'], check_h2=v['check_h2'])
        r['sec_per_sweep'] = v['info']['sec'] / max(1, v['info']['sweeps_done'])
    except Exception as e:
        r['err'] = str(e)
    S['vmc'] = r; dump(); log('vmc ' + json.dumps({k: r[k] for k in r if k in ('rc', 'wall', 'sec_per_sweep', 'err')}))
# --------------------------------------------------------------------------- GFMC
for st, p, M, cap in (('gfmc0', 0, a.g0_M, a.g0_sec), ('gfmc1', 1, a.g1_M, a.g1_sec)):
    if st not in stages: continue
    d = os.path.join(OUT, st)
    cons = os.path.join(OUT, 'vmc', 'consts.json')
    if not os.path.exists(cons):
        S[st] = dict(err='no consts.json (vmc stage did not finish)'); dump(); continue
    rc, sec = run([PY, 'gfmc_run.py', d, '--params', params, '--consts', cons, '--p', str(p), '--L', str(a.L), '--proj', 'translations', '--M', str(M),
                   '--dtau', '0.1', '--blocks', '100000', '--burn', '3', '--seed', '1', '--chunk', str(a.chunk), '--maxsec', str(cap), '--ckpt_every', '1000'],
                  cap + 900, d + '.txt')
    r = dict(rc=rc, wall=sec, M=M, dtau=0.1)
    try:
        b = np.load(os.path.join(d, 'blocks.npz')); N = a.L * a.L
        sec_b = b['sec']; em = b['emix'] / N; ev = b['elvar'] / N ** 2 * N   # elvar/N: per-site variance of E_L weighted
        r.update(blocks=int(len(sec_b)), sec_per_block_median=float(np.median(sec_b[1:] if len(sec_b) > 1 else sec_b)),
                 evals_per_block=float(np.mean(b['evals'])), hops_per_block=float(np.mean(b['hops'])), ess_mean=float(np.mean(b['ess'])),
                 elvar_per_site=float(np.mean(b['elvar']) / N), emix_mean=float(np.mean(em[3:])) if len(em) > 6 else None,
                 emix_std_blocks=float(np.std(em[3:], ddof=1)) if len(em) > 8 else None)
        r['sec_per_walker_tau'] = r['sec_per_block_median'] / (M * 0.1)
        if r['emix_std_blocks'] is not None:   # per-walker-block spread, per site: block error of M walkers x dtau -> sigma_w = std * sqrt(M)  (units per site, tau_int not removed)
            r['sigma_walker_per_site_raw'] = r['emix_std_blocks'] * np.sqrt(M)
    except Exception as e:
        r['err'] = str(e)
    S[st] = r; dump(); log(st + ' ' + json.dumps(r))
log('PILOT_DONE')
