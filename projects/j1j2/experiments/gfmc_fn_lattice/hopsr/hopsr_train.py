"""Hop-aware SR fine-tune: minimise E_1(theta) = E_VMC((1 + alpha (H - E)) v_theta), v = Re[w psi_P], alpha by Ritz, by stochastic
reconfiguration (natural gradient) with the sampled zero-variance gradient of hopsr_lib (x ~ |psi_P|^2, persistent Metropolis chains).

  python hopsr_train.py OUT SPEC.json
SPEC keys [defaults]: L [6], J2 [0.5], proj ['translations'], init (rbmpp params npz, required), cls ['all' | 'f' | 'rbm'],
  w [null = phase from samples at start, then fixed; or [re, im]], chains [4096], sweeps_per_step [1] (sample sets of all chains per step),
  sweep [N proposals per chain per sweep], therm [50], steps [100], lr [0.02] (lr_end, cosine), shift [1e-2] (shift_end),
  max_dnorm [0.1] (cap on sqrt(d^T S d)), chunk [512] (neighbour evaluations), jchunk [64] (jacobian), groups [32] (jackknife),
  ckpt_every [10], seed [0], maxsec [1e18] (stop cleanly after this many seconds), resume [false] (OUT/state.npz),
  objective ['E1' | 'E0' (control: plain E_v SR with the same machinery)].
S = Re cov(O, O) of psi_P's log-derivatives (the rbmpp_train.py metric), F = grad/2.  w and the class are fixed during the run.
Writes OUT/progress.txt (PROGRESS lines, last TRAIN_DONE), OUT/train.json (per step: E/N, E1/N +- jackknife, alpha, var/N, |d|_S, |grad|,
timings), params_XXXXX.npz, params_final.npz (rbmpp format), state.npz.
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from hopsr_lib import *

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
g = SPEC.get; T00 = time.time(); PROG = os.path.join(OUT, 'progress.txt')


def log(msg):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {msg}'; print(line, flush=True); open(PROG, 'a').write(line + '\n')


lat = Lattice(g('L', 6), g('J2', 0.5), g('proj', 'translations')); N = lat.N
z = np.load(g('init')); use_rbm = bool(z['rbm']) if 'rbm' in z.files else True
p = {k: jnp.asarray(z[k]) for k in KEYS}
flat, unrav = ravel_pytree(p); P = flat.size
mask = class_mask(p, g('cls', 'all'))
if not use_rbm: mask &= class_mask(p, 'f')
est = HopEstimator(lat, use_rbm, unrav, chunk=g('chunk', 512), jchunk=g('jchunk', 64))
sweep = make_sampler(lat, make_fns(lat, jnp.float32, use_rbm))
sr = make_sr(mask)
C = g('chains', 4096); nsw = g('sweep', N); K = g('sweeps_per_step', 1); G = g('groups', 32)
assert C % G == 0
log(f'PROGRESS hopsr_train L {lat.L} proj {lat.proj} images {lat.nI} params {P} class {g("cls", "all")} active {int(mask.sum())} chains {C} '
    f'x {K} sweeps/step objective {g("objective", "E1")} init {g("init")} jax {jax.__version__} {jax.devices()}')

STATE = os.path.join(OUT, 'state.npz'); F = os.path.join(OUT, 'train.json'); hist = []; it0 = 0
key = jax.random.PRNGKey(3000 + g('seed', 0))
if g('resume') and os.path.exists(STATE):
    zs = np.load(STATE); flat = jnp.asarray(zs['flat']); S = jnp.asarray(zs['S']); key = jnp.asarray(zs['key'], dtype=jnp.uint32)
    it0 = int(zs['it']); w = complex(zs['w']); E_ref = float(zs['E_ref'])
    hist = [h for h in json.load(open(F))['hist'] if h['it'] <= it0]; log(f'PROGRESS RESUMED from step {it0}')
else:
    S = random_configs(N, C, 9000 + g('seed', 0)); t = time.time()
    for _ in range(g('therm', 50)):
        key, k = jax.random.split(key); S, acc = sweep(to32(unrav(flat)), S, k, nsw)
    log(f'PROGRESS thermalised {g("therm", 50)} sweeps ({time.time() - t:.0f}s) acc {float(acc):.3f}')
    if g('w') is not None:
        w = complex(*g('w'))
    else:
        w, lam, imw = phase_w(est, flat, np.asarray(S).astype(np.float64))
        log(f'PROGRESS phase w {w:.6f} (lowest 2x2 eigenvalue/N {lam / N:.6f}, Im weight {imw:.3e}) from {C} samples')
    E_ref = None


def save(tag, fl):
    q = unrav(jnp.asarray(fl))
    np.savez(os.path.join(OUT, f'params_{tag}.npz'), **{k: np.asarray(v) for k, v in q.items()}, proj=lat.proj, rbm=use_rbm)


def save_state(it):
    tmp = os.path.join(OUT, 'state_tmp.npz')
    np.savez(tmp, flat=np.asarray(flat), S=np.asarray(S), key=np.asarray(key), it=it, w=w, E_ref=E_ref); os.replace(tmp, STATE)


if it0 == 0: save('00000', flat)
steps = g('steps', 100); lr0, lr1 = g('lr', 0.02), g('lr_end', g('lr', 0.02)); sh0, sh1 = g('shift', 1e-2), g('shift_end', g('shift', 1e-2))
res = dict(spec=SPEC, nparams=int(P), nactive=int(mask.sum()), images=lat.nI, w=[w.real, w.imag], hist=hist)
stop = 'steps'
for it in range(it0 + 1, steps + 1):
    ts = time.time()
    c = 0.5 * (1 - np.cos(np.pi * (it - 1) / max(1, steps - 1)))
    lr = lr0 + (lr1 - lr0) * c; sh = np.exp(np.log(sh0) + (np.log(sh1) - np.log(sh0)) * c)
    Xs = []
    for _ in range(K):
        key, k = jax.random.split(key); S, acc = sweep(to32(unrav(flat)), S, k, nsw); Xs.append(np.asarray(S).astype(np.float64))
    X = np.concatenate(Xs); tsm = time.time() - ts
    if E_ref is None:
        E_l, _ = est.at_y(flat, X[:min(len(X), 2048)]); E_ref = float(np.mean(E_l.real))
    r = est.estimate(flat, X, w, E_ref, groups=G)
    grad = r['grad'] if g('objective', 'E1') == 'E1' else r['grad_p0']
    t1 = time.time()
    dl, nrm, gn = sr(r['O'], jnp.asarray(grad), lr, sh, g('max_dnorm', 0.1)); dl.block_until_ready()
    flat = flat + dl; tsolve = time.time() - t1
    E_ref = float(r['E'])
    rec = dict(it=it, E=r['E'] / N, E_err=r['E_err'] / N, E1=r['E1'] / N, E1_err=r['E1_err'] / N, E1_gform=r['E1_gform'] / N, alpha=r['alpha'],
               alpha_err=r['alpha_err'], var=r['var'] / N, check_h2=r['check_h2'], acc=float(acc), dnorm=float(nrm), gnorm=float(gn),
               grad_relerr=float(np.linalg.norm(r['err'][mask]) / max(np.linalg.norm(r['grad'][mask]), 1e-300)),
               lr=lr, shift=sh, t_sample=tsm, t_x=r['t_x'], t_y=r['t_y'], t_red=r['t_red'], t_solve=tsolve, t_step=time.time() - ts,
               n_y=r['n_y'], ns=r['ns'], wall=time.time() - T00)
    hist.append(rec)
    log(f'PROGRESS it {it} E/N {rec["E"]:.7f} +- {rec["E_err"]:.1e} E1/N {rec["E1"]:.7f} +- {rec["E1_err"]:.1e} alpha {rec["alpha"]:.4f} '
        f'var/N {rec["var"]:.3e} chk {rec["check_h2"]:.3f} |d|_S {rec["dnorm"]:.3f} |F| {rec["gnorm"]:.2e} relerr(grad) {rec["grad_relerr"]:.2f} '
        f'acc {rec["acc"]:.2f} | {rec["t_step"]:.1f}s (sample {tsm:.1f} x {r["t_x"]:.1f} y {r["t_y"]:.1f} [{r["n_y"]} evals] red {r["t_red"]:.1f} solve {tsolve:.1f})')
    if it % g('ckpt_every', 10) == 0 or it == steps:
        save(f'{it:05d}', flat); json.dump(res, open(F, 'w'), default=float); save_state(it)
    if not np.isfinite(rec['E1']):
        stop = 'non-finite'; log('PROGRESS non-finite E1, stop'); break
    if time.time() - T00 > g('maxsec', 1e18):
        stop = 'maxsec'; save(f'{it:05d}', flat); save_state(it); log(f'PROGRESS MAXSEC stop after step {it}'); break
save('final', flat); res['sec'] = time.time() - T00; res['stop'] = stop
json.dump(res, open(F, 'w'), default=float)
hh = hist[-min(len(hist), 10):]
log(f'TRAIN_DONE stop {stop} steps {len(hist)} last-10 mean E1/N {np.mean([h["E1"] for h in hh]):.7f} '
    f'median s/step {np.median([h["t_step"] for h in hist]):.2f} wall {res["sec"]:.0f}s')
