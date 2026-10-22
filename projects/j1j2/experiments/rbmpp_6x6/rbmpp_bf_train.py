"""SR (stochastic reconfiguration) VMC training of PP + 3x3 local backflow (rbmpp_bf_lib, the cx_t1s 'ppbf' network), Test B0.

  python rbmpp_bf_train.py OUT SPEC.json
Adapted from rbmpp_train.py (same SR step: diagonally rescaled S + shift, Cholesky, cap on sqrt(d^T S d), cosine lr / shift
schedule, Metropolis exchange sampler with the Hastings factor).  Differences: real wave function (O, E_loc real), theta =
[f, u1, u2] (1944 entries, f diagonal frozen), local energy by brute force (no rank-2 update with backflow), resumable.
SPEC keys (defaults in brackets):
  proj ['full'], init [npz of rpT1_pp params (key f) or a theta .npy; u = 0 when the npz has no u1/u2 keys], steps [3000: horizon of
  the lr schedule and hard cap], chains [2048], sweep [36 proposals per chain between samples], therm [20], lr [0.05], lr_end [0.01],
  shift [1e-3], max_dnorm [0.3], chunk [64: configs per eloc call], jchunk [128], ckpt_every [50], seed [1],
  wall_sec [null: stop training (budget out, not converged) when this process has run that long],
  plateau_window [100], plateau_span [200], plateau_tol [1e-6], plateau_min_step [300]:
     B0 rule: the mean of E/N over the last plateau_window steps changes by < plateau_tol (per site) over plateau_span steps
     -> converged.  Evaluated every step once it >= plateau_min_step.  The record also stores se_diff, the statistical error of that
     difference (from the per-step errors): at 2048 chains se_diff is ~5e-6 (full) to 1e-5, i.e. above plateau_tol = 1e-6, so a pass
     can be a noise coincidence; plateau_hold [1] = number of consecutive steps the criterion must hold.
  eloc_dtype ['float32': rms difference to float64 1.2e-7 per site on 64 sampled configs, bias -6e-9, vs a per-step statistical error of 5e-5 to 1e-4], sampler_dtype ['float32'], bond_trim [true: only the anti-aligned bonds, kmax rounded up to 8],
  max_steps_this_run [null: stop after that many steps in this process (smoke tests)].
Files in OUT: params_XXXXX.npz (theta, f, u1, u2, proj) every ckpt_every steps and params_final.npz; state.npz (theta, chains S,
PRNG key, step; written every ckpt_every steps, atomically) -> rerunning the same command resumes; train.json (spec, hist per step:
E/N, err, var/N, acc, |d|_S, wall, timings; plateau record); progress.txt (one PROGRESS line per 10 steps).
"""
import json, os, sys, time
import numpy as np
import jax
import jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rbmpp_bf_lib import (Lattice, make_bf_fns, make_sampler, random_configs, to32, theta_from_f, active_mask, N, NTH)
jax.config.update("jax_enable_x64", True)

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
T00 = time.time(); g = SPEC.get


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


def progress(msg):
    with open(os.path.join(OUT, 'progress.txt'), 'a') as fh: fh.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}\n')
    log(msg)


# ---------------------------------------------------------------- code guard (sha256 of the files run, as the cx_* scripts)
import hashlib
CODE_FILES = ('rbmpp_bf_train.py', 'rbmpp_bf_lib.py', 'rbmpp_lib.py')
SHA_RUN = {fn: hashlib.sha256(open(os.path.join(HERE, fn), 'rb').read()).hexdigest() for fn in CODE_FILES}
SHA_EXP = g('code_sha256')
if SHA_EXP is not None:
    bad = [fn for fn in CODE_FILES if SHA_EXP.get(fn) != SHA_RUN[fn]]
    if bad:
        progress(f'CODE DRIFT: sha256 mismatch for {bad}; exit 3'); sys.exit(3)

lat = Lattice(6, 0.5, g('proj', 'full'))
mask = jnp.asarray(active_mask())
EDT = jnp.float64 if g('eloc_dtype', 'float32') == 'float64' else jnp.float32
SDT = jnp.float64 if g('sampler_dtype', 'float32') == 'float64' else jnp.float32
fns = make_bf_fns(lat, EDT)
fns_s = make_bf_fns(lat, SDT)
sweep = make_sampler(lat, fns_s)
cast = (lambda q: q) if SDT == jnp.float64 else to32
nsw = g('sweep', N)
CH = g('chunk', 64); JCH = g('jchunk', 128)
BI = np.asarray(lat.BI); BJ = np.asarray(lat.BJ)

jac_v = jax.jit(jax.vmap(jax.grad(lambda th, s: make_bf_fns(lat, jnp.float64)['logpsi'](th, s)), in_axes=(None, 0)))
_el_cache = {}


def eloc_fn(kmax):
    if kmax not in _el_cache:
        if kmax is None: f = jax.vmap(fns['eloc'], in_axes=(None, 0))
        else: f = jax.vmap(lambda th, s: fns['eloc_bonds'](th, s, kmax), in_axes=(None, 0))
        _el_cache[kmax] = jax.jit(f)
    return _el_cache[kmax]


def pad_to(Sx, ch):
    m = Sx.shape[0]
    return (jnp.concatenate([Sx, jnp.repeat(Sx[:1], ch - m, 0)]) if m < ch else Sx), m


def eloc_all(theta, S):
    """E_loc, log|psi| for all chains (chunks; per chunk only the anti-aligned bonds, kmax rounded up to 8)."""
    Sh = np.asarray(S); outE = []; outL = []
    for i in range(0, Sh.shape[0], CH):
        sc = Sh[i:i + CH]; m = sc.shape[0]
        if m < CH: sc = np.concatenate([sc, np.repeat(sc[:1], CH - m, 0)])
        kmax = None
        if g('bond_trim', True):
            nmax = int(np.max(np.sum(sc[:, BI] != sc[:, BJ], 1)))
            kmax = min(lat.NB, 8 * ((nmax + 7) // 8))
            if kmax >= lat.NB: kmax = None
        E, l = eloc_fn(kmax)(theta, jnp.asarray(sc))
        outE.append(E[:m]); outL.append(l[:m])
    return jnp.concatenate(outE), jnp.concatenate(outL)


def jac_all(theta, S):
    out = []
    for i in range(0, S.shape[0], JCH):
        s_, m = pad_to(S[i:i + JCH], JCH); out.append(jac_v(theta, s_)[:m])
    return jnp.concatenate(out)


@jax.jit
def sr_solve(O, El, lr, shift, maxn):
    """real SR: S = cov(O, O), F = cov(O, E_loc) (rbmpp_train.sr_solve with O, E_loc real)."""
    Om = O - jnp.mean(O, 0); Ec = El - jnp.mean(El); ns = O.shape[0]
    S = Om.T @ Om / ns; F = Om.T @ Ec / ns
    S = jnp.where(mask[:, None] & mask[None, :], S, 0.0); F = jnp.where(mask, F, 0.0)
    d = jnp.maximum(jnp.diag(S), 1e-12); sq = jnp.sqrt(d)
    Ss = S / (sq[:, None] * sq[None, :]) + shift * jnp.eye(S.shape[0])
    y = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(Ss), F / sq)
    dl = jnp.where(mask, -lr * y / sq, 0.0)
    nrm = jnp.sqrt(jnp.maximum(dl @ S @ dl, 0.0))
    dl = dl * jnp.minimum(1.0, maxn / jnp.maximum(nrm, 1e-30))
    return dl, nrm, jnp.sqrt(jnp.sum(F * F))


def save_params(tag, th):
    th = np.asarray(th)
    np.savez(os.path.join(OUT, f'params_{tag}.npz'), theta=th, f=th[:N * N].reshape(N, N), u1=th[N * N:].reshape(2, 9, N)[0],
             u2=th[N * N:].reshape(2, 9, N)[1], proj=lat.proj, rbm=False)


def atomic(path, writer):
    tmp = path + '.tmp'; writer(tmp); os.replace(tmp, path)


# ---------------------------------------------------------------- start (or resume)
STATE = os.path.join(OUT, 'state.npz'); TJ = os.path.join(OUT, 'train.json')
steps = g('steps', 3000); lr0, lr1 = g('lr', 0.05), g('lr_end', 0.01); sh0 = g('shift', 1e-3)
res = dict(spec=SPEC, code_sha256=SHA_RUN, images=lat.nI, proj=lat.proj, nparams=NTH, nactive=int(mask.sum()), hist=[], done=False)
if os.path.exists(STATE):
    z = np.load(STATE)
    theta = jnp.asarray(z['theta']); S = jnp.asarray(z['S']); key = jnp.asarray(z['key']); it0 = int(z['it'])
    res = json.load(open(TJ)); res['hist'] = res['hist'][:it0]; res.setdefault('resumes', []).append(dict(at_step=it0, time=time.strftime('%FT%T')))
    progress(f'RESUMED from {STATE} at step {it0} (train.json hist truncated to {it0})')
else:
    ini = g('init')
    if ini.endswith('.npy'): th0 = np.load(ini)
    else:
        z = np.load(ini)
        th0 = np.asarray(z['theta']) if 'theta' in z.files else theta_from_f(z['f'], np.stack([z['u1'], z['u2']]) if 'u1' in z.files else None)
    assert th0.size == NTH, th0.size
    theta = jnp.asarray(th0, jnp.float64)
    key = jax.random.PRNGKey(1000 + g('seed', 1)); S = random_configs(N, g('chains', 2048), g('seed', 1)); it0 = 0
    res['init_sha256'] = hashlib.sha256(open(ini, 'rb').read()).hexdigest()
    t0 = time.time()
    for _ in range(g('therm', 20)):
        key, k = jax.random.split(key); S, acc = sweep(cast(theta), S, k, nsw)
    progress(f'thermalised ({time.time() - t0:.0f}s), acc {float(acc):.3f}')
    save_params('00000', theta)
progress(f'start: proj {lat.proj} ({lat.nI} images) params {NTH} (active {int(mask.sum())}) chains {S.shape[0]} step {it0}, '
         f'device {jax.devices()[0]}, eloc {EDT.__name__}, sampler {SDT.__name__}')


def plateau():
    """B0 rule: |mean(E[-w:]) - mean(E[-w-span:-span])| < tol (per site)."""
    w, span = g('plateau_window', 100), g('plateau_span', 200)
    E = np.array([h['E'] for h in res['hist']])
    if len(E) < max(g('plateau_min_step', 300), w + span): return None
    d = abs(E[-w:].mean() - E[-w - span:len(E) - span].mean())
    errs = np.array([h['err'] for h in res['hist']])      # per-step statistical error of E/N (independent chains within a step)
    se = float(np.sqrt(np.sum(errs[-w:] ** 2) / w ** 2 + np.sum(errs[-w - span:len(E) - span] ** 2) / w ** 2))   # SE of the difference of the two window means
    return dict(it=len(E), diff=float(d), se_diff=se, tol_over_se=float(g('plateau_tol', 1e-6) / se), mean_now=float(E[-w:].mean()), mean_before=float(E[-w - span:len(E) - span].mean()),
                tol=g('plateau_tol', 1e-6), ok=bool(d < g('plateau_tol', 1e-6)))


stop = None; n_run = 0; n_hold = 0
for it in range(it0 + 1, steps + 1):
    ts = time.time()
    c = 0.5 * (1 - np.cos(np.pi * (it - 1) / max(1, steps - 1)))
    lr = lr0 + (lr1 - lr0) * c; sh = sh0
    key, k = jax.random.split(key)
    S, acc = sweep(cast(theta), S, k, nsw); jax.block_until_ready(S); t1 = time.time()
    El, lP = eloc_all(theta.astype(EDT), S); El = El.astype(jnp.float64); jax.block_until_ready(El); t2 = time.time()
    O = jac_all(theta, S); jax.block_until_ready(O); t3 = time.time()
    dl, nrm, gn = sr_solve(O, El, lr, sh, g('max_dnorm', 0.3)); theta = theta + dl; jax.block_until_ready(theta); t4 = time.time()
    ns = S.shape[0]; E = float(jnp.mean(El)); var = float(jnp.var(El))
    rec = dict(it=it, E=E / N, err=float(np.sqrt(var / ns) / N), var=var / N, acc=float(acc), dnorm=float(nrm), gnorm=float(gn), lr=lr,
               t_step=t4 - ts, t_sample=t1 - ts, t_eloc=t2 - t1, t_jac=t3 - t2, t_sr=t4 - t3, wall=time.time() - T00)
    try: rec['gpu_peak_GB'] = jax.devices()[0].memory_stats()['peak_bytes_in_use'] / 1e9
    except Exception: pass
    res['hist'].append(rec); n_run += 1
    if it % 10 == 0 or it <= 3 or n_run <= 3:
        progress(f'it {it:5d} E/N {rec["E"]:.6f} +- {rec["err"]:.1e} var/N {rec["var"]:.3e} acc {rec["acc"]:.2f} |d|_S {rec["dnorm"]:.3f} '
                 f'gpu_peak {rec.get("gpu_peak_GB", float("nan")):.1f} GB ({rec["t_step"]:.1f}s: sample {rec["t_sample"]:.1f} eloc {rec["t_eloc"]:.1f} jac {rec["t_jac"]:.1f} sr {rec["t_sr"]:.1f})')
    pl = plateau(); res['plateau'] = pl
    if not np.isfinite(E): stop = 'non-finite energy'; break
    n_hold = n_hold + 1 if (pl and pl['ok']) else 0
    if n_hold >= g('plateau_hold', 1): stop = 'converged'
    if it % g('ckpt_every', 50) == 0 or stop or it == steps:
        save_params(f'{it:05d}', theta)
        atomic(STATE, lambda p: np.savez(open(p, 'wb'), theta=np.asarray(theta), S=np.asarray(S), key=np.asarray(key), it=it))
        atomic(TJ, lambda p: json.dump(res, open(p, 'w'), default=float))
    if stop: break
    if g('wall_sec') and time.time() - T00 > g('wall_sec'): stop = 'budget out'; break
    if g('max_steps_this_run') and n_run >= g('max_steps_this_run'): stop = 'max_steps_this_run'; break
if stop is None: stop = 'steps exhausted'
it_end = len(res['hist'])
save_params('final', theta)
atomic(STATE, lambda p: np.savez(open(p, 'wb'), theta=np.asarray(theta), S=np.asarray(S), key=np.asarray(key), it=it_end))
E_ = np.array([h['E'] for h in res['hist']])
res['stop'] = stop; res['converged'] = stop == 'converged'; res['steps_done'] = it_end; res['done'] = True
res['tail_mean_100'] = float(E_[-100:].mean()) if len(E_) else None
res['train_sec_this_run'] = time.time() - T00; res['s_per_step_median'] = float(np.median([h['t_step'] for h in res['hist'][-max(n_run, 1):]]))
atomic(TJ, lambda p: json.dump(res, open(p, 'w'), default=float))
progress(f'TRAIN DONE stop={stop} converged={res["converged"]} steps={it_end} E/N(last100)={res["tail_mean_100"]} '
         f'sec={res["train_sec_this_run"]:.0f}')
