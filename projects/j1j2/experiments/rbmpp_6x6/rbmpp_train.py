"""SR (stochastic reconfiguration = natural gradient / imaginary-time projection) optimisation of RBM+PP by VMC.

  python rbmpp_train.py OUT SPEC.json
SPEC keys (defaults in brackets):
  L [6], J2 [0.5], proj ['simple'], M [16], rbm [true], seed [0], init [null: npz of params to start from;
  'rbm_from_pp': true adds a fresh small random RBM to an init without one], steps [2000], chains [2048],
  sweep [N proposals per chain between samples], therm [20 sweeps], lr [0.03] (lr_end, cosine), shift [1e-3]
  (relative diagonal shift of the rescaled S, shift_end), max_dnorm [0.2] (cap on sqrt(d^T S d)), chunk [256],
  ckpt_every [50], eval_samples [0: final high-statistics VMC energy with blocking], eval_proj [list].
Every step records E/N, its naive error, Var(E_loc)/N, acceptance, |d|_S, wall-clock. Parameters are saved
every ckpt_every steps (params_XXXXX.npz, ~40 kB each) so that intermediate (moderate) checkpoints can be used.
SR: theta real (f, Re b, Im b, Re W, Im W); O = d log psi_P / d theta (complex);
  S = Re cov(O, O), F = Re cov(O, E_loc);  d = -lr (S + shift diag S)^-1 F (solved in the diagonally rescaled
  basis by Cholesky).
"""
import json, os, sys, time
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from rbmpp_lib import Lattice, init_params, make_fns, make_sampler, random_configs, to32
jax.config.update("jax_enable_x64", True)

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
T00 = time.time()


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


g = SPEC.get
lat = Lattice(g('L', 6), g('J2', 0.5), g('proj', 'simple'))
N = lat.N
use_rbm = g('rbm', True)
p = init_params(lat, M=g('M', 16), seed=g('seed', 0), rbm=use_rbm)
if g('init'):
    z = np.load(g('init'))
    for k in z.files:
        if k in p: p[k] = jnp.asarray(z[k])
    if g('rbm_from_pp'):
        q = init_params(lat, M=g('M', 16), seed=g('seed', 0) + 100, rbm=True)
        for k in ('bre', 'bim', 'Wre', 'Wim'): p[k] = q[k]
if not use_rbm:
    for k in ('bre', 'bim', 'Wre', 'Wim'): p[k] = jnp.zeros_like(p[k])
flat, unravel = ravel_pytree(p)
P = flat.size
# active-parameter mask built on the pytree (ravel_pytree orders keys alphabetically: Wim, Wre, bim, bre, f)
mask = ravel_pytree({k: jnp.full(v.shape, (k == 'f') or use_rbm) for k, v in p.items()})[0] > 0.5
assert int(mask.sum()) == (P if use_rbm else N * N)
log(f'lattice {lat.L}x{lat.L} proj {lat.proj} ({lat.nI} images) rbm {use_rbm} params {P} (active {int(mask.sum())})')

fns = make_fns(lat, jnp.float64, use_rbm)
SDT = jnp.float64 if g('sampler_dtype', 'float32') == 'float64' else jnp.float32
sweep = make_sampler(lat, make_fns(lat, SDT, use_rbm))
cast = (lambda q: q) if SDT == jnp.float64 else to32
CH = g('chunk', 256)
eloc_v = jax.jit(jax.vmap(fns['eloc'], in_axes=(None, 0)))
jac_v = jax.jit(jax.vmap(jax.jacrev(lambda fl, s: fns['logpsi_ri'](unravel(fl), s)), in_axes=(None, 0)))


def chunks(fn, *args, S):
    outs = [fn(*args, S[i:i + CH]) for i in range(0, S.shape[0], CH)]
    if isinstance(outs[0], tuple):
        return tuple(jnp.concatenate([o[k] for o in outs]) for k in range(len(outs[0])))
    return jnp.concatenate(outs)


@jax.jit
def sr_solve(O, El, lr, shift, maxn):
    Om = O - jnp.mean(O, 0)
    Ec = El - jnp.mean(El)
    ns = O.shape[0]
    S = jnp.real(Om.conj().T @ Om) / ns
    F = jnp.real(Om.conj().T @ Ec) / ns
    S = jnp.where(mask[:, None] & mask[None, :], S, 0.0); F = jnp.where(mask, F, 0.0)
    d = jnp.maximum(jnp.diag(S), 1e-12)
    sq = jnp.sqrt(d)
    Ss = S / (sq[:, None] * sq[None, :]) + shift * jnp.eye(S.shape[0])
    y = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(Ss), F / sq)
    dl = -lr * y / sq
    dl = jnp.where(mask, dl, 0.0)
    nrm = jnp.sqrt(jnp.maximum(dl @ S @ dl, 0.0))
    dl = dl * jnp.minimum(1.0, maxn / jnp.maximum(nrm, 1e-30))
    return dl, nrm, jnp.sqrt(jnp.sum(F * F))


def save(tag, fl):
    q = unravel(fl)
    np.savez(os.path.join(OUT, f'params_{tag}.npz'), **{k: np.asarray(v) for k, v in q.items()},
             proj=lat.proj, rbm=use_rbm)


key = jax.random.PRNGKey(1000 + g('seed', 0))
S = random_configs(N, g('chains', 2048), g('seed', 0))
nsw = g('sweep', N)
t0 = time.time()
for t in range(g('therm', 20)):
    key, k = jax.random.split(key)
    S, acc = sweep(cast(unravel(flat)), S, k, nsw)
log(f'thermalised ({time.time() - t0:.0f}s), acc {float(acc):.3f}')

steps = g('steps', 2000)
lr0, lr1 = g('lr', 0.03), g('lr_end', g('lr', 0.03))
sh0, sh1 = g('shift', 1e-3), g('shift_end', g('shift', 1e-3))
hist = []
F = os.path.join(OUT, 'train.json')
res = dict(spec=SPEC, nparams=int(P), nactive=int(mask.sum()), images=lat.nI, hist=hist)
save('00000', flat)
for it in range(1, steps + 1):
    ts = time.time()
    c = 0.5 * (1 - np.cos(np.pi * (it - 1) / max(1, steps - 1)))
    lr = lr0 + (lr1 - lr0) * c; sh = np.exp(np.log(sh0) + (np.log(sh1) - np.log(sh0)) * c)
    key, k = jax.random.split(key)
    S, acc = sweep(cast(unravel(flat)), S, k, nsw)
    tsm = time.time() - ts
    El, lP = chunks(eloc_v, unravel(flat), S=S)
    Elr = jnp.real(El)
    J = chunks(jac_v, flat, S=S)                          # (ns, 2, P)
    O = J[:, 0, :] + 1j * J[:, 1, :]
    dl, nrm, gn = sr_solve(O, El, lr, sh, g('max_dnorm', 0.2))
    flat = flat + dl
    E = float(jnp.mean(Elr)); var = float(jnp.var(Elr)); ns = S.shape[0]
    rec = dict(it=it, E=E / N, err=np.sqrt(var / ns) / N, var=var / N, Eim=float(jnp.mean(jnp.imag(El))) / N,
               acc=float(acc), dnorm=float(nrm), gnorm=float(gn), lr=lr, shift=sh, t_step=time.time() - ts,
               t_sample=tsm, wall=time.time() - T00)
    hist.append(rec)
    if it % 10 == 0 or it <= 3:
        log(f'it {it:5d} E/N {E / N:.6f} +- {rec["err"]:.1e} var/N {rec["var"]:.3e} acc {rec["acc"]:.2f} '
            f'|d|_S {rec["dnorm"]:.3f} |F| {rec["gnorm"]:.2e} ({rec["t_step"]:.2f}s, sample {tsm:.2f}s)')
    if it % g('ckpt_every', 50) == 0 or it == steps:
        save(f'{it:05d}', flat); json.dump(res, open(F, 'w'), default=float)
    if not np.isfinite(E):
        log('non-finite energy, stop'); break
save('final', flat)
res['train_sec'] = time.time() - T00
json.dump(res, open(F, 'w'), default=float)


# ---------------------------------------------------------------- final high-statistics VMC energy (blocking)
def vmc_energy(fl, proj, nsamp, key, S0):
    lat.set_proj(proj)
    f64 = make_fns(lat, jnp.float64, use_rbm); f32 = make_fns(lat, jnp.float32, use_rbm)
    ev = jax.jit(jax.vmap(f64['eloc'], in_axes=(None, 0))); sw = make_sampler(lat, f32)
    pp = unravel(fl); p32 = to32(pp); Sx = S0
    for _ in range(g('therm', 20)):
        key, k = jax.random.split(key); Sx, _ = sw(p32, Sx, k, nsw)
    means = []
    while len(means) * Sx.shape[0] < nsamp:
        key, k = jax.random.split(key); Sx, _ = sw(p32, Sx, k, nsw)
        El = jnp.concatenate([ev(pp, Sx[i:i + CH])[0] for i in range(0, Sx.shape[0], CH)])
        means.append(np.asarray(jnp.real(El)))
    A = np.array(means)                                    # (sweeps, chains)
    m = A.mean()
    chain_means = A.mean(0)                                # chains are independent: error from chain means
    err = chain_means.std(ddof=1) / np.sqrt(A.shape[1])
    return dict(proj=proj, E=m / N, err=err / N, var=float(A.var()) / N, nsamp=int(A.size), sweeps=A.shape[0])


if g('eval_samples', 0):
    res['eval'] = []
    for proj in g('eval_proj', [lat.proj]):
        te = time.time()
        r = vmc_energy(flat, proj, g('eval_samples'), jax.random.PRNGKey(77), S)
        r['sec'] = time.time() - te; res['eval'].append(r); log('eval', r)
        json.dump(res, open(F, 'w'), default=float)
res['sec'] = time.time() - T00
json.dump(res, open(F, 'w'), default=float)
log('DONE', res['sec'])
