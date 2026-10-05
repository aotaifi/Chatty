"""Plain-VMC continuation of the published 6x6 J1-J2 ViT (J2/J1=0.5), compute-matched baseline.

Recipe (Sinibaldi et al. arXiv:2502.09725, supplement; nqsmagic example conventions):
  ViT(4 layers, d=60, 10 heads, 2x2 patches, translation-invariant attention) symmetrised over the
  b x b=2x2 patch shifts via nqsmagic.utils._logpsi_transl_2d; float64/complex128; 154980 real params.
  MinSR / kernel-form SR (netket VMC_SR, use_ntk=True), diag_shift 1e-4, SGD step = lr, Ns = 6000,
  MetropolisExchange(d_max=2, sweep_size=N) in S^z=0, H = J1 sum S.S + J2 sum S.S (no Marshall rule).
Learning rate is constant (late-stage; checkpoint already converged): --lr.
Logs per step: train-sample energy/N, step wall time (device-synchronised), cumulative training GPU time.
Every --eval-every steps: independent large-sample energy/N of the current parameters (time not counted in
training GPU time but logged).  Resumable (ckpt.mpack + state.json).
"""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.expanduser('~/chatty_vmc6/nqsmagic')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--lr', type=float, default=0.005)
ap.add_argument('--steps', type=int, default=1000)           # target total step count
ap.add_argument('--time-budget', type=float, default=1e9)    # training seconds this invocation may use
ap.add_argument('--eval-every', type=int, default=50)
ap.add_argument('--eval-nsamples', type=int, default=65536)
ap.add_argument('--eval-nchains', type=int, default=4096)
ap.add_argument('--ns', type=int, default=6000)
ap.add_argument('--nchains', type=int, default=1500)
ap.add_argument('--discard', type=int, default=4)
ap.add_argument('--diag', type=float, default=1e-4)
ap.add_argument('--seed', type=int, default=20261004)
ap.add_argument('--chunk', type=int, default=1500)
ap.add_argument('--eval-chunk', type=int, default=2048)
ap.add_argument('--jchunk', type=int, default=250)
ap.add_argument('--therm', type=int, default=60)
ap.add_argument('--skip-eval0', action='store_true')
ap.add_argument('--init', default=os.path.expanduser('~/chatty_vmc6/data/vit_J2=0.50_N=6x6_k=0.mpack'))
ap.add_argument('--root', default=os.path.expanduser('~/chatty_vmc6/runs'))
ap.add_argument('--final-eval-nsamples', type=int, default=262144)
args = ap.parse_args()

import numpy as np
import jax, jax.numpy as jnp
import netket as nk, flax, optax
import netket.jax as nkjax
import vit_dt   # copy of nqsmagic ViT (float64) without lax.fori_loop (netket chunked shard_map is incompatible with jax 0.10 + fori_loop); checked == nqsmagic below
from flax import serialization
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

T0 = time.time()
def log(*a): print(time.strftime('[%H:%M:%S]'), *a, flush=True)
OUT = os.path.join(args.root, args.tag); os.makedirs(OUT, exist_ok=True)
log('devices', jax.devices(), 'netket', nk.__version__, 'args', vars(args))

L = 6; N = 36; J2 = 0.5
hi = nk.hilbert.Spin(s=.5, N=N, total_sz=0)
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)
H = nk.operator.Heisenberg(hi, g, J=[0.25, 0.25 * J2], sign_rule=[False, False]).to_jax_operator()  # S.S = sigma.sigma/4
model = vit_dt.make_model_6x6()
apply = nkjax.HashablePartial(vit_dt.logpsi_transl_2d, model.apply, 2)
model_ref = ViT(num_layers=4, d_model=60, heads=10, L_eff=9, b=2, transl_invariant=True, two_dimensional=True)
apply_ref = nkjax.HashablePartial(_logpsi_transl_2d, model_ref.apply, 2)

def make_state(nchains, nsamples, discard, seed, chunk):
    sam = nk.sampler.MetropolisExchange(hi, graph=g, d_max=2, n_chains=nchains, sweep_size=N)
    return nk.vqs.MCState(sam, apply_fun=apply, n_samples=nsamples, n_discard_per_chain=discard, seed=seed,
                          variables=model.init(jax.random.PRNGKey(1234), jnp.zeros((1, N))), chunk_size=chunk)

vs = make_state(args.nchains, args.ns, args.therm, args.seed, args.chunk)          # long initial thermalisation
ve = make_state(args.eval_nchains, args.eval_nsamples, args.therm, args.seed + 1, args.eval_chunk)
with open(args.init, 'rb') as f:
    pars0 = serialization.from_state_dict(vs.variables, serialization.msgpack_restore(f.read())['variables'])  # mpack = full MCState dict
vs.variables = pars0
log('nparams', vs.n_parameters)
_x = hi.random_state(jax.random.PRNGKey(7), 64)
_d = jnp.abs(jax.jit(lambda v, x: apply(v, x))(vs.variables, _x) - jax.jit(lambda v, x: apply_ref(v, x))(vs.variables, _x)).max()
log('CHECK max|dlogpsi| vit_dt vs nqsmagic', float(_d))

state_path = os.path.join(OUT, 'state.json'); ckpt_path = os.path.join(OUT, 'ckpt.mpack')
st = dict(step=0, train_sec=0.0, eval_sec=0.0, wall_sec=0.0, invocations=0)
if os.path.exists(state_path) and os.path.exists(ckpt_path):
    st = json.load(open(state_path))
    with open(ckpt_path, 'rb') as f:
        vs.variables = serialization.from_bytes(vs.variables, f.read())
    log('RESUMED', st)
st['invocations'] += 1

# ---- MinSR / kernel-form SR step, same algebra as netket.driver.VMC_SR(use_ntk=True, mode='complex') -------
# (netket's on-the-fly NTK and direct-Jacobian paths both exceed 44 GB on the A40 for Ns=6000, Np=154980,
#  so the identical linear algebra is done here with a row-blocked Gram matrix.)
from jax.flatten_util import ravel_pytree
flat0, unravel = ravel_pytree(vs.parameters)
NP = flat0.size
def f_re_im(flat, x):
    z = apply({'params': unravel(flat)}, x[None])[0]
    return jnp.stack([z.real, z.imag])
jac1 = jax.jacrev(f_re_im)                                  # (2, Np) per sample
def jac_all(flat, X):                                       # X (Ns, N) -> (Ns*2, Np), rows [Re_1, Im_1, Re_2, ...]
    Xc = X.reshape(-1, args.jchunk, N)
    J = jax.lax.map(lambda xs: jax.vmap(jac1, in_axes=(None, 0))(flat, xs), Xc)
    return J.reshape(-1, NP)
jac_all = jax.jit(jac_all)
NB = 4
def gram(O):
    n = O.shape[0]; bs = n // NB
    blocks = {}
    for i in range(NB):
        Oi = jax.lax.slice_in_dim(O, i * bs, (i + 1) * bs, axis=0)
        for j in range(i, NB):
            Oj = jax.lax.slice_in_dim(O, j * bs, (j + 1) * bs, axis=0)
            blocks[(i, j)] = jax.lax.dot_general(Oi, Oj, (((1,), (1,)), ((), ())), precision=jax.lax.Precision.HIGHEST)
    rows = []
    for i in range(NB):
        rows.append(jnp.concatenate([blocks[(i, j)] if j >= i else blocks[(j, i)].T for j in range(NB)], axis=1))
    return jnp.concatenate(rows, axis=0)
gram = jax.jit(gram)
@jax.jit
def solve_update(T0, dvec, O, diag):
    Ns = O.shape[0] // 2
    T = T0.reshape(Ns, 2, Ns, 2)
    col = T.mean(axis=0, keepdims=True); row = T.mean(axis=2, keepdims=True); glob = col.mean(axis=2, keepdims=True)
    T = (T - col - row + glob).reshape(2 * Ns, 2 * Ns) / Ns
    Ts = T + diag * jnp.eye(2 * Ns)
    Lc = jnp.linalg.cholesky(Ts)
    ok = jnp.all(jnp.isfinite(Lc))
    a = jax.scipy.linalg.cho_solve((Lc, True), dvec)
    a = jnp.where(ok, a, jnp.linalg.solve(Ts, dvec))
    a = a.reshape(Ns, 2)
    a = (a - a.mean(axis=0, keepdims=True)) / jnp.sqrt(Ns)
    return jax.lax.dot_general(a.reshape(-1), O, (((0,), (0,)), ((), ()))), ok     # O^T a  (Np,)
def minsr_step(vs):
    tt = {}
    t = time.time(); X = vs.samples.reshape(-1, N); jax.block_until_ready(X); tt['t_sample'] = time.time() - t
    t = time.time(); Eloc = vs.local_estimators(H); stats = nk.stats.statistics(Eloc); jax.block_until_ready(Eloc); tt['t_eloc'] = time.time() - t
    Ns = X.shape[0]
    el = jnp.asarray(Eloc).reshape(-1); de = el - el.mean(); dv = 2.0 * de / jnp.sqrt(Ns)
    dvec = jnp.stack([dv.real, dv.imag], axis=-1).reshape(-1)
    flat, _ = ravel_pytree(vs.parameters)
    t = time.time(); O = jac_all(flat, X); jax.block_until_ready(O); tt['t_jac'] = time.time() - t
    t = time.time(); T0 = gram(O); jax.block_until_ready(T0); tt['t_gram'] = time.time() - t
    t = time.time(); upd, ok = solve_update(T0, dvec, O, args.diag); jax.block_until_ready(upd); tt['t_solve'] = time.time() - t
    del O, T0
    vs.parameters = unravel(flat - args.lr * upd)
    tt['chol_ok'] = bool(ok)
    return stats, tt

vs.n_discard_per_chain = args.therm
vs.n_discard_per_chain = args.therm; vs.reset(); vs.sample(); vs.n_discard_per_chain = args.discard   # thermalise once; later steps keep chains
log('thermalised')

def save():
    tmp = ckpt_path + '.tmp'
    with open(tmp, 'wb') as f: f.write(serialization.to_bytes(vs.variables))
    os.replace(tmp, ckpt_path)
    st['wall_sec_total'] = st['wall_sec'] + (time.time() - T0)
    json.dump(st, open(state_path, 'w'))

def do_eval(nsamp=None, final=False):
    t = time.time()
    ve.variables = vs.variables
    if nsamp is not None and nsamp != ve.n_samples:
        ve.n_samples = nsamp
    if ve.n_discard_per_chain < 100 and not hasattr(ve, '_therm'):
        pass
    est = ve.expect(H)
    dt = time.time() - t
    rec = dict(step=st['step'], E_site=float(est.mean.real) / N, err_site=float(est.error_of_mean) / N,
               tau=float(getattr(est, 'tau_corr', float('nan'))), R=float(getattr(est, 'R_hat', float('nan'))),
               nsamples=ve.n_samples, eval_dt=dt, train_sec=st['train_sec'], train_gpu_h=st['train_sec'] / 3600.,
               final=final)
    st['eval_sec'] += dt
    with open(os.path.join(OUT, 'eval.jsonl'), 'a') as f: f.write(json.dumps(rec) + '\n')
    log('EVAL', json.dumps(rec))
    return rec

ve.n_discard_per_chain = args.therm; ve.reset(); ve.sample(); ve.n_discard_per_chain = 4
if st['step'] == 0 and not args.skip_eval0:
    do_eval()
hist = open(os.path.join(OUT, 'history.jsonl'), 'a')
t_inv = 0.0
while st['step'] < args.steps and t_inv < args.time_budget:
    first = (t_inv == 0.0)
    t = time.time()
    e, tt = minsr_step(vs)
    jax.block_until_ready(vs.parameters)
    dt = time.time() - t
    st['step'] += 1; st['train_sec'] += dt; t_inv += dt
    rec = dict(step=st['step'], E_site=float(e.mean.real) / N, err_site=float(e.error_of_mean) / N,
               var=float(e.variance), step_dt=dt, **tt, train_sec=st['train_sec'], train_gpu_h=st['train_sec'] / 3600.,
               lr=args.lr, first_of_invocation=first, job=os.environ.get('SLURM_JOB_ID'))
    hist.write(json.dumps(rec) + '\n'); hist.flush()
    if st['step'] <= 5 or st['step'] % 10 == 0:
        log('STEP', json.dumps(rec))
    if st['step'] % args.eval_every == 0:
        save(); do_eval()
st_done = st['step'] >= args.steps
save()
if st_done or t_inv >= args.time_budget:
    do_eval()
if st_done:
    do_eval(args.final_eval_nsamples, final=True)
    import shutil; shutil.copy(ckpt_path, os.path.join(OUT, 'final_params.mpack'))
save()
log('DONE', json.dumps(st))
