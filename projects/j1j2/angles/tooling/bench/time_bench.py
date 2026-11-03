"""B1 timing: time per configuration per walker step (all bonds), steady state, all arms run the same loop:
   each step picks a random antiparallel bond, exchanges it, then returns all bond ratios of the new configuration.
   R: fresh inverse + slogdet + GEMM table.  Q: fresh inverse + slogdet + one lrux.det_lru ratio per bond (all NB bonds, mask).
   L: persistent (z,G,U,D); lrux.det_lru(...,True) update of G + Q's ratios from stored G.  H: same update + R's GEMM table from stored G.
   Qc/Lc (extra): Q/L with the antiparallel bonds compacted into a static-size list K.  INV: fresh inverse + slogdet only (cost of a refresh).
   Ld<T>: L with lrux.det_lru_delayed, max_delay T, merge when the buffer is full.
"""
import os, sys, json, time, argparse, subprocess
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b1_lib import *
ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--B', default='64'); ap.add_argument('--n', default='32,50')
ap.add_argument('--arms', default='R,Q,L,H,Qc,Lc,INV'); ap.add_argument('--nsteps', type=int, default=8); ap.add_argument('--reps', type=int, default=9)
ap.add_argument('--real', action='store_true'); ap.add_argument('--delayed', default='')
a = ap.parse_args()
dtype = jnp.float64 if a.real else jnp.complex128
FILES = {32: ('/scratch/a/A.Otaifi/ansatz_runs/fastratio/data/params_final.npz', 8), 50: ('/scratch/a/A.Otaifi/ansatz_runs/fastratio/data/params10_random.npz', 10)}
res = dict(devices=[str(d) for d in jax.devices()], dtype=str(np.dtype(dtype)), x64=bool(jax.config.jax_enable_x64), nsteps=a.nsteps, reps=a.reps, jax=jax.__version__,
           env={k: os.environ.get(k) for k in ['XLA_FLAGS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'SLURM_JOB_ID', 'SLURMD_NODENAME', 'CUDA_VISIBLE_DEVICES']},
           results={})
try:
    res['nvidia_smi_L'] = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout
except Exception as e:
    res['nvidia_smi_L'] = 'n/a: %r' % e
try:
    res['lscpu_model'] = [l for l in subprocess.run(['lscpu'], capture_output=True, text=True).stdout.splitlines() if 'Model name' in l]
except Exception:
    pass
print(json.dumps({k: res[k] for k in ['devices', 'dtype', 'env']}), flush=True)


def save():
    json.dump(res, open(a.out, 'w'), indent=1)


def timeit(fn, args, reps):
    t0 = time.time(); out = fn(*args); jax.block_until_ready(out); comp = time.time() - t0
    out = fn(*args); jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(*args); jax.block_until_ready(out); ts.append(time.perf_counter() - t0)
    return ts, comp, out


for n in [int(x) for x in a.n.split(',')]:
    path, L = FILES[n]
    f = load_f(path, dtype); A = Arms(L)
    for B in [int(x) for x in a.B.split(',')]:
        Z = jnp.asarray(random_configs(A.N, B, 7))
        keys = jax.random.split(jax.random.PRNGKey(5), (a.nsteps, B))
        cnt = np.asarray(jnp.sum(Z[:, A.BI] != Z[:, A.BJ], 1))
        K = int(min(A.NB, np.ceil(1.08 * max(cnt.max(), 0.5 * A.NB + 4 * np.sqrt(A.NB * 0.25 * 1.5)))))
        arms = [x for x in a.arms.split(',') if x] + ['Ld%s' % t for t in a.delayed.split(',') if t]
        for arm in arms:
            tag = '%s|n%d|B%d' % (arm, n, B)
            try:
                if arm in ('R', 'Q', 'INV'):
                    if arm == 'R':
                        s1 = lambda f, z, k: A.stepR(f, z, k)
                    elif arm == 'Q':
                        s1 = lambda f, z, k: A.stepQ(f, z, k)
                    else:
                        def s1(f, z, k):
                            p, q = A.pick(z, k); z = A.swap(z, p, q); U, D = A.lists(z); F = f[U][:, D]
                            sg, ld = jnp.linalg.slogdet(F); G = jnp.linalg.inv(F)
                            return z, jnp.sum(G) + ld
                    init = lambda f, Z: Z
                elif arm in ('L', 'H', 'Lc'):
                    s1 = {'L': lambda f, s, k: A.stepL(f, s, k), 'H': lambda f, s, k: A.stepH(f, s, k), 'Lc': lambda f, s, k: A.stepL(f, s, k, K=K)}[arm]
                    init = jax.vmap(A.init_state, (None, 0))
                elif arm == 'Qc':
                    s1 = lambda f, z, k: A.stepQ(f, z, k, K=K); init = lambda f, Z: Z
                elif arm.startswith('Ld'):
                    T = int(arm[2:])
                    init = jax.vmap(lambda f, z: A.init_carrier(f, z, T), (None, 0))
                    s1 = None
                else:
                    raise ValueError(arm)
                if s1 is not None:
                    def run(f, st, keys):
                        def body(c, k):
                            c, chk = jax.vmap(s1, (None, 0, 0))(f, c, k)
                            return c, chk
                        return jax.lax.scan(body, st, keys)
                else:
                    def run(f, st, keys, T=T):
                        outs = []
                        for s in range(a.nsteps):
                            st, r = jax.vmap(lambda f, c, k: A.stepLd(f, c, k, s % T, T), (None, 0, 0))(f, st, keys[s])
                            outs.append(jnp.sum(r, -1))
                        return st, jnp.stack(outs)
                st0 = jax.jit(init)(f, Z); jax.block_until_ready(st0)
                fn = jax.jit(run)
                ts, comp, out = timeit(fn, (f, st0, keys), a.reps)
                per = np.array(ts) / (B * a.nsteps)
                q1, med, q3 = np.percentile(per, [25, 50, 75])
                ok = bool(jnp.all(jnp.isfinite(out[1])))
                ov = None
                if arm in ('Qc', 'Lc'):
                    zf = out[0] if arm == 'Qc' else out[0][0]
                    ov = int(jnp.sum(jnp.sum(zf[:, A.BI] != zf[:, A.BJ], 1) > K))
                res['results'][tag] = dict(median_s=float(med), q1_s=float(q1), q3_s=float(q3), reps_s=per.tolist(), compile_s=comp, finite=ok, K=K if arm in ('Qc', 'Lc') else None,
                                          K_overflow_configs=ov)
                print('%-18s median %.3e s/cfg/step  IQR [%.3e, %.3e]  compile %.1fs finite=%s' % (tag, med, q1, q3, comp, ok), flush=True)
            except Exception as e:
                msg = repr(e)[:400]
                res['results'][tag] = dict(error=msg)
                print(tag, 'FAILED', msg, flush=True)
            save()
save()
print('done', a.out)
