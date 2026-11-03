"""B1 correctness (CPU): each arm vs brute-force slogdet on 200 configurations at n=32 (production f) and n=50 (10x10 random f)."""
import os, sys, json, time
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b1_lib import *
OUT = sys.argv[1] if len(sys.argv) > 1 else '/scratch/a/A.Otaifi/tooling_bench/b1'
D32 = '/scratch/a/A.Otaifi/ansatz_runs/fastratio/data/params_final.npz'
D50 = '/scratch/a/A.Otaifi/ansatz_runs/fastratio/data/params10_random.npz'
CFG = '/scratch/a/A.Otaifi/ansatz_runs/gpu_profile/acc_f64/cfg.npz'
res = {'devices': [str(d) for d in jax.devices()], 'x64': bool(jax.config.jax_enable_x64), 'dtype': 'complex128 (f real in file, cast)', 'tests': {}}


def metrics(r, bf, valid):
    el, nw = relerr(r, bf, valid)
    sm = sign_mismatch(r, bf, valid)
    d = jnp.where(valid, jnp.abs(r - bf) / jnp.abs(bf), 0.0)
    flat = int(jnp.argmax(d)); ci, bi = divmod(flat, d.shape[1])
    return dict(max_rel_elementwise=float(el.max()), max_rel_normwise=float(nw.max()), sign_mismatches=int(sm.sum()),
                n_bonds=int(valid.sum()), worst_abs_ratio_bf=float(jnp.abs(bf[ci, bi])), worst_config=ci, worst_bond=bi,
                p99_rel_elementwise=float(jnp.quantile(d[valid], 0.99)), n_nonfinite=int((~jnp.isfinite(r)).sum()))


def run(tag, L, fpath, Z):
    f = load_f(fpath, jnp.complex128)
    A = Arms(L)
    Z = jnp.asarray(Z)
    B = Z.shape[0]
    bfun = jax.jit(jax.vmap(A.brute, (None, 0)))
    t = time.time(); bf = bfun(f, Z); bf.block_until_ready(); tbf = time.time() - t
    valid = Z[:, A.BI] != Z[:, A.BJ]
    out = {'n': A.n, 'B': B, 'brute_s': tbf}
    vm = lambda fn: jax.jit(jax.vmap(fn, (None, 0)))
    out['R'] = metrics(vm(A.R_ratios)(f, Z)[1], bf, valid)
    out['Q'] = metrics(vm(A.Q_fresh)(f, Z)[1], bf, valid)
    out['Q_raw_quantax_gauge'] = metrics(vm(lambda f, z: A.Q_fresh(f, z, raw=True))(f, Z)[1], bf, valid)
    out['Q_broken_idx_swapped'] = metrics(vm(lambda f, z: A.Q_fresh(f, z, swap_idx=True))(f, Z)[1], bf, valid)
    # compacted-list variant Qc: K = 0.5 NB + margin; also check the cap is not exceeded
    cnt = int(valid.sum(1).max()); K = min(A.NB, int(np.ceil(cnt * 1.1)))
    out['Qc_K'] = K; out['max_antiparallel_count'] = cnt
    out['Qc'] = metrics(vm(lambda f, z: A.Q_fresh(f, z, K=K))(f, Z)[1], bf, valid)
    # persistent arms: states after k exchanges (random antiparallel bonds), compared to brute force on the new z
    st = vm(A.init_state)(f, Z)
    keys = jax.random.split(jax.random.PRNGKey(123), (20, B))
    mv = jax.jit(jax.vmap(lambda f, st, k: A.move(f, st, *A.pick(st[0], k))[0], (None, 0, 0)))
    Lr = vm(lambda f, st: A.Q_ratios(f, st[1], st[2], st[3], st[0]))
    Hr = vm(lambda f, st: A.H_ratios(f, st[1], st[2], st[3], st[0]))
    T = 8
    car = jax.jit(jax.vmap(lambda f, z: A.init_carrier(f, z, T), (None, 0)))(f, Z)
    stepsd = {}

    def ldstep(t):
        if t not in stepsd:
            stepsd[t] = jax.jit(jax.vmap(lambda f, s, k: A.stepLd(f, s, k, t, T), (None, 0, 0)))
        return stepsd[t]
    for s in range(20):
        st = mv(f, st, keys[s])
        car, rd = ldstep(s % T)(f, car, keys[s])
        if s + 1 in (1, 3, 8, 20):
            z = st[0]; v2 = z[:, A.BI] != z[:, A.BJ]
            bf2 = bfun(f, z)
            out['L_after_%d' % (s + 1)] = metrics(Lr(f, st), bf2, v2)
            out['H_after_%d' % (s + 1)] = metrics(Hr(f, st), bf2, v2)
            out['Ld8_after_%d' % (s + 1)] = metrics(rd, bf2, v2)
    res['tests'][tag] = out
    print(tag, json.dumps({k: (v if not isinstance(v, dict) else {a: v[a] for a in ('max_rel_elementwise', 'max_rel_normwise', 'sign_mismatches')}) for k, v in out.items()}, indent=0), flush=True)


run('n32_random_prod_f', 8, D32, random_configs(64, 200, 11))
run('n50_random_10x10_f', 10, D50, random_configs(100, 200, 12))
# near-node extra (n=32): 200 smallest |det f[U,D]| among the 10000 'met' configurations (identity image)
f32 = load_f(D32, jnp.complex128); A8 = Arms(8)
met = np.load(CFG)['met']
assert (met > 0).sum(1).min() == 32 and (met > 0).sum(1).max() == 32
ld = np.asarray(jax.jit(jax.vmap(lambda z: jnp.linalg.slogdet(f32[A8.lists(z)[0]][:, A8.lists(z)[1]])[1]))(jnp.asarray(met)))
order = np.argsort(ld)[:200]
res['near_node_pool'] = dict(source=CFG, key='met', n_total=len(met), log_abs_det_smallest=float(ld[order[0]]), log_abs_det_200th=float(ld[order[-1]]),
                             log_abs_det_median_all=float(np.median(ld)))
run('n32_near_node_200', 8, D32, met[order])
json.dump(res, open(os.path.join(OUT, 'correct.json'), 'w'), indent=1)
print('wrote', os.path.join(OUT, 'correct.json'))
