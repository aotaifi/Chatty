"""B1 drift (CPU): persistent inverse G updated by lrux.det_lru along a walk of random antiparallel exchanges, 8x8 production f,
starts = the 50 smallest-|psi| of the 200 near-node configurations.  Every CH steps the bond ratios from the stored G (L: lrux ratio per bond;
H: R's GEMM table) are compared with those from a fresh inverse (R), and ||G F - I||_max and cond(F) are recorded.
mode 'uniform': every proposal accepted (2000 exchanges, literal brief);  mode 'metropolis': |psi|^2 Metropolis, 4000 proposals (accepted count recorded)."""
import os, sys, json, time
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b1_lib import *
OUT = sys.argv[1] if len(sys.argv) > 1 else '/scratch/a/A.Otaifi/tooling_bench/b1'
NST = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
D32 = '/scratch/a/A.Otaifi/ansatz_runs/fastratio/data/params_final.npz'
CFG = '/scratch/a/A.Otaifi/ansatz_runs/gpu_profile/acc_f64/cfg.npz'
CH = 2
THR = 1e-8
f = load_f(D32, jnp.complex128); A = Arms(8); n = A.n
met = np.load(CFG)['met']
ld = np.asarray(jax.jit(jax.vmap(lambda z: jnp.linalg.slogdet(f[A.lists(z)[0]][:, A.lists(z)[1]])[1]))(jnp.asarray(met)))
order = np.argsort(ld)[:200]
starts = met[order[:50]]
meta = dict(source=CFG, key='met', start_indices=[int(i) for i in order[:50]], start_log_abs_det_min=float(ld[order[0]]), start_log_abs_det_max=float(ld[order[49]]),
            log_abs_det_median_all=float(np.median(ld)), CH=CH, THR=THR, devices=[str(d) for d in jax.devices()])
vm = lambda fn, ax=(None, 0): jax.jit(jax.vmap(fn, ax))


def stats(f, st):
    z, G, U, D = st
    valid = z[A.BI] != z[A.BJ]
    _, rf = A.R_ratios(f, z)
    rl = A.Q_ratios(f, G, U, D, z)
    rh = A.H_ratios(f, G, U, D, z)
    F = f[U][:, D]
    e = lambda r: relerr(r[None], rf[None], valid[None])
    (el_l, nw_l), (el_h, nw_h) = e(rl), e(rh)
    return dict(el_L=el_l[0], nw_L=nw_l[0], el_H=el_h[0], nw_H=nw_h[0],
                GFI=jnp.max(jnp.abs(G @ F - jnp.eye(n))), cond=jnp.linalg.cond(F), min_abs_ratio=jnp.min(jnp.where(valid, jnp.abs(rf), jnp.inf)),
                nvalid=jnp.sum(valid))


def step_u(f, st, key):
    p, q = A.pick(st[0], key)
    return A.move(f, st, p, q)[0], jnp.bool_(True)


def step_m(f, st, key):
    k1, k2 = jax.random.split(key)
    p, q = A.pick(st[0], k1)
    new, ratio = A.move(f, st, p, q)
    nx = jnp.sum(st[0][A.BI] != st[0][A.BJ]); ny = jnp.sum(new[0][A.BI] != new[0][A.BJ])
    acc = jnp.log(jax.random.uniform(k2)) < 2.0 * jnp.log(jnp.abs(ratio)) + jnp.log(nx / ny)
    return jax.tree_util.tree_map(lambda a, b: jnp.where(acc, a, b), new, st), acc


def run(mode, nprop):
    stepf = step_u if mode == 'uniform' else step_m
    B = len(starts)

    def block(f, st, keys):                                  # keys (CH, B, 2)
        def body(c, k):
            st, na = c
            st, acc = jax.vmap(stepf, (None, 0, 0))(f, st, k)
            return (st, na + acc.astype(jnp.int32)), None
        (st, na), _ = jax.lax.scan(body, (st, jnp.zeros(B, jnp.int32)), keys)
        return st, na
    blockj = jax.jit(block); statj = vm(stats)
    st = vm(A.init_state)(f, jnp.asarray(starts))
    key = jax.random.PRNGKey(2024 if mode == 'uniform' else 2025)
    rec = {k: [] for k in ['el_L', 'nw_L', 'el_H', 'nw_H', 'GFI', 'cond', 'min_abs_ratio']}
    steps = [0]; nacc = [np.zeros(B, int)]
    s0 = statj(f, st)
    for k in rec: rec[k].append(np.asarray(s0[k]))
    t0 = time.time(); na = np.zeros(B, int)
    for blk in range(nprop // CH):
        key, sub = jax.random.split(key)
        keys = jax.random.split(sub, (CH, B))
        st, dn = blockj(f, st, keys); na = na + np.asarray(dn)
        s = statj(f, st)
        for k in rec: rec[k].append(np.asarray(s[k]))
        steps.append((blk + 1) * CH); nacc.append(na.copy())
        if blk % 100 == 0: print('PROGRESS drift %s step %d t=%.0fs' % (mode, (blk + 1) * CH, time.time() - t0), flush=True)
    rec = {k: np.stack(v) for k, v in rec.items()}           # (nchecks, B)
    nacc = np.stack(nacc); steps = np.array(steps)
    # final-state ground truth: fresh inverse and stored G vs brute force
    z = st[0]; valid = z[:, A.BI] != z[:, A.BJ]
    bf = vm(A.brute)(f, z)
    rf = vm(lambda f, z: A.R_ratios(f, z)[1])(f, z)
    rl = vm(lambda f, s: A.Q_ratios(f, s[1], s[2], s[3], s[0]))(f, st)
    rh = vm(lambda f, s: A.H_ratios(f, s[1], s[2], s[3], s[0]))(f, st)
    fin = {nm: dict(zip(['max_rel_elementwise', 'max_rel_normwise'], [float(x.max()) for x in relerr(r, bf, valid)])) for nm, r in [('fresh_R', rf), ('stored_L', rl), ('stored_H', rh)]}
    out = dict(mode=mode, n_proposals=nprop, wall_s=time.time() - t0, final_vs_bruteforce=fin,
               accepted_total_min=int(nacc[-1].min()), accepted_total_median=float(np.median(nacc[-1])))
    for key_, label in [('el_L', 'L_elementwise'), ('nw_L', 'L_normwise'), ('el_H', 'H_elementwise'), ('nw_H', 'H_normwise')]:
        x = rec[key_]
        first = np.where((x > THR).any(0), (x > THR).argmax(0), -1)
        # step count in ACCEPTED exchanges at first exceedance (uniform: equal to proposals)
        fs = np.array([nacc[first[b], b] if first[b] >= 0 else -1 for b in range(B)])
        ok = fs >= 0
        out['first_exceed_%s' % label] = dict(per_start_accepted_steps=fs.tolist(), n_never=int((~ok).sum()),
                                              median=float(np.median(fs[ok])) if ok.any() else None, min=int(fs[ok].min()) if ok.any() else None,
                                              max=int(fs[ok].max()) if ok.any() else None)
    sel = slice(None, None, max(1, len(steps) // 400))
    out['curves'] = dict(steps=steps[sel].tolist(), **{k + '_median': np.median(v, 1)[sel].tolist() for k, v in rec.items()},
                         **{k + '_max': v.max(1)[sel].tolist() for k, v in rec.items()},
                         accepted_median=np.median(nacc, 1)[sel].tolist())
    # check at multiples of 10 accepted-ish steps (the brief's cadence), value of max over starts at steps 10,20,50,100,200,500,1000,2000
    cad = {}
    for s_ in (10, 20, 50, 100, 200, 500, 1000, 2000):
        i = np.searchsorted(steps, s_)
        if i < len(steps):
            cad[str(s_)] = {k: float(np.max(v[i])) for k, v in rec.items() if k != 'min_abs_ratio'}
            cad[str(s_)]['cond_median'] = float(np.median(rec['cond'][i]))
    out['max_over_starts_at_step'] = cad
    out['cond_over_walk'] = dict(median=float(np.median(rec['cond'])), p99=float(np.quantile(rec['cond'], 0.99)), max=float(rec['cond'].max()))
    return out


res = dict(meta=meta)
res['uniform'] = run('uniform', NST)
print('uniform done', json.dumps({k: res['uniform']['first_exceed_' + k] for k in ['L_elementwise', 'H_elementwise', 'L_normwise', 'H_normwise']}, default=lambda x: None)[:600], flush=True)
res['metropolis'] = run('metropolis', 2 * NST)
json.dump(res, open(os.path.join(OUT, 'drift.json'), 'w'))
print('wrote', os.path.join(OUT, 'drift.json'))
