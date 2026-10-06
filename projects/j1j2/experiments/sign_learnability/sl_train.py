#!/usr/bin/env python3
"""Stage B (GPU, jax env): train one sign classifier and evaluate it on EVERY orbit of the symmetric sector.

Learner (same family as stored_signs / stored_signs_6x6): translation-invariant periodic CNN on the torus
(k x k gather convolutions, residual GELU blocks with LayerNorm, per-site head summed over sites / sqrt(N)),
output = (logit, aux).  Loss = BCE(logit, y) [+ lam * MSE(aux, asinh(T - r)) for Krylov-step targets].
Training: uniform minibatches (with replacement) over the unique training orbits (tempered samples, optionally
+ all one-hop neighbours, from sl_prep.py), random point-group element + random spin flip per sample (the
targets are exactly invariant under the full group).  Evaluation: logit averaged over point group x spin flip
(translations are built in), on all orbit representatives -> logits.npy.
Targets: c1..c4 = Krylov step labels c_k = sgn[T_{k-1} - r_{k-1}] (exact amplitude, exact s_{k-1});
         gs = exact ground-state sign relative to Marshall; m2, m3 = full Krylov sign s_k relative to Marshall.
Usage: python sl_train.py --N 36 --target c1 --beta 0.5 --seed 0 --n 10000 --size M --data DATA --runs RUNS
       python sl_train.py --tasks FILE --lo I --hi J   (run lines I..J-1 of a task file, each line = arguments)
"""
import argparse, json, os, sys, time, shlex, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sl_common as SC

SIZES = dict(XS=(8, 2), S=(16, 3), M=(32, 4), L=(64, 6), XL=(128, 6))


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int)
    ap.add_argument('--target', default='c1')
    ap.add_argument('--beta', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--nbr', type=int, default=1)
    ap.add_argument('--size', default='M')
    ap.add_argument('--kern', type=int, default=3)
    ap.add_argument('--lam', type=float, default=0.0)
    ap.add_argument('--epochs', type=float, default=10)
    ap.add_argument('--min-steps', type=int, default=5000)
    ap.add_argument('--max-steps', type=int, default=40000)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--init-seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--data', default='/home/a/A.Otaifi/chatty_signlearn/data')
    ap.add_argument('--runs', default='/home/a/A.Otaifi/chatty_signlearn/runs')
    ap.add_argument('--tasks', default=None)
    ap.add_argument('--lo', type=int, default=0)
    ap.add_argument('--hi', type=int, default=10 ** 9)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--eval-mem', type=float, default=1.5e8, help='floats per gathered activation in eval')
    return ap


def run_name(a):
    t = f"{a.target}_b{a.beta}_s{a.seed}_n{a.n}_{a.size}_nbr{a.nbr}_lam{a.lam}"
    if a.kern != 3: t += f"_k{a.kern}"
    if a.epochs != 10 or a.max_steps != 40000: t += f"_ep{a.epochs:g}_mx{a.max_steps}"
    if a.init_seed: t += f"_i{a.init_seed}"
    return t + (f"_{a.tag}" if a.tag else '')


def unpack(states, N):
    S = np.asarray(states, np.uint64)
    return (((S[:, None] >> np.arange(N, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.int8) * 2 - 1)


_CACHE = {}


def load_data(base, N, target):
    d = SC.data_dir(base, N)
    if ('D', N) not in _CACHE:
        _CACHE.clear()
        _CACHE[('D', N)] = dict(states=np.load(f'{d}/states.npy'), p0=np.load(f'{d}/p0.npy'),
                                meta=json.load(open(f'{d}/meta.json')))
    D = _CACHE[('D', N)]
    if ('y', N, target) not in _CACHE:
        if target == 'gs':
            y = np.load(f'{d}/sigma0.npy'); r = None; T = None
        elif target.startswith('m'):        # full Krylov sign s_k relative to Marshall (not the step factor)
            k = int(target[1:]); y = (np.load(f'{d}/s_{k}.npy') * np.load(f'{d}/marshall.npy')).astype(np.int8)
            if np.sum(D['p0'] * y) < 0: y = -y
            r = None; T = None
        else:
            k = int(target[1:]); y = np.load(f'{d}/c_{k}.npy'); r = np.load(f'{d}/r_{k-1}.npy', mmap_mode='r')
            T = D['meta']['steps'][k]['T_prev']
        _CACHE[('y', N, target)] = (y, r, T)
    return D, _CACHE[('y', N, target)]


def train_one(a):
    import jax, jax.numpy as jnp
    import flax.linen as nn
    import optax
    jax.config.update('jax_default_matmul_precision', 'highest')
    out = os.path.join(a.runs, f'N{a.N}', run_name(a))
    if os.path.exists(f'{out}/run.json') and not a.force:
        print('SKIP (done)', out, flush=True); return
    os.makedirs(out, exist_ok=True)
    try:                                   # claim the task (several workers may share one task file)
        os.close(os.open(f'{out}/.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        if not a.force:
            print('SKIP (claimed by another worker)', out, flush=True); return
    t00 = time.time()
    N = a.N
    D, (y, r, T) = load_data(a.data, N, a.target)
    states, p0 = D['states'], D['p0']
    tr = np.load(f"{SC.data_dir(a.data, N)}/train_b{a.beta}_s{a.seed}.npz")
    idx = tr[f"{'nbr' if a.nbr else 'smp'}_{a.n}"].astype(np.int64)
    nbr, perms, _ = SC.geometry(N, a.kern)
    invp = np.argsort(perms, axis=1).astype(np.int32)       # x'[j] = x[invp[g, j]]
    G = len(invp)
    Xtr = jnp.asarray(unpack(states[idx], N))
    ytr = jnp.asarray((y[idx] > 0).astype(np.float32))
    use_aux = a.lam > 0 and r is not None
    atr = jnp.asarray(np.arcsinh(T - np.asarray(r)[idx]).astype(np.float32)) if use_aux else jnp.zeros(len(idx), jnp.float32)
    ch, depth = SIZES[a.size]
    nbrj = jnp.asarray(nbr); invj = jnp.asarray(invp); sqN = float(np.sqrt(N))

    class GCNN(nn.Module):
        @nn.compact
        def __call__(self, x):                    # x: (B, N) float32 +-1
            h = x[:, :, None]

            def conv(h, c, name):
                g = h[:, nbrj, :]                 # (B, N, K, C)
                return nn.Dense(c, name=name)(g.reshape(g.shape[0], g.shape[1], -1))
            h = nn.gelu(conv(h, ch, 'c0'))
            for dd in range(1, depth):
                h = h + nn.gelu(conv(nn.LayerNorm(name=f'ln{dd}')(h), ch, f'c{dd}'))
            o = nn.Dense(2, name='head')(h).sum(axis=1) / sqN
            return o[:, 0], o[:, 1]

    model = GCNN()
    params = model.init(jax.random.PRNGKey(a.init_seed), jnp.zeros((1, N), jnp.float32))['params']
    npar = int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)))
    ntr = len(idx)
    nsteps = int(min(a.max_steps, max(a.min_steps, a.epochs * ntr / a.batch)))
    sched = optax.warmup_cosine_decay_schedule(0., a.lr, max(1, nsteps // 50), nsteps)
    opt = optax.adamw(sched, weight_decay=a.wd); st = opt.init(params)

    def loss_fn(p, xb, yb, ab):
        lo, au = model.apply({'params': p}, xb)
        bce = jnp.mean(optax.sigmoid_binary_cross_entropy(lo, yb))
        l = bce + (a.lam * jnp.mean((au - ab) ** 2) if use_aux else 0.)
        return l, bce

    CH = 250

    @jax.jit
    def chunk(p, st, key):
        def body(c, _):
            p, st, key = c
            key, k1, k2, k3 = jax.random.split(key, 4)
            ib = jax.random.randint(k1, (a.batch,), 0, ntr)
            g = jax.random.randint(k2, (a.batch,), 0, G)
            f = jnp.where(jax.random.bernoulli(k3, 0.5, (a.batch,)), 1., -1.)
            xb = jnp.take_along_axis(Xtr[ib].astype(jnp.float32), invj[g], axis=1) * f[:, None]
            (l, b), gr = jax.value_and_grad(loss_fn, has_aux=True)(p, xb, ytr[ib], atr[ib])
            u, st = opt.update(gr, st, p)
            return (optax.apply_updates(p, u), st, key), b
        (p, st, key), bs = jax.lax.scan(body, (p, st, key), None, length=CH)
        return p, st, key, bs.mean()

    key = jax.random.PRNGKey(10_000 + a.init_seed)
    hist = []; t0 = time.time()
    nch = max(1, nsteps // CH)
    for i in range(nch):
        params, st, key, b = chunk(params, st, key)
        if (i + 1) % max(1, nch // 20) == 0 or i == nch - 1:
            hist.append(dict(step=(i + 1) * CH, bce=float(b), sec=round(time.time() - t0, 1)))
            print('TRAIN', json.dumps(hist[-1]), flush=True)
    t_train = time.time() - t0

    @jax.jit
    def sym_logit(p, x):
        x = x.astype(jnp.float32); acc = 0.
        for g in range(G):
            xg = x[:, invj[g]]
            acc = acc + model.apply({'params': p}, xg)[0] + model.apply({'params': p}, -xg)[0]
        return acc / (2 * G)

    t0 = time.time()
    Dn = len(states); B = int(2 ** np.floor(np.log2(max(1024, a.eval_mem / (N * len(nbr[0]) * ch))))); lg = np.empty(Dn, np.float32)
    for i in range(0, Dn, B):
        s = states[i:i + B]; m = len(s)
        if m < B: s = np.concatenate([s, np.repeat(s[:1], B - m)])
        lg[i:i + m] = np.asarray(sym_logit(params, jnp.asarray(unpack(s, N))))[:m]
    t_eval = time.time() - t0
    np.save(f'{out}/logits.npy', lg)
    pred = np.where(lg >= 0, 1, -1).astype(np.int8)
    wrong = pred != y
    seen = np.zeros(Dn, bool); seen[idx] = True
    res = dict(args={k: v for k, v in vars(a).items() if k not in ('tasks', 'lo', 'hi', 'force', 'eval_mem')}, name=run_name(a),
               nparams=npar, n_train=int(ntr), nsteps=nsteps, t_train=t_train, t_eval=t_eval,
               device=str(jax.devices()[0]), hist=hist,
               w_lab=float(p0[wrong].sum()), w_triv=float(p0[y < 0].sum()),
               err_train_unweighted=float(wrong[idx].mean()),
               p0_seen=float(p0[seen].sum()), w_lab_unseen=float(p0[wrong & ~seen].sum()),
               frac_orbits_wrong=float(wrong.mean()), sec=time.time() - t00)
    json.dump(res, open(f'{out}/run.json', 'w'), indent=1)
    print('RESULT', json.dumps({k: v for k, v in res.items() if k not in ('hist', 'args')}), flush=True)


if __name__ == '__main__':
    ap = parser(); a = ap.parse_args()
    if a.tasks:
        lines = [l.strip() for l in open(a.tasks) if l.strip() and not l.startswith('#')]
        for j, line in enumerate(lines[a.lo:a.hi], a.lo):
            aj = ap.parse_args(shlex.split(line) + (['--force'] if a.force else []))
            aj.data, aj.runs, aj.eval_mem = a.data, a.runs, a.eval_mem
            print(f'=== task {j}: {line}', flush=True)
            try:
                train_one(aj)
            except Exception as ex:
                import traceback; traceback.print_exc(); print('FAILED task', j, flush=True)
    else:
        train_one(a)
