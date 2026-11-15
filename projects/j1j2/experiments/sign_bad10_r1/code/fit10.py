"""T-BAD10 round 1, stage FIT: n_1 = Marshall x sgn(logit), 64x10 CNN (screen arm d: 64 channels x 10 layers, from scratch, D4 x flip averaged logit), BCE on y (K[M] relative to Marshall),
AdamW lr 2e-3 warmup(steps/50)-cosine, wd 1e-4, batch 1024, one random D4 x flip image per sample, 30 steps per root; learning curve over --roots.  Train roots: the first n roots of the
train chains (id % 5 != 0, ascending chain id, nsamp roots per chain) and ALL states of their complete neighbourhoods.  Held-out: states whose origin chain is held out (id % 5 == 0) and that occur
in no train root's neighbourhood (leak-free); E_L error on the held-out roots' complete neighbourhoods:  E_L(|ViT| n; x) with n = net on x and N1(x) vs E_L(|ViT| K[M]; x) (exact labels), vs Marshall.
usage: python fit10.py OUT LABELS_DIR --roots 1000 2000 3200 [--steps-per-root 30]"""
import sys, os, json, time, glob, hashlib, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('labels'); ap.add_argument('--roots', type=int, nargs='+', default=[1000, 2000, 3200])
ap.add_argument('--steps-per-root', type=float, default=30); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--nheld-log', type=int, default=20000)
ap.add_argument('--ch', type=int, default=64); ap.add_argument('--depth', type=int, default=10); ap.add_argument('--min_steps', type=int, default=100)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
log = lambda *s: print(time.strftime('[%H:%M:%S]'), *s, flush=True)
import jax, jax.numpy as jnp, optax
import vitlib as VL, s10_net as SN
from vitlib import BI, BJ, HW, JB, N
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
log('device', jax.devices())
def unpack(B): return np.where(np.unpackbits(B, axis=1)[:, :N] > 0, 1, -1).astype(np.int8)

def build_table(d):
    files = sorted(glob.glob(d + '/chunk_*.npz')); assert files, 'no label chunks'
    R_b = []; R_ch = []; R_n1 = []; L1b = []; Y = []; LP = []; EL = []; RR = []
    for f in files:
        c = np.load(f); R_b.append(c['roots_b']); R_ch.append(c['chain']); R_n1.append(c['n1']); L1b.append(c['L1b']); Y.append(c['y']); LP.append(c['lp1']); EL.append(c['ELM']); RR.append(c['r'])
    rb = np.concatenate(R_b); rch = np.concatenate(R_ch); n1 = np.concatenate(R_n1); L1b = np.concatenate(L1b); y = np.concatenate(Y); lp = np.concatenate(LP); ELM = np.concatenate(EL); rr = np.concatenate(RR)
    off = np.concatenate([[0], np.cumsum(n1)]); mem_root = np.repeat(np.arange(len(n1)), n1)
    void = np.ascontiguousarray(L1b).view(np.dtype((np.void, L1b.shape[1]))).ravel(); u, first, inv = np.unique(void, return_index=True, return_inverse=True)
    assert np.all(y[first][inv] == y), 'label not a function of the state'
    origin = rch[mem_root[first]]                                    # chain of the first root (data order) whose neighbourhood contains the state
    # pairs offset per root for r
    npairs_state = np.add.reduceat(np.zeros(1), [0]) if False else None
    T = dict(states_b=L1b[first], y=y[first], lp=lp[first], origin=origin, mem_root=mem_root, mem_state=inv, rchain=rch, n1=n1, off=off, L1b_all=L1b, r=rr, ELM_all=ELM, y_all=y)
    return T

T = build_table(a.labels); nroots = len(T['n1']); train_idx = np.nonzero(T['rchain'] % 5 != 0)[0]; train_idx = train_idx[np.lexsort((train_idx, T['rchain'][train_idx]))]
held_idx = np.nonzero(T['rchain'] % 5 == 0)[0]; states = unpack(T['states_b']); yS = T['y']
def sets(n):
    sel = train_idx[:n]; assert len(sel) == n, ('not enough train roots', len(sel), n)
    m = np.isin(T['mem_root'], sel); return sel, np.unique(T['mem_state'][m])
all_tr = sets(len(train_idx))[1]; hmask = (T['origin'] % 5 == 0); hmask[all_tr] = False; hv_all = np.nonzero(hmask)[0]
log('TABLE', json.dumps(dict(states=len(yS), roots=nroots, train_roots=len(train_idx), held_roots=len(held_idx), held_states_leakfree=len(hv_all), minus_share=float((yS < 0).mean()))))
MREF = None
try: MREF = json.load(open(a.labels + '/sampler_meta.json'))['mref10']
except Exception: MREF = float(np.median(2 * T['lp']))
LN10 = float(np.log(10.0))

def marsh(X): return VL.marshall(X).astype(np.float64)
def evaluate(net):
    lo = net.logit(states[hv_all].astype(np.float32)); pred = np.where(lo >= 0, 1, -1); wrong = pred != yS[hv_all]
    d = np.floor((2 * T['lp'][hv_all] - MREF) / LN10).astype(int)
    out = dict(n_held_states=int(len(hv_all)), disagree=float(wrong.mean()), triv_noflip=float((yS[hv_all] < 0).mean()), on_flips=float(wrong[yS[hv_all] < 0].mean()), on_keeps=float(wrong[yS[hv_all] > 0].mean()))
    dec = []
    for k in sorted(set(d.tolist())):
        m = d == k
        if m.sum() >= 1: dec.append(dict(decade=int(k), n=int(m.sum()), disagree=float(wrong[m].mean()), flip_frac=float((yS[hv_all][m] < 0).mean())))
    out['by_decade'] = dec
    # E_L error on held-out roots' complete neighbourhoods
    r_off = np.zeros(nroots + 1, np.int64)
    # pairs per root: sum over its L1 states of antiparallel bonds
    L1all = unpack(T['L1b_all']); npair_state = (L1all[:, BI] != L1all[:, BJ]).sum(1); pr = np.add.reduceat(npair_state, T['off'][:-1]); r_off[1:] = np.cumsum(pr)
    eN = []; eL = []; eM = []; ar = np.arange
    for ri in held_idx:
        s0, n1 = T['off'][ri], T['n1'][ri]; L1 = L1all[s0:s0 + n1].astype(np.float32); x = L1[0]; nb = np.nonzero(x[BI] != x[BJ])[0]; r = T['r'][r_off[ri]:r_off[ri] + len(nb)].astype(np.float64)
        nn = np.where(net.logit(L1) >= 0, 1, -1) * marsh(L1); lab = marsh(L1) * T['y_all'][s0:s0 + n1]; mm = marsh(L1)
        dg = VL.diag0(x[None])[0]
        f = lambda s: dg + s[0] * np.sum(HW[nb] * s[1:] * r)
        eN.append(f(nn)); eL.append(f(lab)); eM.append(f(mm))
    eN, eL, eM = map(np.array, (eN, eL, eM)); dE = eN - eL
    out['EL'] = dict(n_roots=int(len(held_idx)), mean_abs_err=float(np.abs(dE).mean()), rms_err=float(np.sqrt((dE ** 2).mean())), mean_signed=float(dE.mean()), mean_abs_err_per_site=float(np.abs(dE).mean() / N),
                     frac_gt_1=float((np.abs(dE) > 1).mean()), mean_EL_net=float(eN.mean()), mean_EL_label=float(eL.mean()), mean_EL_marshall=float(eM.mean()),
                     base_mean_abs_err_Marshall_vs_label=float(np.abs(eM - eL).mean()), base_rms_Marshall_vs_label=float(np.sqrt(((eM - eL) ** 2).mean())))
    return out

def train(net, Xtr, ytr, hvX, hvy, steps, lr, batch=1024, seed=0, nlog=20):
    m = net.m; params = net.params; sched = optax.warmup_cosine_decay_schedule(0., lr, max(1, steps // 50), steps, end_value=0.0)
    opt = optax.adamw(sched, weight_decay=1e-4); st = opt.init(params)
    Sj = jnp.asarray(Xtr); yj = jnp.asarray((ytr > 0).astype(np.float32)); n = len(Xtr)
    def loss_fn(p, X, yb): return jnp.mean(optax.sigmoid_binary_cross_entropy(m.apply({'params': p}, X), yb))
    def body(carry, _):
        p, st, key = carry; key, k1, k2 = jax.random.split(key, 3)
        ib = jax.random.randint(k1, (batch,), 0, n); X = Sj[ib].astype(jnp.float32); gi = jax.random.randint(k2, (batch,), 0, 16)
        Xi = jnp.take_along_axis(X, SN.PERM[gi // 2], axis=1) * SN.SG[gi % 2][:, None]
        l, gr = jax.value_and_grad(loss_fn)(p, Xi, yj[ib]); u, st = opt.update(gr, st, p)
        return (optax.apply_updates(p, u), st, key), l
    every = max(1, steps // nlog)
    @jax.jit
    def run(p, st, key):
        (p, st, key), ls = jax.lax.scan(body, (p, st, key), None, length=every); return p, st, key, ls.mean()
    key = jax.random.PRNGKey(seed); hist = []; t0 = time.time(); done = 0
    while done < steps:
        params, st, key, lm = run(params, st, key); done += every; net.params = params
        lv = net.logit(hvX); err = float(np.mean(np.where(lv >= 0, 1, -1) != hvy)); h = dict(step=done, bce=float(lm), held_err=err, sec=time.time() - t0); hist.append(h); log('TRAIN', json.dumps(h))
    return hist

for n in a.roots:
    sel, st = sets(n); steps = max(a.min_steps, int(round(a.steps_per_root * n / 100.0) * 100)); rg = np.random.default_rng(a.seed + n)
    hv = rg.choice(hv_all, min(a.nheld_log, len(hv_all)), replace=False)
    net = SN.Net10(a.ch, a.depth, seed=a.seed, mode='random'); t0 = time.time()
    log('FIT', json.dumps(dict(roots=n, train_states=len(st), steps=steps, npar=net.npar, train_flip_share=float((yS[st] < 0).mean()))))
    hist = train(net, states[st], yS[st], states[hv].astype(np.float32), yS[hv], steps, a.lr, seed=a.seed)
    ps = os.path.join(a.out, f'net_{n}.pkl'); net.save(ps); sh = sha(ps)
    res = dict(roots=n, train_states=int(len(st)), steps=steps, train_sec=time.time() - t0, net=ps, sha256=sh, hist=hist, npar=net.npar)
    t1 = time.time(); res.update(evaluate(net)); res['eval_sec'] = time.time() - t1
    sub = states[st][:50000].astype(np.float32); res['train_err'] = float(np.mean(np.where(net.logit(sub) >= 0, 1, -1) != yS[st][:50000]))
    json.dump(res, open(os.path.join(a.out, f'fit_{n}.json'), 'w'), indent=1, default=float)
    log('FIT_DONE', json.dumps({k: v for k, v in res.items() if k not in ('hist', 'by_decade')}, default=float)); log('BY_DECADE', json.dumps(res['by_decade']))
    last = (ps, sh)
open(os.path.join(a.out, 'final_net.txt'), 'w').write(f'{last[0]} {last[1]}\n'); log('R1_FIT_DONE')
