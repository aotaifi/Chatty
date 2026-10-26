"""Stage A' (PREREG.md, Amendment 1): one global Krylov step G = psi (1 + alpha (E_L - E)) with CACHED bond densities on a
finite-receptive-field CNN guide, O(N) per sweep.   python gcache.py OUTDIR correct|timing [options]

Guide: L x L torus, spins s_i = +-1 (Sz = s/2), Sz = 0.  R conv layers 3x3 (periodic), tanh, C channels, linear readout per
site, log|psi| = sum_i phi_i (sum readout, receptive field (2R+1)^2), psi = Marshall sign x exp(log|psi|).  Random real weights.
H = J1 sum_<ij> S_i.S_j + J2 sum_<<ij>> S_i.S_j, J2 = 0.5.  Bonds: 4 per anchor site (r,c): (r,c)-(r,c+1), (r,c)-(r+1,c) [J1],
(r,c)-(r+1,c+1), (r,c+1)-(r+1,c) [J2]; every bond lies in the 2x2 box anchored at (r,c).
  eps_b(x) = J_b [s_i s_j/4 + [s_i != s_j] (1/2) sgn_b psi_amp(x^b)/psi_amp(x)],  sgn_b = -1 (J1, Marshall), +1 (J2);  E_L = sum_b eps_b.
Patch-local ratio: log|psi|(x^b) - log|psi|(x) = sum over output sites within R of b's box of [phi(x^b) - phi(x)], evaluated on
a (2+4R)^2 input crop with VALID convolutions (outputs (2+2R)^2; duplicated torus sites masked when L < 2+2R).  Never a full pass.
Cached step for a proposed exchange c (y = x^c): only bonds b in A_c = {b : torus Chebyshev distance(endpoints of b, sites of c)
<= 2R} can change eps_b (their ratio depends on inputs within 2R of their endpoints), so E_L(y) = E_L(x) + sum_{b in A_c}
[eps_b(y) - eps_b(x)], psi_G(y)/psi_G(x) = [psi(y)/psi(x)] G(y)/G(x).  Local energy of psi_G:
  E_LG(x) = sum_c [diag_c + off_c(x) G(x^c)/G(x)],  off_c = eps_c - diag_c,  G(x^c) from the same cached update.
Direct control (timing): E_L(y) recomputed from all 4N bonds (patch-local ratios, no cache).  Reference (correctness): full
periodic forward passes of the whole lattice for every configuration (independent code path), float64.
Broken control: A_c minus its farthest bond.
"""
import os, sys, json, time, argparse
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

J2 = 0.5


def log(out, *a):
    s = time.strftime('%Y-%m-%dT%H:%M:%S') + ' PROGRESS ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    if out: open(os.path.join(out, 'progress.txt'), 'a').write(s + '\n')


# ------------------------------------------------------------------ tables (numpy)
def tables(L, R, drop_farthest=False):
    N = L * L
    st = lambda r, c: (r % L) * L + (c % L)
    I, J, Jb, sg, anc = [], [], [], [], []
    for r in range(L):
        for c in range(L):
            for a, b, t in (((r, c), (r, c + 1), 0), ((r, c), (r + 1, c), 0), ((r, c), (r + 1, c + 1), 1), ((r, c + 1), (r + 1, c), 1)):
                I.append(st(*a)); J.append(st(*b)); Jb.append(1.0 if t == 0 else J2); sg.append(-1.0 if t == 0 else 1.0); anc.append((r, c))
    nb = len(I)
    Sin, Sout = 2 + 4 * R, 2 + 2 * R
    crop = np.zeros((nb + 1, Sin, Sin), np.int32); flip = np.zeros((nb + 1, Sin, Sin), bool)
    for b, (r, c) in enumerate(anc):
        rr = (r - 2 * R + np.arange(Sin)) % L; cc = (c - 2 * R + np.arange(Sin)) % L
        crop[b] = rr[:, None] * L + cc[None, :]
        flip[b] = (crop[b] == I[b]) | (crop[b] == J[b])
    m = np.array([all((d - e) % L != 0 for e in range(d)) for d in range(Sout)], float)
    outmask = m[:, None] * m[None, :]
    # torus Chebyshev distance between sites
    rs, cs = np.divmod(np.arange(N), L)
    dr = np.abs(rs[:, None] - rs[None, :]); dr = np.minimum(dr, L - dr)
    dc = np.abs(cs[:, None] - cs[None, :]); dc = np.minimum(dc, L - dc)
    D = np.maximum(dr, dc)
    I = np.array(I); J = np.array(J)
    sets = []
    for c in range(nb):
        d = np.minimum.reduce([D[I, I[c]], D[I, J[c]], D[J, I[c]], D[J, J[c]]])
        idx = np.nonzero(d <= 2 * R)[0]
        if drop_farthest:
            idx = np.delete(idx, np.argmax(d[idx]))
        sets.append(idx)
    K = max(len(s) for s in sets)
    A = np.full((nb, K), nb, np.int32)
    for c, s in enumerate(sets): A[c, :len(s)] = s
    dummy = lambda v, x: np.concatenate([np.asarray(v), [x]])
    return dict(L=L, N=N, nb=nb, R=R, K=K, Kmean=float(np.mean([len(s) for s in sets])),
                I=dummy(I, 0).astype(np.int32), J=dummy(J, 1).astype(np.int32), Jb=dummy(Jb, 0.0), sg=dummy(sg, 1.0),
                crop=crop, flip=flip, outmask=outmask, A=A)


def to_dev(T, dt):
    out = {}
    for k, v in T.items():
        if isinstance(v, np.ndarray):
            out[k] = jnp.asarray(v, dt) if v.dtype == np.float64 else jnp.asarray(v)
        else:
            out[k] = v
    return out


def init_params(key, R, C, dt, scale=1.0, wscale=0.3):
    conv = []; cin = 1
    for l in range(R):
        key, k1, k2 = jax.random.split(key, 3)
        W = scale * jax.random.normal(k1, (3, 3, cin, C), jnp.float64) / np.sqrt(9 * cin)
        b = 0.1 * jax.random.normal(k2, (C,), jnp.float64)
        conv.append((W.astype(dt), b.astype(dt))); cin = C
    key, k3 = jax.random.split(key)
    w = (wscale * jax.random.normal(k3, (C,), jnp.float64)).astype(dt)
    return dict(conv=conv, w=w)


DN = ('NHWC', 'HWIO', 'NHWC')


def cnn_valid(p, x):          # x (M, S, S) -> phi (M, S-2R, S-2R)
    h = x[..., None]
    for W, b in p['conv']:
        h = jnp.tanh(jax.lax.conv_general_dilated(h, W, (1, 1), 'VALID', dimension_numbers=DN) + b)
    return h @ p['w']


def logamp_full(p, x, L):     # x (M, N) -> log|psi| (M,), full periodic pass (reference path)
    h = x.reshape(-1, L, L, 1)
    for W, b in p['conv']:
        h = jnp.pad(h, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='wrap')
        h = jnp.tanh(jax.lax.conv_general_dilated(h, W, (1, 1), 'VALID', dimension_numbers=DN) + b)
    return (h @ p['w']).sum((1, 2))


# ------------------------------------------------------------------ patch-local pieces (single chain; vmap/map outside)
def dlog_bonds(p, T, x, bids):
    old = x[T['crop'][bids]]
    new = jnp.where(T['flip'][bids], -old, old)
    M = old.shape[0]
    phi = cnn_valid(p, jnp.concatenate([old, new], 0))
    return ((phi[M:] - phi[:M]) * T['outmask']).sum((1, 2))


def eps_bonds(p, T, x, bids):
    xi = x[T['I'][bids]]; xj = x[T['J'][bids]]
    dl = dlog_bonds(p, T, x, bids)
    anti = xi != xj
    return T['Jb'][bids] * (xi * xj / 4 + jnp.where(anti, 0.5 * T['sg'][bids] * jnp.exp(jnp.where(anti, dl, 0.0)), 0.0))


def swap(T, x, c):
    i, j = T['I'][c], T['J'][c]
    return x.at[i].set(x[j]).at[j].set(x[i])


def Gf(EL, alpha, E): return 1.0 + alpha * (EL - E)


def eps_all(p, T, x):
    return jnp.concatenate([eps_bonds(p, T, x, jnp.arange(T['nb'])), jnp.zeros((1,), x.dtype)])


def update_cached(p, T, A, x, eps, EL, c):
    """y = x^c; returns y, bids (A_c), eps_b(y) on A_c, E_L(y)."""
    y = swap(T, x, c)
    bids = A[c]
    en = eps_bonds(p, T, y, bids)
    return y, bids, en, EL + (en - eps[bids]).sum()


def ratio_psi(p, T, x, c):
    anti = x[T['I'][c]] != x[T['J'][c]]
    dl = dlog_bonds(p, T, x, c[None])[0]
    return jnp.where(anti, T['sg'][c] * jnp.exp(jnp.where(anti, dl, 0.0)), 1.0)


def ratio_G_cached(p, T, A, x, eps, EL, c, alpha, E):
    y, bids, en, ELy = update_cached(p, T, A, x, eps, EL, c)
    return ratio_psi(p, T, x, c) * Gf(ELy, alpha, E) / Gf(EL, alpha, E), ELy


def ELG_cached(p, T, A, x, eps, EL, alpha, E, bs):
    def per_c(c):
        _, _, _, ELy = update_cached(p, T, A, x, eps, EL, c)
        diag = T['Jb'][c] * x[T['I'][c]] * x[T['J'][c]] / 4
        return diag + (eps[c] - diag) * Gf(ELy, alpha, E) / Gf(EL, alpha, E)
    return jax.lax.map(per_c, jnp.arange(T['nb']), batch_size=bs).sum()


def ELG_direct(p, T, x, EL, alpha, E, bs):
    eps = eps_all(p, T, x)
    def per_c(c):
        ELy = eps_all(p, T, swap(T, x, c)).sum()
        diag = T['Jb'][c] * x[T['I'][c]] * x[T['J'][c]] / 4
        return diag + (eps[c] - diag) * Gf(ELy, alpha, E) / Gf(EL, alpha, E)
    return jax.lax.map(per_c, jnp.arange(T['nb']), batch_size=bs).sum()


# ------------------------------------------------------------------ sweeps (batch of chains)
def make_sweep(p, T, A, alpha, E, mode, bc):
    nb, N = T['nb'], T['N']
    def step_chain(args):
        x, eps, EL, c, u = args
        if mode == 'cached':
            y, bids, en, ELy = update_cached(p, T, A, x, eps, EL, c)
        else:
            y = swap(T, x, c); ELy = eps_all(p, T, y).sum()
        r = ratio_psi(p, T, x, c) * Gf(ELy, alpha, E) / Gf(EL, alpha, E)
        acc = u < r * r
        x = jnp.where(acc, y, x); EL = jnp.where(acc, ELy, EL)
        if mode == 'cached':
            eps = jnp.where(acc, eps.at[bids].set(en), eps)
        return x, eps, EL, acc
    def sweep(X, EPS, EL, key):
        def body(t, carry):
            X, EPS, EL, key, nacc = carry
            key, k1, k2 = jax.random.split(key, 3)
            c = jax.random.randint(k1, (X.shape[0],), 0, nb)
            u = jax.random.uniform(k2, (X.shape[0],), X.dtype)
            X, EPS, EL, acc = jax.lax.map(step_chain, (X, EPS, EL, c, u), batch_size=bc)
            return X, EPS, EL, key, nacc + acc.sum()
        return jax.lax.fori_loop(0, N, body, (X, EPS, EL, key, 0))
    return jax.jit(sweep)


def random_configs(rng, B, N, dt):
    X = np.ones((B, N)); X[:, N // 2:] = -1
    for b in range(B): rng.shuffle(X[b])
    return jnp.asarray(X, dt)


# ------------------------------------------------------------------ correctness (float64)
def make_reference(p, T, alpha, E, bs=8):
    """Full-pass reference (jitted once): for one config x, E_L(x), E_L(x^c) and psi_G(x^c)/psi_G(x) for all c, and E_LG(x).
    Every log|psi| here is a full periodic forward pass of the whole lattice (no crops, no caches)."""
    L, nb = T['L'], T['nb']
    I, J, Jb, sg = (jnp.asarray(np.asarray(T[k])[:nb]) for k in ('I', 'J', 'Jb', 'sg'))
    ar = jnp.arange(nb)
    def swaps(x):
        return jnp.repeat(x[None], nb, 0).at[ar, I].set(x[J]).at[ar, J].set(x[I])
    def EL_full(x):
        l = logamp_full(p, jnp.concatenate([x[None], swaps(x)], 0), L)
        anti = x[I] != x[J]
        return (Jb * (x[I] * x[J] / 4 + jnp.where(anti, 0.5 * sg * jnp.exp(l[1:] - l[0]), 0.0))).sum()
    def ref(x):
        EL = EL_full(x)
        Y = swaps(x)
        ELy = jax.lax.map(EL_full, Y, batch_size=bs)
        l = logamp_full(p, jnp.concatenate([x[None], Y], 0), L)
        anti = x[I] != x[J]
        rpsi = jnp.where(anti, sg * jnp.exp(l[1:] - l[0]), 1.0)
        rG = rpsi * Gf(ELy, alpha, E) / Gf(EL, alpha, E)
        ELG = (Jb * x[I] * x[J] / 4 + jnp.where(anti, 0.5 * Jb * rG, 0.0)).sum()
        return dict(EL=EL, ELy=ELy, rG=rG, anti=anti, ELG=ELG)
    return jax.jit(ref), jax.jit(jax.vmap(EL_full))


def run_correct(out, L, R, C, ncfg, seed, res):
    dt = jnp.float64
    T = to_dev(tables(L, R), dt); Tb = to_dev(tables(L, R, drop_farthest=True), dt)
    p = init_params(jax.random.PRNGKey(seed), R, C, dt)
    rng = np.random.default_rng(seed + 1)
    X = random_configs(rng, ncfg, L * L, dt)
    epsf = jax.jit(jax.vmap(lambda x: eps_all(p, T, x)))
    EPS = epsf(X); EL = EPS.sum(1)
    E = float(EL.mean()); alpha = 0.2 / float(EL.std() + 1e-12)
    log(out, f'correct L={L} R={R} C={C} ncfg={ncfg} nb={T["nb"]} |A_c| max {T["K"]} mean {T["Kmean"]:.1f}; E_L mean {E:.4f} std {float(EL.std()):.4f} alpha {alpha:.4g}')
    allc = jnp.arange(T['nb'])
    reff, ELfull_v = make_reference(p, T, alpha, E, bs=max(1, 4096 // T['nb']))
    def cached_one(TT, x, eps, el):
        rG, ELy = jax.vmap(lambda c: ratio_G_cached(p, TT, TT['A'], x, eps, el, c, alpha, E))(allc)
        return rG, ELy, ELG_cached(p, TT, TT['A'], x, eps, el, alpha, E, 64)
    cj = jax.jit(lambda x, eps, el: cached_one(T, x, eps, el)); cb = jax.jit(lambda x, eps, el: cached_one(Tb, x, eps, el))
    st = dict(rG=[], rG_b=[], ELG=[], ELG_b=[], ELy=[], EL=[])
    for k in range(ncfg):
        ref = reff(X[k])
        rG, ELy, ELG = cj(X[k], EPS[k], EL[k]); rGb, ELyb, ELGb = cb(X[k], EPS[k], EL[k])
        a = np.asarray(ref['anti'])
        rel = lambda u, v: np.abs(np.asarray(u) - np.asarray(v)) / np.abs(np.asarray(v))
        st['rG'].append(rel(rG, ref['rG'])[a].max()); st['rG_b'].append(rel(rGb, ref['rG'])[a].max())
        st['ELG'].append(float(rel(ELG, ref['ELG']))); st['ELG_b'].append(float(rel(ELGb, ref['ELG'])))
        st['ELy'].append(rel(ELy, ref['ELy']).max()); st['EL'].append(float(rel(EL[k], ref['EL'])))
        if k % 10 == 9 or L >= 16: log(out, f'correct L={L} config {k + 1}/{ncfg}: rG rel {st["rG"][-1]:.2e} (broken {st["rG_b"][-1]:.2e}), E_LG rel {st["ELG"][-1]:.2e} (broken {st["ELG_b"][-1]:.2e})')
    r = {k: dict(max=float(np.max(v)), median=float(np.median(v))) for k, v in st.items()}
    # cache consistency after cached Metropolis sweeps with acceptances
    sw = make_sweep(p, T, T['A'], alpha, E, 'cached', 64)
    X2, EPS2, EL2, _, nacc = sw(X, EPS, EL, jax.random.PRNGKey(seed + 2))
    X2, EPS2, EL2, _, nacc2 = sw(X2, EPS2, EL2, jax.random.PRNGKey(seed + 3))
    EPSf = epsf(X2)
    r['after_2_sweeps'] = dict(accepted=int(nacc + nacc2), proposals=int(2 * ncfg * L * L),
                               eps_cache_maxabs=float(jnp.abs(EPS2 - EPSf).max()),
                               EL_rel_max=float((jnp.abs(EL2 - EPSf.sum(1)) / jnp.abs(EPSf.sum(1))).max()),
                               EL_vs_fullpass_rel_max=float((jnp.abs(EL2 - ELfull_v(X2)) / jnp.abs(EL2)).max()))
    r.update(L=L, R=R, C=C, ncfg=ncfg, nb=T['nb'], A_max=T['K'], A_mean=T['Kmean'], alpha=alpha, E=E)
    res[f'correct_L{L}_R{R}'] = r
    log(out, f'correct L={L}: rel err psi_G ratio max {r["rG"]["max"]:.3e} (broken {r["rG_b"]["max"]:.3e}); E_LG max {r["ELG"]["max"]:.3e} (broken {r["ELG_b"]["max"]:.3e}); E_L(x^c) max {r["ELy"]["max"]:.3e}; after sweeps {r["after_2_sweeps"]}')


# ------------------------------------------------------------------ timing (float32)
def timeit(f, args, nrep):
    ts = []
    for _ in range(nrep):
        t0 = time.perf_counter(); o = f(*args); jax.block_until_ready(o); ts.append(time.perf_counter() - t0)
    return ts


def run_timing(out, L, R, C, B, Bel, modes, nrep, seed, res, crop_budget):
    dt = jnp.float32
    T = to_dev(tables(L, R), dt)
    p = init_params(jax.random.PRNGKey(seed), R, C, dt)
    rng = np.random.default_rng(seed + 1)
    X = random_configs(rng, B, L * L, dt)
    nb = T['nb']
    epsf = jax.jit(lambda X: jax.lax.map(lambda x: eps_all(p, T, x), X, batch_size=max(1, crop_budget // nb)))
    EPS = epsf(X); EL = EPS.sum(1); jax.block_until_ready(EL)
    E = float(EL.mean()); alpha = 0.2 / float(EL.std() + 1e-6)
    r = dict(L=L, N=L * L, R=R, C=C, B=B, Bel=Bel, nb=nb, A_max=T['K'], A_mean=T['Kmean'], dtype='float32')
    # plain psi local energy, batch B
    t = timeit(epsf, (X,), nrep); r['EL_plain_s_per_batch'] = t; r['EL_plain_s_per_sample'] = min(t) / B
    for mode in modes:
        K = T['K'] if mode == 'cached' else nb
        bc = max(1, min(B, crop_budget // K))
        sw = make_sweep(p, T, T['A'], alpha, E, mode, bc)
        key = jax.random.PRNGKey(seed + 7)
        t0 = time.perf_counter(); swc = sw.lower(X, EPS, EL, key).compile(); tc = time.perf_counter() - t0
        t = timeit(swc, (X, EPS, EL, key), nrep)
        o = swc(X, EPS, EL, key)
        r[f'sweep_{mode}'] = dict(s_per_sweep=t, s_min=min(t), compile_s=tc, chain_chunk=bc, acc_rate=float(o[4]) / (B * L * L))
        log(out, f'timing L={L} sweep {mode}: {min(t):.3f} s/sweep (reps {[round(v, 3) for v in t]}), chunk {bc}, acc {r[f"sweep_{mode}"]["acc_rate"]:.3f}')
        Xe, EPSe, ELe = X[:Bel], EPS[:Bel], EL[:Bel]
        bs = max(1, crop_budget // K)
        if mode == 'cached':
            f = jax.jit(lambda X, EPS, EL: jax.vmap(lambda x, e, l: ELG_cached(p, T, T['A'], x, e, l, alpha, E, max(1, bs // Bel)))(X, EPS, EL))
        else:
            f = jax.jit(lambda X, EPS, EL: jax.vmap(lambda x, l: ELG_direct(p, T, x, l, alpha, E, max(1, bs // Bel)))(X, EL))
        f(Xe, EPSe, ELe).block_until_ready()
        t = timeit(f, (Xe, EPSe, ELe), nrep)
        r[f'ELG_{mode}'] = dict(s_per_batch=t, s_per_sample=min(t) / Bel)
        log(out, f'timing L={L} E_LG {mode}: {min(t) / Bel * 1e3:.3f} ms/sample (batch {Bel}); plain E_L {r["EL_plain_s_per_sample"] * 1e3:.4f} ms/sample (batch {B})')
    res[f'timing_L{L}_R{R}'] = r


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('task', choices=['correct', 'timing'])
    ap.add_argument('--L', type=int, nargs='+', default=[6, 8]); ap.add_argument('--R', type=int, default=3)
    ap.add_argument('--C', type=int, default=8); ap.add_argument('--ncfg', type=int, nargs='+', default=[100])
    ap.add_argument('--B', type=int, default=1024); ap.add_argument('--Bel', type=int, default=64)
    ap.add_argument('--modes', nargs='+', default=['cached', 'direct']); ap.add_argument('--nrep', type=int, default=2)
    ap.add_argument('--crop_budget', type=int, default=1 << 17); ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    resf = os.path.join(a.out, f'{a.task}{a.tag}.json')
    res = json.load(open(resf)) if os.path.exists(resf) else {}
    log(a.out, f'start {a.task} {vars(a)} jax {jax.__version__} {jax.devices()}')
    for k, L in enumerate(a.L):
        if a.task == 'correct':
            run_correct(a.out, L, a.R, a.C, a.ncfg[min(k, len(a.ncfg) - 1)], a.seed, res)
        else:
            run_timing(a.out, L, a.R, a.C, a.B, a.Bel, a.modes, a.nrep, a.seed, res, a.crop_budget)
        open(resf, 'w').write(json.dumps(res, indent=1))
    log(a.out, f'done {a.task}')
