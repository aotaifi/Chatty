"""GPU lattice fixed-node DMC/GFMC referee for SYMMETRIC guides on the 6x6 J1-J2 model (J2/J1 = 0.5), for calibration
against the exact symmetric-sector FN energy (experiments/stall_6x6).

Guide = per-orbit tables (la_g, s_g) on the 15.8M canonical representatives (sorted uint64 = psi0 table order); the guide
value of ANY full-basis configuration is the value at its canonical representative (min over 288 space-group images and
spin flip).  The full-space lattice FN ground state of this symmetric guide is the exact sector FN eigenvector of
stall_6x6 (st6_sector.Sector.fn_solve), so DMC - E_FN(exact) is a pure sampling/population bias.

The propagation algorithm is that of experiments/learned_loop_6x6/ll6_fn.py (= it2_fn.py), vectorised over populations:
  propagator 1 - tau (H_FN - Eref), Eref = mean local energy of the population,
  tau = min(tau_max, 0.8 / max(0, max(d_FN - Eref))), one move per walker per step (stay or one allowed hop),
  weight tot = stay + sum of allowed hop weights, systematic resampling every step,
  mixed estimator = mean local energy E_L = d_FN - sum_allowed J/2 a(y)/a(x) of the resampled population,
  averaged over steps with beta_after >= burn and beta_before < beta_target.
Initial walkers: M states of the 256-state pool energy_krylov_vs_vit_6x6_indep.npz (distinct if M <= 256, else with replacement).
Per population we store Eref_t and tau_t for all steps, so the estimator for any (burn, beta) window and the
population-weighted (Sorella-type correction-factor) estimators are evaluated offline (fncal_analyze.py).

Usage: python fncal_dmc.py GUIDE OUTDIR --cells M:NPOP,M:NPOP,... [--beta 2.4] [--seed 1] [--tau_max 0.025]
  GUIDE in {vitvit, vitex, exvit}   (amplitude, sign): |ViT|+ViT sign, |ViT|+exact sign, |psi0|+ViT sign
"""
import os, sys, json, time, argparse
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

L = 6; N = 36; J2 = 0.5
DATA = os.environ.get('FNCAL_DATA', '/project/theorie/a/A.Otaifi/chatty_stall6/data')
POOL = os.environ.get('FNCAL_POOL', os.path.expanduser('~/chatty_ll6/data/energy_krylov_vs_vit_6x6_indep.npz'))
from fncal_common import EXACT, E0_SITE, GUIDES, window_estimates


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


# ------------------------------------------------------------------ lattice (identical to ll6_core / st6_sector)
def _site(x, y):
    return (x % L) + L * (y % L)


def bonds():
    nn_, nnn = [], []
    for y in range(L):
        for x in range(L):
            i = _site(x, y)
            nn_ += [(i, _site(x + 1, y)), (i, _site(x, y + 1))]
            nnn += [(i, _site(x + 1, y + 1)), (i, _site(x + 1, y - 1))]
    return nn_, nnn


def space_group():
    def pm(f):
        p = np.empty(N, np.int64)
        for y in range(L):
            for x in range(L):
                xx, yy = f(x, y); p[_site(x, y)] = _site(xx, yy)
        return p
    c4 = pm(lambda x, y: (-y, x)); sx = pm(lambda x, y: (-x, y)); r = np.arange(N)
    pts = []
    for k in range(4):
        for refl in (False, True):
            q = r.copy()
            for _ in range(k): q = c4[q]
            if refl: q = sx[q]
            pts.append(q)
    out = []
    for ty in range(L):
        for tx in range(L):
            t = pm(lambda x, y: (x + tx, y + ty))
            for q in pts: out.append(t[q])
    return np.array(out)


NN, NNN = bonds()
BI = np.array([b[0] for b in NN + NNN], np.uint64); BJ = np.array([b[1] for b in NN + NNN], np.uint64)
JB = np.array([1.0] * len(NN) + [J2] * len(NNN))
NB = len(JB)
MASKS = (np.uint64(1) << BI) | (np.uint64(1) << BJ)
FULL = np.uint64((1 << N) - 1)
PERMS = space_group()            # (288, 36): bit i of x -> bit PERMS[g, i] of image


def byte_tables(perms):
    T = np.zeros((perms.shape[0], 5, 256), np.uint64)
    vals = np.arange(256, dtype=np.uint64)
    for b in range(5):
        for k in range(8):
            i = 8 * b + k
            if i >= N: break
            bit = (vals >> np.uint64(k)) & np.uint64(1)
            T[:, b, :] |= bit[None, :] << PERMS[:, i].astype(np.uint64)[:, None]
    return T


def apply_np(T, x):
    """x (n,) uint64 -> (n, 288) images."""
    img = np.zeros((len(x), T.shape[0]), np.uint64)
    for b in range(5):
        byte = ((x >> np.uint64(8 * b)) & np.uint64(255)).astype(np.intp)
        img |= T[:, b, :][:, byte].T
    return img


# ------------------------------------------------------------------ guide tables
def load_guide(name, data=DATA):
    """returns reps (uint64), la_g (float64, log of normalised clamped amplitude), s_g (float64 +-1) on the reps."""
    p = np.load(os.path.join(data, 'psi0_6x6_table.npz'))
    z = np.load(os.path.join(data, 'vit_reps_tables.npz'))
    reps = p['reps']; n = p['orbit'].astype(np.float64); amp0 = p['amp']
    la_v = z['la'].astype(np.float64); s_v = z['s'].astype(np.float64)
    la_0 = np.log(np.maximum(np.abs(amp0), 1e-300)); s_0 = np.where(amp0 >= 0, 1.0, -1.0)
    la, s = {'vitvit': (la_v, s_v), 'vitex': (la_v, s_0), 'exvit': (la_0, s_v)}[name]
    la = la - la.max()
    a = np.exp(la); a = a / np.sqrt(np.sum(n * a * a)); a = np.maximum(a, 1e-15)     # as st6_sector.fn_solve
    return reps, np.log(a), s


# ------------------------------------------------------------------ jitted local data / step
class Engine:
    def __init__(self, guide, chunk=512):
        reps, la, s = load_guide(guide)
        self.guide = guide; self.D = len(reps); self.chunk = chunk
        T = byte_tables(PERMS)
        Gm = apply_np(T, MASKS)                                  # (36?, ...) -> image of each bond mask: (144, 288)
        self.Gm2 = jnp.asarray(np.concatenate([Gm.T, Gm.T], 0))  # (576, 144): candidate k in [A, A^FULL] xor Gm2[k, b]
        self.T = jnp.asarray(T); self.reps = jnp.asarray(reps)
        self.la = jnp.asarray(la); self.s = jnp.asarray(s)
        self.BI = jnp.asarray(BI); self.BJ = jnp.asarray(BJ); self.MASKS = jnp.asarray(MASKS)
        self.JB = jnp.asarray(JB); self.diag0 = 0.25 * JB.sum()
        pool = np.load(POOL)['states'].astype(np.uint64).reshape(-1)
        self.pool = jnp.asarray(pool)
        self._build()

    def _build(self):
        T, reps, la, s, Gm2 = self.T, self.reps, self.la, self.s, self.Gm2
        BI, BJ, JB, MASKS, diag0 = self.BI, self.BJ, self.JB, self.MASKS, self.diag0
        FULLj = jnp.uint64(FULL)
        one = jnp.uint64(1)

        def local_chunk(x):                       # x (c,) uint64
            c = x.shape[0]
            A = jnp.zeros((c, T.shape[0]), jnp.uint64)
            for b in range(5):
                byte = ((x >> jnp.uint64(8 * b)) & jnp.uint64(255)).astype(jnp.int32)
                A = A | T[:, b, :][:, byte].T
            CS = jnp.concatenate([A, A ^ FULLj], 1)                              # (c, 576)
            mx = jnp.min(CS, 1)
            mc = jnp.min(CS[:, :, None] ^ Gm2[None, :, :], axis=1)               # (c, 144) canonical child
            ix = jnp.clip(jnp.searchsorted(reps, mx), 0, reps.shape[0] - 1)
            ic = jnp.clip(jnp.searchsorted(reps, mc.reshape(-1)), 0, reps.shape[0] - 1).reshape(c, NB)
            valid = (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & one).astype(bool)
            w = 0.5 * JB[None, :] * jnp.exp(la[ic] - la[ix][:, None]) * valid
            allowed = valid & (s[ix][:, None] * s[ic] < 0)
            diag = diag0 - 0.5 * jnp.sum(valid * JB[None, :], 1)
            wa = jnp.where(allowed, w, 0.0)
            dfn = diag + jnp.sum(w - wa, 1)
            eh = dfn - jnp.sum(wa, 1)
            return dfn, eh, wa

        self.local_chunk = local_chunk

        def local(x):                              # x (B,)
            B = x.shape[0]
            dfn, eh, wa = jax.lax.map(local_chunk, x.reshape(B // self.chunk, self.chunk))
            return dfn.reshape(B), eh.reshape(B), wa.reshape(B, NB)
        self.local = jax.jit(local)

        def step(x, key, beta, beta_target, tau_max):
            P, M = x.shape
            dfn, eh, wa = local(x.reshape(-1))
            dfn = dfn.reshape(P, M); eh = eh.reshape(P, M); wa = wa.reshape(P, M, NB)
            Eref = jnp.mean(eh, 1)
            mxd = jnp.maximum(0.0, jnp.max(dfn - Eref[:, None], 1))
            tau = jnp.minimum(tau_max, jnp.where(mxd > 0, 0.8 / jnp.maximum(mxd, 1e-300), tau_max))
            active = beta < beta_target
            tau = jnp.where(active, tau, 0.0)
            k1, k2, k3 = jax.random.split(key, 3)
            stay = 1.0 - tau[:, None] * (dfn - Eref[:, None])
            ws = tau[:, None, None] * wa
            tot = stay + jnp.sum(ws, 2)
            u = jax.random.uniform(k1, (P, M)) * tot
            cs = jnp.cumsum(ws, 2)
            j = jnp.sum(cs <= (u - stay)[:, :, None], 2)
            last = jnp.max(jnp.where(wa > 0, jnp.arange(NB)[None, None, :], 0), 2)
            j = jnp.minimum(j, last)
            hop = u >= stay
            y = jnp.where(hop, x ^ MASKS[j], x)
            bw = tot / jnp.sum(tot, 1, keepdims=True)
            c = jnp.cumsum(bw, 1); c = c.at[:, -1].set(1.0)
            pos = (jax.random.uniform(k2, (P, 1)) + jnp.arange(M)[None, :]) / M
            idx = jax.vmap(lambda cc, pp: jnp.searchsorted(cc, pp, side='right'))(c, pos)
            idx = jnp.minimum(idx, M - 1)
            xn = jnp.take_along_axis(y, idx, 1)
            ndist = 1 + jnp.sum(jnp.diff(idx, axis=1) != 0, 1)      # distinct parents after resampling
            return xn, Eref, tau, ndist, beta + tau, jnp.min(stay), k3
        self.step = jax.jit(step)

    def init_pop(self, key, P, M):
        pool = self.pool
        if M <= pool.shape[0]:
            idx = jax.vmap(lambda k: jax.random.permutation(k, pool.shape[0])[:M])(jax.random.split(key, P))
        else:
            idx = jax.random.randint(key, (P, M), 0, pool.shape[0])
        return pool[idx]

    def run_batch(self, key, P, M, beta_target, tau_max, nsteps_max=2000):
        kinit, key = jax.random.split(key)
        x = self.init_pop(kinit, P, M)
        beta = jnp.zeros(P)
        Er, Ta, Nd = [], [], []
        minstay = 1e9
        for it in range(nsteps_max):
            key, k = jax.random.split(key)
            x, Eref, tau, nd, beta, ms, _ = self.step(x, k, beta, beta_target, tau_max)
            Er.append(Eref); Ta.append(tau); Nd.append(nd)
            minstay = min(minstay, float(ms) if it % 10 == 0 else minstay)
            if it % 5 == 0 and not bool(jnp.any(beta < beta_target)): break
        # one more evaluation: Eref of the final resampled population (estimator of the last step)
        key, k = jax.random.split(key)
        x, Eref, tau, nd, beta, ms, _ = self.step(x, k, jnp.full(P, 1e9), beta_target, tau_max)
        Er.append(Eref); Ta.append(jnp.zeros(P)); Nd.append(nd)
        return (np.asarray(jnp.stack(Er, 1)), np.asarray(jnp.stack(Ta, 1)), np.asarray(jnp.stack(Nd, 1)), minstay)


def run_cell(eng, guide, outdir, M, npop, beta, seed, tau_max=0.025, walkers_per_batch=16384, tag=''):
    P = max(1, min(npop, walkers_per_batch // M))
    while (P * M) % eng.chunk: P += 1          # B must be a multiple of chunk
    tag = tag or f'{guide}_M{M}_b{beta}_s{seed}'
    out = os.path.join(outdir, f'cell_{tag}.npz')
    Es, Ts, Nds = [], [], []
    done = 0; bi = 0; t0 = time.time()
    while done < npop:
        key = jax.random.fold_in(jax.random.PRNGKey(seed), bi)
        t1 = time.time()
        Er, Ta, Nd, ms = eng.run_batch(key, P, M, beta, tau_max)
        Es.append(Er); Ts.append(Ta); Nds.append(Nd)
        e, nn = window_estimates(Er, Ta, 0.4, 1.2)
        done += P; bi += 1
        log(f'{guide} M={M} batch {bi} pops {done}/{npop} P={P} steps={Er.shape[1]} sec={time.time()-t1:.1f} minstay={ms:.3f} '
            f'E(0.4-1.2)={np.nanmean(e)/N:.7f} tau_mean={Ta[Ta>0].mean():.4f} distinct={Nd.mean()/M:.3f}')
        if bi % 10 == 0 or done >= npop:
            S = max(x.shape[1] for x in Es)
            pad = lambda arrs, fill, dt: np.concatenate([np.pad(x, ((0, 0), (0, S - x.shape[1])), constant_values=fill) for x in arrs], 0).astype(dt)
            np.savez(out, Eref=pad(Es, np.nan, np.float64), tau=pad(Ts, 0.0, np.float32), ndist=pad(Nds, 0, np.int32), M=M,
                     beta_target=beta, tau_max=tau_max, guide=guide, seed=seed, E_exact_site=EXACT[guide], sec=time.time() - t0, npop=done)
    log('CELL DONE', out, f'{time.time()-t0:.0f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('guide'); ap.add_argument('outdir')
    ap.add_argument('--cells', required=True, help='comma list M:NPOP, e.g. 128:4096,512:1024')
    ap.add_argument('--beta', type=float, default=2.4); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--tau_max', type=float, default=0.025)
    ap.add_argument('--walkers_per_batch', type=int, default=16384)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()
    eng = Engine(a.guide)
    log('engine ready', a.guide, 'D', eng.D, 'devices', jax.devices(), f'{time.time()-t0:.0f}s')
    for c in a.cells.split(','):
        M, npop = (int(v) for v in c.split(':'))
        run_cell(eng, a.guide, a.outdir, M, npop, a.beta, a.seed, a.tau_max, a.walkers_per_batch, a.tag)
    log('DONE', f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
