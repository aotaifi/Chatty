"""Lattice fixed-node GFMC (p = 0) with a pluggable guide: any batched function logpsi(sigma (B, N) +-1) -> complex (B,).
Guide c(x) = Re psi(x) (its sign = sign Re psi; a global phase w is folded into logpsi as + log w).  J1-J2, J2 = 0.5, L x L PBC.

Walker dynamics, branching and estimators are the LAUNCHED M1 code unchanged (m1_code/gfmc_fn_lib.py, md5 40a1c409: GFMC, GuideBase.finish,
ten Haaf FN: kept hop rate -K r for r = c(y)/c(x) < 0, sign-violating K r > 0 moved to the diagonal = V_sf).  Only the guide provider is new:
LogpsiGuide.rows evaluates logpsi at x and at every one-swap neighbour y (1 + n_anti(x) evaluations per hop), ratio
  r = c(y)/c(x) = Re[exp(l_y - Re l_x)] / Re[exp(i Im l_x)].
Optional mixed precision (ViT): logpsi_lo (f32 'highest') for all pairs; pairs with |psi(y)/psi(x)|_lo < thr (1e-3, the A0.3 held-out
trigger) are recomputed with logpsi_hi (f64, x and y).  drop_vsf=True is the deliberately broken control (V_sf omitted).

  python gfmc_guide.py OUT --guide {rbmpp|table6|vit10} [guide options] --L L --M 256 --dtau 0.1 --blocks B --burn b --seed s
OUT/progress.txt (PROGRESS lines, last 'GFMC_GUIDE_DONE ...'), ckpt.npz (atomic, every --ckpt_every blocks, RNG state; --resume),
blocks.npz, E_mix.npy (per-block mixed estimator / N after burn, time order: quotable.py series input), result.json.
"""
import argparse, json, os, sys, time, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); M1 = os.path.join(HERE, 'm1_code')
sys.path.insert(0, M1)
from gfmc_fn_lib import GuideBase, GFMC, blocked_error, iat      # launched M1 walker code (unchanged copy)


def bonds(L, J2=0.5):
    """bond list in the order of rbmpp_lib.Lattice / st6_sector (site = x + L y; NN then NNN)."""
    site = lambda x, y: (x % L) + L * (y % L)
    nn_, nnn = [], []
    for y in range(L):
        for x in range(L):
            i = site(x, y)
            nn_ += [(i, site(x + 1, y)), (i, site(x, y + 1))]
            nnn += [(i, site(x + 1, y + 1)), (i, site(x + 1, y - 1))]
    bl = nn_ + nnn
    return (np.array([b[0] for b in bl], np.int32), np.array([b[1] for b in bl], np.int32),
            np.array([1.0] * len(nn_) + [J2] * len(nnn)))


class LogpsiGuide(GuideBase):
    def __init__(self, logpsi, L, J2=0.5, chunk=256, small=None, in_dtype=np.float64, logpsi_hi=None, thr=1e-3,
                 chunk_hi=64, drop_vsf=False):
        self.N = L * L; self.BI, self.BJ, self.JB = bonds(L, J2); self.NB = len(self.BI)
        self.lo = logpsi; self.hi = logpsi_hi; self.thr = thr; self.drop_vsf = drop_vsf
        self.chunk = chunk; self.small = small or max(chunk // 8, 8); self.chunk_hi = chunk_hi; self.in_dtype = in_dtype
        self.n_eval = 0; self.t_eval = 0.0; self.n_eval_hi = 0
        self.n_pairs = 0; self.n_rechk = 0; self.n_rechk_x = 0; self.n_flip_caught = 0; self.n_x = 0
        self.n_xphase_small = 0; self.max_im = 0.0; self.min_abs_cx = np.inf

    def _eval(self, fn, Z, chunk, small, dt):
        K = Z.shape[0]; out = np.empty(K, np.complex128); i = 0
        while i < K:
            rem = K - i; c = chunk if rem > small else small; n = min(c, rem)
            blk = Z[i:i + n]
            if n < c:
                blk = np.concatenate([blk, np.repeat(blk[:1], c - n, 0)])
            out[i:i + n] = np.asarray(fn(blk.astype(dt)))[:n]
            i += n
        return out

    def ev(self, Z):
        t0 = time.time(); l = self._eval(self.lo, Z, self.chunk, self.small, self.in_dtype)
        self.n_eval += Z.shape[0]; self.t_eval += time.time() - t0
        return l

    def ev_hi(self, Z):
        t0 = time.time(); l = self._eval(self.hi, Z, self.chunk_hi, max(self.chunk_hi // 8, 1), np.float64)
        self.n_eval_hi += Z.shape[0]; self.t_eval += time.time() - t0
        return l

    @staticmethod
    def ratio(LX, LY):
        return np.exp(LY - LX.real).real / np.exp(1j * LX.imag).real

    def rows(self, X):
        B = X.shape[0]; s = X.astype(np.float64)
        anti = X[:, self.BI] != X[:, self.BJ]
        wi, bi = np.nonzero(anti)
        Y = s[wi].copy(); r = np.arange(len(wi))
        Y[r, self.BI[bi]] = s[wi, self.BJ[bi]]; Y[r, self.BJ[bi]] = s[wi, self.BI[bi]]
        l = self.ev(np.concatenate([s, Y]))
        lx = l[:B]; LY = l[B:].copy(); LX = lx[wi].copy()
        self.n_pairs += len(wi); self.n_x += B
        if self.hi is not None:
            sel = np.exp((LY - LX).real) < self.thr          # |psi(y)/psi(x)| < thr in the low precision
            if sel.any():
                ux, inv = np.unique(wi[sel], return_inverse=True)
                lxh = self.ev_hi(s[ux]); lyh = self.ev_hi(Y[sel])
                r_lo = self.ratio(LX[sel], LY[sel])
                LX[sel] = lxh[inv]; LY[sel] = lyh
                self.n_flip_caught += int(np.sum(np.sign(r_lo) != np.sign(self.ratio(LX[sel], LY[sel]))))
                self.n_rechk += int(sel.sum()); self.n_rechk_x += len(ux)
        cxp = np.exp(1j * lx.imag).real
        self.min_abs_cx = min(self.min_abs_cx, float(np.min(np.abs(cxp))))
        self.n_xphase_small += int(np.sum(np.abs(cxp) < 1e-3))
        rr = self.ratio(LX, LY)
        if not (np.all(np.isfinite(rr)) and np.all(np.isfinite(lx.real))):
            raise FloatingPointError(f'non-finite guide ratio: {int(np.sum(~np.isfinite(rr)))} pairs, {int(np.sum(~np.isfinite(lx.real)))} walkers')
        return self.finish(X, anti, wi, bi, rr)

    def finish(self, X, anti, wi, bi, rr):
        if not self.drop_vsf:
            return GuideBase.finish(self, X, anti, wi, bi, rr)
        K = 0.5 * self.JB[bi]; keep = rr < 0                      # BROKEN control: V_sf (sign-violating diagonal) omitted
        rate = np.zeros(anti.shape); rate[wi[keep], bi[keep]] = -K[keep] * rr[keep]
        return rate, self.diag0(X)

    def stats(self):
        return dict(n_eval=self.n_eval, n_eval_hi=self.n_eval_hi, n_pairs=self.n_pairs, n_rechk=self.n_rechk,
                    rechk_frac=self.n_rechk / max(self.n_pairs, 1), n_rechk_x=self.n_rechk_x, n_flip_caught=self.n_flip_caught,
                    n_x=self.n_x, n_xphase_small=self.n_xphase_small, min_abs_cos_phase_x=self.min_abs_cx)


# ------------------------------------------------------------------------------------------ guide factories
def guide_rbmpp(a):
    """RBM+PP psi_P (launched rbmpp_lib copy, brute force over the projection images, f64), times w (from --consts or --w)."""
    import jax, jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    from rbmpp_lib import Lattice, make_fns
    z = np.load(a.params); use_rbm = bool(z['rbm']) if 'rbm' in z.files else True
    lat = Lattice(a.L, 0.5, a.proj)
    p = {k: jnp.asarray(z[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}
    w = complex(*json.load(open(a.consts))['w']) if a.consts else complex(*a.w)
    f = jax.jit(jax.vmap(make_fns(lat, jnp.float64, use_rbm)['logpsi'], in_axes=(None, 0)))
    lw = np.log(w)
    return (lambda S: np.asarray(f(p, jnp.asarray(S))) + lw), None, np.float64, dict(w=[w.real, w.imag], images=lat.nI, z=z, use_rbm=use_rbm)


def guide_table6(a):
    """6x6 exact table of a symmetric state (sector reps, per-config log amplitude lP and sign sP, e.g. RBM+PP table A):
    logpsi(x) = lP[rep(x)] + i pi [sP < 0].  numpy, CPU (canonicalisation with the st6_sector byte tables)."""
    import st6_sector as SS
    zt = np.load(a.table); reps = np.asarray(zt['reps']).astype(np.uint64)
    zs = np.load(a.syma); lP = zs['lP'].astype(np.float64); sP = zs['sP']
    T = SS.byte_tables(SS.space_group())                    # (288, 5, 256) uint64
    FULL = np.uint64((1 << 36) - 1); code = (np.uint64(1) << np.arange(36, dtype=np.uint64))
    lc = lP + 1j * np.pi * (sP < 0)
    miss = [0]

    def f(S):
        x = ((S > 0).astype(np.uint64) * code).sum(1, dtype=np.uint64)
        img = np.zeros((x.shape[0], T.shape[0]), np.uint64)
        for b in range(5):
            img |= T[:, b, :][:, ((x >> np.uint64(8 * b)) & np.uint64(255)).astype(np.int64)].T
        m = np.minimum(img.min(1), (img ^ FULL).min(1))
        idx = np.clip(np.searchsorted(reps, m), 0, len(reps) - 1)
        bad = reps[idx] != m
        if bad.any():
            miss[0] += int(bad.sum())
        out = lc[idx].copy(); out[bad] = -np.inf
        return out
    return f, None, np.float64, dict(reps=len(reps), miss=miss)


def guide_vit10(a):
    """our 10x10 ViT (vit10_jax), symm or plain; prec f64 or f32h (+ f64 recheck of pairs with |r| < --recheck)."""
    import jax, jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    sys.path.insert(0, HERE); import vit10_jax as V, validate_vit10 as VV
    BI, BJ, JB = bonds(a.L)                                   # ViT site = 10 x + y (transposed labels): same bond set and couplings
    assert {frozenset(p): j for p, j in zip(zip(BI.tolist(), BJ.tolist()), JB)} == {frozenset(p): j for p, j in zip(VV.BI.tolist(), VV.BJ)}
    path = V.SYMM if a.kind == 'symm' else V.PLAIN; mk = V.make_symm if a.kind == 'symm' else V.make_plain
    f64 = jax.jit(mk(V.load(path, jnp.float64), dt=jnp.float64))
    hi = lambda S: np.asarray(f64(jnp.asarray(S, jnp.float64))).astype(np.complex128)
    if a.prec == 'f64':
        return hi, None, np.float64, {}
    f32 = jax.jit(mk(V.load(path, jnp.float32), dt=jnp.float32, prec='highest'))
    lo = lambda S: np.asarray(f32(jnp.asarray(S, jnp.float32))).astype(np.complex128)
    return lo, (hi if a.recheck > 0 else None), np.float32, {}


# ------------------------------------------------------------------------------------------ walkers init
def metropolis_init(logpsi, L, M, seed, nsweep, dt=np.float64):
    """exchange Metropolis on |psi|^2 from random Sz = 0 configs (swap a uniform antiparallel bond, Hastings n_a(x)/n_a(y))."""
    BI, BJ, _ = bonds(L); N = L * L; rng = np.random.default_rng(seed + 7919)
    X = np.ones((M, N)); X[:, N // 2:] = -1
    for r in range(M): rng.shuffle(X[r])
    lp = np.asarray(logpsi(X.astype(dt))).astype(np.complex128); acc = 0
    for it in range(nsweep):
        anti = X[:, BI] != X[:, BJ]; na = anti.sum(1)
        b = np.argmax(rng.random(anti.shape) * anti, 1); r = np.arange(M)
        Y = X.copy(); Y[r, BI[b]], Y[r, BJ[b]] = X[r, BJ[b]], X[r, BI[b]]
        lq = np.asarray(logpsi(Y.astype(dt))).astype(np.complex128); nay = (Y[:, BI] != Y[:, BJ]).sum(1)
        ok = np.log(rng.random(M)) < 2 * (lq.real - lp.real) + np.log(na / nay)
        X[ok] = Y[ok]; lp = np.where(ok, lq, lp); acc += ok.mean()
    return X.astype(np.int8), acc / max(nsweep, 1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--guide', required=True, choices=['rbmpp', 'table6', 'vit10'])
    ap.add_argument('--L', type=int, required=True); ap.add_argument('--M', type=int, default=256); ap.add_argument('--dtau', type=float, default=0.1)
    ap.add_argument('--blocks', type=int, default=200); ap.add_argument('--burn', type=int, default=50); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--chunk', type=int, default=256); ap.add_argument('--chunk_hi', type=int, default=64)
    ap.add_argument('--ckpt_every', type=int, default=10); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--maxsec', type=float, default=1e18); ap.add_argument('--E0', type=float, default=None, help='reference energy per site printed as d_ref')
    ap.add_argument('--init', default='metropolis', choices=['metropolis', 'rbmpp', 'shipped']); ap.add_argument('--nsweep', type=int, default=2000)
    ap.add_argument('--broken', default='none', choices=['none', 'drop_vsf'])
    # rbmpp
    ap.add_argument('--params'); ap.add_argument('--consts'); ap.add_argument('--proj', default='translations'); ap.add_argument('--w', type=float, nargs=2, default=[1.0, 0.0])
    # table6
    ap.add_argument('--table'); ap.add_argument('--syma')
    # vit10
    ap.add_argument('--kind', default='symm'); ap.add_argument('--prec', default='f32h', choices=['f64', 'f32h']); ap.add_argument('--recheck', type=float, default=1e-3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); prog = os.path.join(a.out, 'progress.txt')

    def P(msg):
        line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}'
        print(line, flush=True); open(prog, 'a').write(line + '\n')

    fn, fn_hi, idt, info = {'rbmpp': guide_rbmpp, 'table6': guide_table6, 'vit10': guide_vit10}[a.guide](a)
    N = a.L * a.L
    g = LogpsiGuide(fn, a.L, chunk=a.chunk, in_dtype=idt, logpsi_hi=fn_hi, thr=a.recheck, chunk_hi=a.chunk_hi, drop_vsf=(a.broken == 'drop_vsf'))
    P(f'guide {a.guide} L {a.L} N {N} bonds {g.NB} broken {a.broken} prec {a.prec if a.guide == "vit10" else "f64"} recheck {fn_hi is not None} thr {a.recheck} '
      f'M {a.M} dtau {a.dtau} seed {a.seed} init {a.init} chunk {a.chunk} info {dict((k, v) for k, v in info.items() if k in ("w", "images", "reps"))}')
    ck = os.path.join(a.out, 'ckpt.npz'); rec = []; b0 = 0
    if a.resume and os.path.exists(ck):
        zz = np.load(ck, allow_pickle=True); X0 = zz['X']; rec = list(zz['rec'].tolist()); b0 = int(zz['nb'])
        gf = GFMC(g, X0, a.dtau, a.seed); gf.rng.bit_generator.state = pickle.loads(zz['rng'].tobytes())
        P(f'RESUMED at block {b0} from {ck}')
    else:
        t0 = time.time()
        if a.init == 'rbmpp':
            from gfmc_run import init_walkers                      # launched M1 init (f32 Metropolis on |psi_P|^2)
            from rbmpp_lib import Lattice
            X0, acc = init_walkers(Lattice(a.L, 0.5, a.proj), info['z'], a.M, a.seed, info['use_rbm'])
        elif a.init == 'shipped':
            import validate_vit10 as VV
            S = VV.spins(); X0 = S[np.random.default_rng(a.seed).choice(len(S), a.M, replace=False)].astype(np.int8); acc = float('nan')
        else:
            X0, acc = metropolis_init(lambda S: g._eval(fn, S, a.chunk, g.small, idt), a.L, a.M, a.seed, a.nsweep, idt)
        P(f'init walkers {a.init} acc {acc:.3f} {time.time() - t0:.0f}s Sz {np.abs(X0.sum(1)).max()}')
        gf = GFMC(g, X0, a.dtau, a.seed)
        P(f'initial rows {a.M} walkers {g.n_eval} amplitude evaluations {g.t_eval:.0f}s')
    t_start = time.time(); tot = a.burn + a.blocks

    def save(nb):
        tmp = ck + '.tmp.npz'
        np.savez(tmp, X=gf.X, rec=np.array(rec, dtype=object), nb=nb, rng=np.frombuffer(pickle.dumps(gf.rng.bit_generator.state), np.uint8))
        os.replace(tmp, ck)
    stopped = False; status = 'OK'
    for b in range(b0, tot):
        tb = time.time(); n0 = g.n_eval; st0 = g.stats()
        try:
            r = gf.block()
        except FloatingPointError as e:
            status = f'NAN {e}'; P(f'ABORT block {b + 1}: {e}'); break
        r['sec'] = time.time() - tb; r['evals'] = g.n_eval - n0; r['block'] = b; st = g.stats()
        r['pairs'] = st['n_pairs'] - st0['n_pairs']; r['rechk'] = st['n_rechk'] - st0['n_rechk']; r['flip_caught'] = st['n_flip_caught'] - st0['n_flip_caught']
        r['M'] = gf.M; r['logw_ok'] = bool(np.isfinite(r['emix']) and np.isfinite(r['egrow']))
        rec.append(r)
        dref = f' d_ref {r["emix"] / N - a.E0:+.4e}' if a.E0 is not None else ''
        P(f'block {b + 1}/{tot} E_mix/N {r["emix"] / N:.8f} E_grow/N {r["egrow"] / N:.8f}{dref} ess {r["ess"]:.1f} elvar {r["elvar"]:.4f} '
          f'hops {r["hops"]} iters {r["iters"]} evals {r["evals"]} rechk {r["rechk"]}/{r["pairs"]} flips_caught {r["flip_caught"]} sec {r["sec"]:.1f}')
        if not r['logw_ok']:
            status = 'NAN energy'; P(f'ABORT block {b + 1}: non-finite energy'); break
        if (b + 1) % a.ckpt_every == 0: save(b + 1)
        if time.time() - t_start > a.maxsec:
            save(b + 1); stopped = True; P(f'MAXSEC reached at block {b + 1}'); break
    nb = len(rec); save(nb)
    em = np.array([r['emix'] for r in rec[a.burn:]]); eg = np.array([r['egrow'] for r in rec[a.burn:]])
    np.save(os.path.join(a.out, 'E_mix.npy'), em / N)
    keys = ('emix', 'egrow', 'ess', 'elvar', 'hops', 'sec', 'evals', 'pairs', 'rechk', 'flip_caught')
    np.savez(os.path.join(a.out, 'blocks.npz'), burn=a.burn, dtau=a.dtau, M=a.M, N=N, **{k: np.array([r[k] for r in rec]) for k in keys})
    out = dict(args=vars(a), nblocks=nb, completed=(not stopped and status == 'OK' and nb == tot), status=status, N=N, guide_stats=g.stats())
    if a.guide == 'table6': out['table_miss'] = info['miss'][0]
    if len(em) >= 32:
        for name, x in (('mix', em), ('grow', eg)):
            m, errs, tau = blocked_error(x)
            out[name] = dict(E_site=m / N, err_site=max(errs[-4:]) / N, err_level0_site=errs[0] / N, tau_int_blocks=iat(x), tau_blocking=tau)
            if a.E0 is not None: out[name]['dE_site'] = m / N - a.E0
    if rec:
        out['sec_per_block_median'] = float(np.median([r['sec'] for r in rec])); out['evals_per_block_mean'] = float(np.mean([r['evals'] for r in rec]))
        out['hops_per_block_mean'] = float(np.mean([r['hops'] for r in rec])); out['sec_per_walker_tau'] = out['sec_per_block_median'] / (a.M * a.dtau)
        out['sec_per_hop'] = float(np.sum([r['sec'] for r in rec]) / max(np.sum([r['hops'] for r in rec]), 1))
    json.dump(out, open(os.path.join(a.out, 'result.json'), 'w'), indent=1, default=float)
    P(f'GFMC_GUIDE_DONE status {status} blocks {nb}/{tot} completed {out["completed"]} ' +
      (f'E_mix/N {out["mix"]["E_site"]:.8f} +- {out["mix"]["err_site"]:.1e}' if 'mix' in out else ''))
