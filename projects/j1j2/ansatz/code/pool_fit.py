"""pool_fit.py: the label-free round fit driven by a SAMPLED POOL instead of the exact 6x6 tables (PLAN_pfaffian.md "B design", build item (iii);
"V (narrowed)").  New file; cx_r1fit.py is NOT modified (its pp2 model, loss and damped GN are COPIED here, generalised to per-pair bond weights).

  python pool_fit.py build OUT SPEC.json     6x6: draw the pool exactly from the tables, label it, weight it  -> OUT/pool.npz, OUT/pool_summary.json
  python pool_fit.py fit   OUT SPEC.json     damped GN of arm 'pp2' (or 'rbmpp') on batches drawn from a pool  -> OUT/theta_step{N}.npy, theta.npy, theta_bar.npy, pool_fit.json

POOL (one entry per x-draw i; K partners each)
  x_i ~ the tempered ladder |B|^(2 beta) per CONFIGURATION, beta in BETAS, S_beta = S / len(BETAS) draws each (i.i.d. at 6x6: orbit r with prob n_r |B_r|^(2 beta) / Z).
  B is the round's guide B_k (its table at 6x6), NOT the model being fitted.
  partners: K = 2 blocked y of x (valid bond, s(x) s(y) > 0 under the given sign), drawn i.i.d. with q(y|x) = w^B_xy / W^B_x,
    w^B_xy = |H_xy| |B(y)/B(x)| (|H_xy| = J_b / 2), W^B_x = sum over blocked y.  No blocked bond: y = x and rho = 0.
  labels: ln T(x) through labels_fn(configs) -> ln T (6x6: TableLabels = the target table read by orbit index after canonicalisation, normalised as cx_r1fit.norm_la).
  importance ratio of the bond weight the table fit uses (w_xy = |H_xy| |T(y)/T(x)|):  rho_ik = w_xy / q(y|x) = W^B_x exp[(lnT_y - lnB_y) - (lnT_x - lnB_x)],
    so (1/K) sum_k rho_ik (e_x - e_yk)^2 is unbiased for sum_y w_xy (e_x - e_y)^2 and (1/K) sum_k rho_ik for W_x.
  decades: d_i = floor((2 lnB_x - med) / ln 10), med = median of 2 lnB over the beta = 1 draws (the norm of B is unknown at 8x8).  Only the decades in
    spec 'decs_rel' are fitted (default +2..-6 = 9 decades; at 6x6 B_4 these are the absolute |B_4|^2 decades -5.07..-14.07, the table fit used -5..-13 of |T|^2).
  band-uniform weights: inside a decade the table fit draws configurations uniformly (orbits with weight n_r).  Per (beta, decade) cell the weight is
    exp(-2 beta lnB_x) normalised in the cell (self-normalised IS; Z_beta cancels); the beta cells of a decade are mixed in proportion to their ESS.
    => w_in (sums to 1 over the training entries of each decade).  Wbar_d = sum_i w_in_i (1/K) sum_k rho_ik (training entries of d).
  held-out: entries whose x orbit is in cx_lib.Lab.HO (same hash; 6x6) - else a hash of the entry index; never drawn for training.

LOSS (= cx_r1fit.per_sample with per-pair weights)   L_x = e_x^2 + sum_k wb_xk (e_x - e_yk)^2,  e = ln A - ln T - c,  wb_xk = lam_b rho_xk / (K Wbar_d(x)).
  A batch draws ceil(M / ND) entries per decade with probability w_in (with replacement): the same population loss as the table fit's stratified batch.
FIT (= cx_r1fit's pp2 arm): damped Gauss-Newton (minSR in sample space when rows <= params, LM in parameter space otherwise), trust radius tau, tau anneal,
  fresh acceptance batch with the stratified paired test, fixed held-out batch, exactly max_steps steps, theta saved at save_theta_at; theta_bar = mean of the saved
  iterates and the final theta (round-5 tail average).  Test-only knob: break_negate_step.
8x8 hooks (not used at 6x6): draw_fn(beta, S, rng) -> configs, lnB;  neighbours_fn(configs) -> (y (S, NB) configs, hw (NB,), valid, blocked, dlnB);
  labels_fn(configs) -> ln T  (pp2_lemma.fJ_fixed / logB plug in here).  make_model below is the 6x6 pp2 (Lattice(6)).
8x8 (added 2026-11-04, build piece (1b); the 6x6 code is untouched): the '8x8 model' block holds make_model8 (arm 'pp8': B = sum_g phi(z_g)[a_1 + lam a_2], complex f_1, f_2 (64 x 64),
  complex RBM M = 16, lam complex, translations x flip, npar 18466; = pp2_lemma rbm 'inner'), warm_theta8 (production rb8_prod + P_2 = P_1 + noise), theta8_to_pp2 / pp2_to_theta8,
  Labels8 (the labels_fn: ln T = Re ln B + c f_J via pp2_lemma.fJ_fixed mode 'fix'), Neighbours8 (neighbours_fn), canon64 / pack64 / unpack64 (configs are uint64 words, bit i = [sigma_i = +1]),
  build8 (pool builder from a draw_fn) and a dev sampler.  fit() with spec 'arm': 'pp8' runs the same damped GN on this model.  Tests: pool_fit8_test.py.
"""
import os, sys, json, time, math, hashlib
from functools import partial
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAB6 = '/scratch/a/A.Otaifi/ansatz_runs/c2_t0b_6x6/tables'
CSRD6 = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr'
TABLE6 = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz'
HW6 = np.concatenate([np.full(72, 0.5), np.full(72, 0.25)])        # |H_xy| = J_b / 2 per bond column of nb_J1 | nb_J2
LN10 = np.log(10.0)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def jdump(path, obj):
    json.dump(obj, open(path, 'w'), indent=1, default=float)


def norm_la_np(la, n):
    """cx_r1fit.norm_la in numpy: la - max - 0.5 ln sum_r n_r exp(2 (la - max))."""
    m = np.max(la); return la - m - 0.5 * np.log(np.sum(n * np.exp(2.0 * (la - m))))


def ho_hash(idx):
    """cx_lib.Lab.HO: ((i * 2654435761 mod 2^32) >> 7) % 5 == 0."""
    return (((np.asarray(idx, np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)) >> np.uint64(7)) % np.uint64(5) == 0


# ============================================================================================ 6x6 exact helpers (tables; no Lab)
class Exact6:
    """reps, n, orbit neighbour tables, canonicalisation; the three pool hooks for 6x6 from tables."""

    def __init__(self, la_B, sign, tab=TAB6, csr=CSRD6, table=TABLE6):
        import jax, jax.numpy as jnp
        if HERE not in sys.path: sys.path.insert(0, HERE)
        sys.path.append('/project/theorie/a/A.Otaifi/Chatty/runs/compress6/code')
        import st6_sector as SS
        self.SS, self.jnp = SS, jnp
        self.n = np.load(f'{csr}/norb.npy').astype(np.float64); self.D = self.n.size
        self.reps = np.load(table)['reps']; assert self.reps.shape == (self.D,) and np.all(np.diff(self.reps) > 0)
        self.t1 = np.load(f'{tab}/nb_J1.npy', mmap_mode='r'); self.t2 = np.load(f'{tab}/nb_J2.npy', mmap_mode='r')
        self.lB = np.asarray(la_B, np.float64); assert self.lB.shape == (self.D,) and np.isfinite(self.lB).all()
        self.s = np.asarray(sign).astype(np.float64); assert self.s.shape == (self.D,) and set(np.unique(self.s)) <= {-1.0, 1.0}
        self.T = jnp.asarray(SS.byte_tables(SS.space_group())); self.reps_j = jnp.asarray(self.reps)
        self._canon = jax.jit(lambda x: SS.canon(self.T, self.reps_j, x))
        self._cdf = {}

    def canon(self, cfg, ch=32768):
        cfg = np.asarray(cfg, np.uint64); out = np.empty(cfg.size, np.int64)
        for i in range(0, cfg.size, ch):
            c = cfg[i:i + ch]; m = c.size
            if m < ch: c = np.concatenate([c, np.repeat(c[:1], ch - m)])
            out[i:i + m] = np.asarray(self._canon(self.jnp.asarray(c)))[:m]
        return out

    def draw_fn(self, beta, S, rng):
        """i.i.d. orbits with prob n_r |B_r|^(2 beta); returns (configs = reps, lnB)."""
        if beta not in self._cdf:
            lw = np.log(self.n) + 2.0 * beta * (self.lB - self.lB.max()); c = np.cumsum(np.exp(lw - lw.max())); self._cdf[beta] = c / c[-1]
        r = np.minimum(np.searchsorted(self._cdf[beta], rng.random(S)), self.D - 1)
        return self.reps[r], self.lB[r], r

    def nb_rows(self, orb):
        o = np.asarray(orb); u, inv = np.unique(o, return_inverse=True)
        t = np.concatenate([np.asarray(self.t1[u]), np.asarray(self.t2[u])], 1)[inv]
        return t

    def neighbours_fn(self, cfg, orb=None):
        """(y configs (S, 144) [orbit reps], hw (144,), valid, blocked, dlnB = lnB(y) - lnB(x), y orbit)."""
        orb = self.canon(cfg) if orb is None else np.asarray(orb)
        t = self.nb_rows(orb); valid = t >= 0; tt = np.where(valid, t, orb[:, None])
        blocked = valid & (self.s[orb][:, None] * self.s[tt] > 0)
        return self.reps[tt], HW6, valid, blocked, self.lB[tt] - self.lB[orb][:, None], tt


class TableLabels:
    """labels_fn(configs) -> ln T, read from a full 6x6 target table by orbit index (normalised as cx_r1fit.norm_la)."""

    def __init__(self, ex, path, normalise=True):
        la = np.load(path).astype(np.float64); assert la.shape == (ex.D,) and np.isfinite(la).all(), path
        self.ex = ex; self.la = norm_la_np(la, ex.n) if normalise else la; self.path = path; self.sha256 = sha(path)

    def __call__(self, cfg, orb=None):
        return self.la[self.ex.canon(cfg) if orb is None else np.asarray(orb)]


# ============================================================================================ pool builder (generic in the three hooks)
def build_pool(draw_fn, neighbours_fn, labels_fn, S, betas, K, seed, decs_rel, ho_fn=None, log=print):
    rng = np.random.default_rng(seed); nb_ = len(betas)
    Sb = [S // nb_ + (1 if q < S % nb_ else 0) for q in range(nb_)]
    xs, lBs, orbs, chain = [], [], [], []
    for q, (b, s_) in enumerate(zip(betas, Sb)):
        c_, l_, o_ = draw_fn(b, s_, rng); xs.append(c_); lBs.append(l_); orbs.append(o_); chain.append(np.full(s_, q, np.int8))
    x = np.concatenate(xs); lBx = np.concatenate(lBs); xorb = np.concatenate(orbs); chain = np.concatenate(chain); S = x.size
    log(f'[pool] drew {S} x ({Sb} per beta {betas})')
    yK = np.empty((S, K), np.uint64); yorb = np.empty((S, K), np.int64); dlB = np.zeros((S, K)); WB = np.zeros(S); nblk = np.zeros(S, np.int32)
    for i0 in range(0, S, 65536):
        i1 = min(S, i0 + 65536)
        ycfg, hw, valid, blocked, dlnB, yo = neighbours_fn(x[i0:i1], xorb[i0:i1])
        lw = np.where(blocked, np.log(hw)[None, :] + np.clip(dlnB, -700, 700), -np.inf)
        mx = np.max(lw, 1, keepdims=True); has = np.isfinite(mx[:, 0]); mx = np.where(has[:, None], mx, 0.0)
        w = np.exp(lw - mx); W = w.sum(1); WB[i0:i1] = np.where(has, W * np.exp(mx[:, 0]), 0.0); nblk[i0:i1] = blocked.sum(1)
        cdf = np.cumsum(w, 1) / np.where(has, W, 1.0)[:, None]
        sel = np.empty((i1 - i0, K), np.int64)
        for k in range(K): sel[:, k] = np.minimum((cdf < rng.random(i1 - i0)[:, None]).sum(1), cdf.shape[1] - 1)
        ar = np.arange(i1 - i0)[:, None]
        yK[i0:i1] = np.where(has[:, None], ycfg[ar, sel], x[i0:i1, None]); yorb[i0:i1] = np.where(has[:, None], yo[ar, sel], xorb[i0:i1, None])
        dlB[i0:i1] = np.where(has[:, None], dlnB[ar, sel], 0.0)
    log(f'[pool] partners drawn; entries without a blocked bond: {int((nblk == 0).sum())}')
    lTx = labels_fn(x, xorb); lTy = labels_fn(yK.reshape(-1), yorb.reshape(-1)).reshape(S, K)
    lBy = lBx[:, None] + dlB
    rho = np.where(nblk[:, None] > 0, WB[:, None] * np.exp(np.clip((lTy - lBy) - (lTx - lBx)[:, None], -700, 700)), 0.0)
    log('[pool] labels done')
    return dict(x=x, x_orb=xorb, chain=chain, betas=np.asarray(betas, np.float64), lnB_x=lBx, y=yK, y_orb=yorb, lnB_y=lBy, WB=WB, nblk=nblk,
                lnT_x=lTx, lnT_y=lTy, rho=rho, K=np.int64(K), seed=np.int64(seed), **pool_weights(lBx, chain, betas, decs_rel, rho, xorb, ho_fn))


def pool_weights(lBx, chain, betas, decs_rel, rho, xorb, ho_fn=None):
    """decade labels relative to the beta = 1 median, band-uniform within-decade weights w_in (training entries), held-out mask, Wbar_d."""
    q1 = int(np.argmin(np.abs(np.asarray(betas) - 1.0))); assert abs(betas[q1] - 1.0) < 1e-12, 'beta = 1 chain required (median)'
    med = float(np.median(2.0 * lBx[chain == q1]))
    dec = np.floor((2.0 * lBx - med) / LN10).astype(np.int64)
    ho = ho_fn(xorb) if ho_fn is not None else ho_hash(np.arange(lBx.size))
    w_in = np.zeros(lBx.size); ess = {}; Wbar = np.zeros(len(decs_rel)); W_est = rho.mean(1)
    for g, d in enumerate(decs_rel):
        md = (dec == d) & ~ho; parts = []; e_ = []
        for q, b in enumerate(betas):
            m = md & (chain == q)
            if not m.any(): e_.append(0.0); parts.append(None); continue
            lw = -2.0 * b * lBx[m]; w = np.exp(lw - lw.max()); w /= w.sum(); parts.append((m, w)); e_.append(1.0 / np.sum(w * w))
        E = np.asarray(e_); assert E.sum() > 0, f'decade {d}: no training entry'
        for q, p in enumerate(parts):
            if p is not None: w_in[p[0]] += (E[q] / E.sum()) * p[1]
        ess[str(d)] = dict(per_beta=E.tolist(), total_mixed=float(1.0 / np.sum(w_in[md] ** 2)))
        Wbar[g] = float(np.sum(w_in[md] * W_est[md]))
    return dict(dec=dec, med2lnB=np.float64(med), ho=ho, w_in=w_in, decs_rel=np.asarray(decs_rel, np.int64), Wbar=Wbar, ess_json=np.asarray(json.dumps(ess)))


def pool_summary(P):
    dec, ch, ho = P['dec'], P['chain'], P['ho']; betas = P['betas'].tolist(); out = dict(S=int(dec.size), K=int(P['K']), betas=betas, med2lnB=float(P['med2lnB']))
    out['counts'] = {str(d): dict(all=int((dec == d).sum()), train=int(((dec == d) & ~ho).sum()), heldout=int(((dec == d) & ho).sum()),
                                  per_beta=[int(((dec == d) & (ch == q)).sum()) for q in range(len(betas))]) for d in range(int(dec.max()), int(dec.min()) - 1, -1)}
    out['decs_rel'] = P['decs_rel'].tolist(); out['Wbar'] = P['Wbar'].tolist(); out['ess'] = json.loads(str(P['ess_json']))
    out['outside_decs'] = int((~np.isin(dec, P['decs_rel'])).sum()); out['no_blocked'] = int((P['nblk'] == 0).sum())
    r = P['rho'][P['nblk'] > 0] / P['WB'][P['nblk'] > 0][:, None]
    out['rho_over_WB_quantiles'] = {q: float(np.quantile(r, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99, 0.999)}
    out['unique_x_orbits'] = int(np.unique(P['x_orb']).size) if 'x_orb' in P else None
    return out


# ============================================================================================ model + loss + GN (copied from cx_r1fit.py arms 'pp2' / 'rbmpp', 6x6)
def make_model(arm, NA=4):
    import jax, jax.numpy as jnp
    from cx_lib import bits, N
    from rbmpp_lib import Lattice, pp2_combine
    lat = Lattice(6, 0.5, 'full'); IMG = lat.IMG; FLP = lat.FL.astype(jnp.float64); NH = N // 2
    NP1 = 2 * N * N + NA * (N + 1); PP2 = arm == 'pp2'; assert arm in ('pp2', 'rbmpp'), arm

    def log_one(f, zz):
        order = jnp.argsort(-zz, stable=True); Uo = order[:NH]; Do = order[NH:]
        sg, ld = jnp.linalg.slogdet(f[Uo][:, Do])
        up = (zz > 0).astype(jnp.int32); dn = 1 - up; inv = jnp.sum(up * (jnp.cumsum(dn) - dn))
        return ld, (1.0 - 2.0 * (inv % 2).astype(jnp.float64)) * sg

    def lncosh(x):
        ax = jnp.abs(x); return ax + jnp.log1p(jnp.exp(-2.0 * ax)) - jnp.log(2.0)

    def A_one(th, sigma):
        f = (th[:N * N] + 1j * th[N * N:2 * N * N]).reshape(N, N); wb = th[2 * N * N:NP1].reshape(NA, N + 1)
        Z = FLP[:, None] * sigma.astype(jnp.float64)[IMG]
        ld, sgn = jax.vmap(lambda zz: log_one(f, zz))(Z)
        Rx = jnp.sum(lncosh(Z @ wb[:, :N].T + wb[:, N][None, :]))
        if PP2:
            f2 = (th[NP1:NP1 + N * N] + 1j * th[NP1 + N * N:NP1 + 2 * N * N]).reshape(N, N); lam = th[NP1 + 2 * N * N] + 1j * th[NP1 + 2 * N * N + 1]
            ld2, sgn2 = jax.vmap(lambda zz: log_one(f2, zz))(Z)
            m, S_ = pp2_combine(ld, sgn, ld2, sgn2, lam); return m + Rx, S_
        m = jax.lax.stop_gradient(jnp.max(ld)); return m + Rx, jnp.sum(sgn * jnp.exp(ld - m))

    def logA_one(th, sigma):
        m, S_ = A_one(th, sigma); return m + jnp.log(jnp.maximum(jnp.abs(S_), 1e-300))
    fwd_j = jax.jit(lambda th, Sx: jax.vmap(lambda s_: logA_one(th, s_))(bits(Sx).astype(jnp.float64)))
    jac_j = jax.jit(jax.vmap(jax.grad(logA_one), in_axes=(None, 0)))
    npar = NP1 + (2 * N * N + 2 if PP2 else 0)
    return dict(fwd_j=fwd_j, jac_j=jac_j, bits=bits, N=N, NP1=NP1, npar=npar, logA_one=logA_one)


class Net:
    FCH, JCH = 512, 64

    def __init__(self, model):
        import jax, jax.numpy as jnp
        self.jax, self.jnp, self.m = jax, jnp, model
        self._put = jax.jit(lambda Gb, blk, i0: jax.lax.dynamic_update_slice(Gb, blk, (i0, 0)), donate_argnums=(0,))

    def _pad(self, Sx, ch):
        m = Sx.shape[0]; jnp = self.jnp
        return (jnp.concatenate([Sx, jnp.repeat(Sx[:1], ch - m)]) if m < ch else Sx), m

    def fwd(self, th, Sx):
        jnp = self.jnp; Sx = jnp.asarray(Sx); out = []
        for i in range(0, Sx.shape[0], self.FCH):
            s_, m = self._pad(Sx[i:i + self.FCH], self.FCH); out.append(self.m['fwd_j'](th, s_)[:m])
        return jnp.concatenate(out)

    def jac(self, th, Sx):
        jnp = self.jnp; Sx = jnp.asarray(Sx); R_ = Sx.shape[0]; Gb = jnp.zeros((R_, th.size), jnp.float64)
        for i in range(0, R_, self.JCH):
            s_, m = self._pad(Sx[i:i + self.JCH], self.JCH)
            Gb = self._put(Gb, self.m['jac_j'](th, self.m['bits'](s_).astype(jnp.float64))[:m], i)
        return Gb


def _jfuns():
    import jax, jax.numpy as jnp
    f64 = jnp.float64

    @jax.jit
    def per_sample(S, lx, ly, wb, c):
        """cx_r1fit.per_sample with per-pair weights wb (M, K)."""
        M = lx.shape[0]; ex = S[:M] - lx - c; ey = S[M:].reshape(M, -1) - ly - c; db = ex[:, None] - ey
        return ex * ex + jnp.sum(wb * db * db, 1), ex, db

    @jax.jit
    def grad_from_G(G, S, lx, ly, wb, c, a):
        """d/dtheta sum_i a_i L_i = 2 sum_i a_i [e_x G_x + sum_k wb_ik (e_x - e_yk)(G_x - G_yk)]  (and d/dc)."""
        M = lx.shape[0]; P = G.shape[1]; ex = S[:M] - lx - c; ey = S[M:].reshape(M, -1) - ly - c; db = ex[:, None] - ey
        Gx = G[:M]; Gy = G[M:].reshape(M, -1, P)
        gx = a * ex + jnp.sum(a[:, None] * wb * db, 1)
        gt = 2.0 * (Gx.T @ gx - jnp.einsum('mk,mkp->p', a[:, None] * wb * db, Gy))
        return gt, -2.0 * jnp.sum(a * ex)

    @jax.jit
    def solve_dir(G, S, lx, ly, wb, c, lam_rel, tau):
        """cx_r1fit.solve_dir with per-pair bond weights wb (M, K): bond row (x, k) = sqrt(wb_xk / M) (e_x - e_yk)."""
        M = lx.shape[0]; P = G.shape[1]; dt = G.dtype
        ex = S[:M] - lx - c; ey = S[M:].reshape(M, -1) - ly - c; Kb = ey.shape[1]
        sv = 1.0 / jnp.sqrt(M); sb = jnp.sqrt(wb / M); sbf = sb.reshape(-1)
        r = jnp.concatenate([sv * ex, (sb * (ex[:, None] - ey)).reshape(-1)])
        cc = jnp.concatenate([jnp.full(M, -sv, f64), jnp.zeros(M * Kb, f64)])
        R = M * (1 + Kb); Gx = G[:M]
        if R <= P + 1:
            rows = jnp.arange(M * Kb); xs = rows // Kb
            i1 = jnp.concatenate([jnp.arange(M), xs]); c1 = jnp.concatenate([jnp.full(M, sv, f64), sbf])
            i2 = jnp.concatenate([jnp.zeros(M, jnp.int32), M + rows]); c2 = jnp.concatenate([jnp.zeros(M, f64), -sbf])
            GG = (G @ G.T).astype(f64)
            H = c1[:, None] * GG[i1] + c2[:, None] * GG[i2]; del GG
            T = H[:, i1] * c1[None, :] + H[:, i2] * c2[None, :] + jnp.outer(cc, cc); del H
            lam = lam_rel * jnp.trace(T) / R
            T = T.at[jnp.arange(R), jnp.arange(R)].add(lam)
            a = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(T, lower=True), r); del T
            w = jnp.zeros(R, f64).at[i1].add(c1 * a).at[i2].add(c2 * a)
            dP = -(G.T @ w.astype(dt)).astype(f64); dc = -(cc @ a)
        else:
            Jv = Gx * sv.astype(dt)
            Jb = ((Gx[:, None, :] - G[M:].reshape(M, Kb, P)) * sb.astype(dt)[:, :, None]).reshape(M * Kb, P)
            Jf = jnp.concatenate([jnp.concatenate([Jv, Jb]).astype(f64), cc[:, None]], 1); del Jv, Jb
            Sm = Jf.T @ Jf; g = Jf.T @ r
            lam = lam_rel * jnp.trace(Sm) / (P + 1)
            d = -jnp.linalg.solve(Sm + lam * jnp.eye(P + 1, dtype=f64), g)
            dP = d[:P]; dc = d[P]
        pc = (Gx.astype(f64) @ dP); pc = pc - jnp.mean(pc)
        prms = jnp.sqrt(jnp.mean(pc * pc))
        sc = jnp.minimum(1.0, tau / jnp.maximum(prms, 1e-30))
        return dP * sc, dc * sc, prms, sc, jnp.sum(r * r)
    return per_sample, grad_from_G, solve_dir


def strat_mean_se(v, g, G):
    v = np.asarray(v); mu = []; var = []
    for k in range(G):
        q = v[g == k]; mu.append(q.mean()); var.append(q.var(ddof=1) / q.size)
    return float(np.mean(mu)), float(np.sqrt(np.sum(var)) / G)


class PoolSampler:
    """ceil(M / ND) entries per fitted decade, drawn with prob w_in (training) or uniformly-in-configuration via w_in recomputed on the held-out entries."""

    def __init__(self, P, heldout, seed, lam_b):
        self.P = P; self.rng = np.random.default_rng(seed); self.K = int(P['K']); self.lam_b = lam_b; self.groups = []
        decs = P['decs_rel']
        if heldout:                                   # held-out entries: their own band-uniform weights (same rule, the ho mask inverted)
            hw_ = pool_weights(P['lnB_x'], P['chain'], P['betas'].tolist(), decs.tolist(), P['rho'], None, ho_fn=lambda _: ~P['ho'])
            w = hw_['w_in']; elig = P['ho']
        else:
            w = P['w_in']; elig = ~P['ho']
        for d in decs:
            idx = np.nonzero((P['dec'] == d) & elig & (w > 0))[0]; cdf = np.cumsum(w[idx]); self.groups.append((idx, cdf / cdf[-1]))

    def draw(self, M):
        G = len(self.groups); m = -(-M // G)
        idx = np.concatenate([g[0][np.minimum(np.searchsorted(g[1], self.rng.random(m)), g[0].size - 1)] for g in self.groups])
        return idx, np.repeat(np.arange(G), m)

    def batch(self, M, idx=None, g=None):
        import jax.numpy as jnp
        P = self.P
        if idx is None: idx, g = self.draw(M)
        wb = self.lam_b * P['rho'][idx] / (self.K * P['Wbar'][g][:, None])
        return dict(idx=idx, g=g, x=jnp.asarray(P['x'][idx]), y=jnp.asarray(P['y'][idx]), lx=jnp.asarray(P['lnT_x'][idx]), ly=jnp.asarray(P['lnT_y'][idx]),
                    wb=jnp.asarray(wb), ng=len(self.groups))


def load_pool(path):
    z = np.load(path, allow_pickle=False); return {k: z[k] for k in z.files}


def warm_theta(SP, mdl):
    if mdl.get('arm') == 'pp8': return warm_theta8(SP, mdl)                          # 8x8 (see the 8x8 block); the 6x6 path below is unchanged
    _w = np.load(SP['warm_theta']); NP1 = mdl['NP1']; N = mdl['N']
    if mdl['npar'] != NP1 and _w.size != mdl['npar']:            # pp2: 2740 P_1 (+RBM) + f_2 of warm_theta2 + lam
        assert _w.size == NP1; _w2 = np.load(SP['warm_theta2']); assert _w2.size == NP1
        _w = np.concatenate([_w, _w2[:2 * N * N], [float(SP.get('lambda_init_re', 0.1)), float(SP.get('lambda_init_im', 0.0))]])
    assert _w.size == mdl['npar'] and _w.dtype == np.float64, (_w.size, mdl['npar'])
    return _w


# ============================================================================================ 8x8 model (two-PP + complex 'inner' RBM), labels and pool hooks
# Added for build piece (1b) (PLAN_pfaffian.md "8x8 BUILD, route 2").  Nothing above this line is touched; the 6x6 arms 'pp2' / 'rbmpp' run exactly as before.
#
#   B(sigma) = sum_g phi(z_g) [ a_1(g) + lam a_2(g) ],   a_j(g) = par(z_g) det f_j[Up(z_g), Dn(z_g)],   phi(z) = prod_k cosh(b_k + z . W[:, k])
#   images z_g = FL_g sigma[IMG_g]: the 64 translations x spin flip (128 images; proj 'translations', the production structure).  f_1, f_2: 64 x 64 COMPLEX,
#   b, W: complex RBM (M = 16 hidden units), lam: complex.  The same functional form as pp2_lemma rbm = 'inner' (P_2 shares the RBM with P_1), written here
#   independently of pp2_lemma (own lattice, own slogdet per image, own complex log-sum-exp) so that pp2_lemma.logB is a real cross-check.
#   The fitted amplitude is A = |B| (the guide is s_G |B| e^{c f_J}; its sign is not fitted), logA = Re log B.
# theta (real, npar = 2 N^2 + 2 M (N + 1) + 2 N^2 + 2 = 18466 at N = 64, M = 16):
#   [Re f_1 (N^2), Im f_1 (N^2), Re b (M), Im b (M), Re W (N M, row-major (N, M)), Im W (N M)  |  Re f_2, Im f_2  |  Re lam, Im lam]
# Configurations at 8x8 are packed in uint64, bit i = 1 iff sigma_i = +1 (the cx_lib.bits convention; 64 sites fill the word exactly).
N8, L8 = 64, 8
_BIT64 = np.uint64(1) << np.arange(64, dtype=np.uint64)


def pack64(sig):
    """(M, 64) +-1 -> (M,) uint64, bit i set iff sig[:, i] > 0."""
    return np.bitwise_or.reduce((np.asarray(sig) > 0).astype(np.uint64) * _BIT64[None, :], axis=1)


def unpack64(w):
    """(M,) uint64 -> (M, 64) int8 +-1."""
    w = np.asarray(w, np.uint64)
    return (((w[:, None] >> np.arange(64, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.int8) * 2 - 1)


def lat8(L=8):
    """bond list (J1 then J2; per site i: right, up | diag, anti-diag; the pp2_lemma order), translation images, JB, HW = JB / 2."""
    N = L * L; site = lambda x, y: (x % L) + L * (y % L); nn_, nnn = [], []
    for y in range(L):
        for x in range(L):
            i = site(x, y); nn_ += [(i, site(x + 1, y)), (i, site(x, y + 1))]; nnn += [(i, site(x + 1, y + 1)), (i, site(x + 1, y - 1))]
    bl = nn_ + nnn; BI = np.array([b[0] for b in bl]); BJ = np.array([b[1] for b in bl]); JB = np.array([1.0] * len(nn_) + [0.5] * len(nnn))
    IMG = np.array([[site(x + tx, y + ty) for y in range(L) for x in range(L)] for ty in range(L) for tx in range(L)])      # (N, N): z_g[s] = sigma[IMG[g, s]]
    return dict(L=L, N=N, BI=BI, BJ=BJ, JB=JB, HW=JB / 2, NB=len(bl), IMG=IMG)


def canon64(words):
    """canonical representative of each configuration under the 64 translations x spin flip (flip = complement at S^z = 0): min over the 128 images."""
    w = np.asarray(words, np.uint64); sig = unpack64(w); IMG = lat8()['IMG']; best = w.copy()
    for g in range(IMG.shape[0]):
        p = pack64(sig[:, IMG[g]]); best = np.minimum(best, np.minimum(p, ~p))
    return best


def unpack8(th, N=N8, M=16):
    """theta -> f1, b, W, f2, lam (complex)."""
    c = lambda a, b: a + 1j * b; n2 = N * N
    f1 = c(th[:n2], th[n2:2 * n2]).reshape(N, N); o = 2 * n2
    b = c(th[o:o + M], th[o + M:o + 2 * M]); o += 2 * M
    W = c(th[o:o + N * M], th[o + N * M:o + 2 * N * M]).reshape(N, M); o += 2 * N * M
    f2 = c(th[o:o + n2], th[o + n2:o + 2 * n2]).reshape(N, N); o += 2 * n2
    return f1, b, W, f2, c(th[o], th[o + 1])


def pack8(f1, b, W, f2, lam):
    """inverse of unpack8 (numpy)."""
    r = lambda a: np.asarray(a).reshape(-1)
    return np.concatenate([r(np.real(f1)), r(np.imag(f1)), r(np.real(b)), r(np.imag(b)), r(np.real(W)), r(np.imag(W)), r(np.real(f2)), r(np.imag(f2)),
                           [np.real(lam), np.imag(lam)]]).astype(np.float64)


def make_model8(SP=None, M=None, _broken=None):
    """the 8x8 model dict, same keys as make_model's (fwd_j, jac_j, bits, N, NP1, npar, logA_one) plus logpsi_one (complex log B).
    _broken (TEST ONLY): 'nopar' drops the permutation parity (|B| is blind to it at S^z = 0 with these images: it shows in the phase only); 'noflip' drops the spin-flip images' sign; 'conjb' conjugates the RBM bias; 'grad_f2' / 'grad_lam' / 'grad_W' keep the value but
    cut the derivative with respect to f_2 / lam / W (stop_gradient)."""
    import jax, jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    M = int(M if M is not None else (SP or {}).get('rbm_M', 16)); N = N8; NH = N // 2; lt = lat8(); n2 = N * N
    NP1 = 2 * n2 + 2 * M * (N + 1); npar = NP1 + 2 * n2 + 2
    IMG = jnp.asarray(np.concatenate([lt['IMG'], lt['IMG']]).astype(np.int32)); FL = jnp.asarray(np.concatenate([np.ones(N), (np.ones(N) if _broken == 'noflip' else -np.ones(N))]))
    AR = jnp.arange(64, dtype=jnp.uint64)

    def bits(Sx):
        return (((Sx[:, None] >> AR[None, :]) & jnp.uint64(1)).astype(jnp.float64) * 2 - 1)

    def logcosh(z):
        s = jnp.where(jnp.real(z) >= 0, 1.0, -1.0); return s * z + jnp.log1p(jnp.exp(-2.0 * s * z)) - jnp.log(2.0)

    def det_par(f, zz):
        order = jnp.argsort(-zz, stable=True); U = order[:NH]; D = order[NH:]
        sg, ld = jnp.linalg.slogdet(f[U][:, D])
        up = (zz > 0).astype(jnp.int32); dn = 1 - up; inv = jnp.sum(up * (jnp.cumsum(dn) - dn))
        par = 1.0 if _broken == 'nopar' else 1.0 - 2.0 * (inv % 2).astype(jnp.float64)
        return ld, sg * par

    def logpsi_one(th, sigma):
        f1, b, W, f2, lam = unpack8(th, N, M)
        if _broken == 'conjb': b = jnp.conj(b)
        if _broken == 'grad_f2': f2 = jax.lax.stop_gradient(f2)
        if _broken == 'grad_lam': lam = jax.lax.stop_gradient(lam)
        if _broken == 'grad_W': W = jax.lax.stop_gradient(W)
        Z = FL[:, None] * sigma.astype(jnp.float64)[IMG]                                # (128, N)
        ld1, s1 = jax.vmap(lambda zz: det_par(f1, zz))(Z); ld2, s2 = jax.vmap(lambda zz: det_par(f2, zz))(Z)
        lr = jnp.sum(logcosh(b[None, :] + Z.astype(jnp.complex128) @ W), 1)              # (128,) complex
        l1 = lr + ld1; l2 = lr + ld2
        m = jax.lax.stop_gradient(jnp.maximum(jnp.max(jnp.real(l1)), jnp.max(jnp.real(l2)) + jnp.log(jnp.maximum(jnp.abs(lam), 1e-300))))
        S = jnp.sum(s1 * jnp.exp(l1 - m)) + lam * jnp.sum(s2 * jnp.exp(l2 - m))
        return m + jnp.log(S)

    def logA_one(th, sigma):
        return jnp.real(logpsi_one(th, sigma))                                           # log|B| = Re log B
    fwd_j = jax.jit(lambda th, Sx: jax.vmap(lambda s_: logA_one(th, s_))(bits(Sx)))
    jac_j = jax.jit(jax.vmap(jax.grad(logA_one), in_axes=(None, 0)))
    psi_j = jax.jit(lambda th, Sx: jax.vmap(lambda s_: logpsi_one(th, s_))(bits(Sx)))
    return dict(arm='pp8', fwd_j=fwd_j, jac_j=jac_j, psi_j=psi_j, bits=bits, N=N, M=M, NP1=NP1, npar=npar, logA_one=logA_one, logpsi_one=logpsi_one)


def theta8_to_pp2(th, M=16):
    """theta -> pp2_lemma.PP2 (complex f1, f2, lam, RBM 'inner', translations): the object pp2_lemma's evaluators take."""
    import pp2_lemma as PL
    f1, b, W, f2, lam = unpack8(np.asarray(th), N8, M)
    return PL.PP2(f1, f2, lam, b, W, L=L8, proj='translations', rbm='inner')


def pp2_to_theta8(p):
    assert p.L == L8 and p.rbm == 'inner' and p.has_p2, (p.L, p.rbm, p.has_p2)
    return pack8(np.asarray(p.f1), np.asarray(p.b), np.asarray(p.W), np.asarray(p.f2), complex(p.lam))


def warm_theta8(SP, mdl=None):
    """theta of B_0 = P_1 + lam P_2.  P_1 = the production RBM+PP (SP['warm_npz'], f real -> complex with Im f_1 = f1_im_sigma * rms(f) * N(0,1), default 0;
    complex RBM kept).  P_2: f_2 = f_1 + p2_sigma_rel * rms(f_1) (xi_re + i xi_im) / sqrt 2, xi ~ N(0,1) (seed p2_seed): noise at FINITE separation (p2_sigma_rel, default 0.3),
    lam = lambda_init_re + i lambda_init_im (default 0.1).  The separation ||f_2 - f_1|| / ||f_1|| is returned in SP['_p2_sep'] when SP is a dict.
    SP['warm_theta8'] (a saved 18466-vector) overrides everything."""
    if SP.get('warm_theta8'):
        w = np.load(SP['warm_theta8']); assert w.dtype == np.float64 and w.size == (mdl['npar'] if mdl else 18466), w.shape; return w
    z = np.load(SP['warm_npz']); f = np.asarray(z['f'], np.float64); assert f.shape == (N8, N8) and bool(z['rbm']) and str(z['proj']) == 'translations'
    rng = np.random.default_rng(int(SP.get('p2_seed', 0))); rms = float(np.sqrt(np.mean(f * f)))
    f1 = f + 1j * float(SP.get('f1_im_sigma', 0.0)) * rms * rng.standard_normal(f.shape)
    xi = rng.standard_normal(f.shape) + 1j * rng.standard_normal(f.shape)
    f2 = f1 + float(SP.get('p2_sigma_rel', 0.3)) * rms * xi / np.sqrt(2.0)
    lam = complex(float(SP.get('lambda_init_re', 0.1)), float(SP.get('lambda_init_im', 0.0)))
    b = np.asarray(z['bre']) + 1j * np.asarray(z['bim']); W = np.asarray(z['Wre']) + 1j * np.asarray(z['Wim'])
    SP['_p2_sep'] = float(np.linalg.norm(f2 - f1) / np.linalg.norm(f1))
    return pack8(f1, b, W, f2, lam)


class Labels8:
    """labels_fn(configs) -> ln T for the 8x8 guide B (a pp2_lemma.PP2 = the round's guide B_k, NOT the model being fitted):
         T = s_G B e^{c f_J[B]},  ln T(x) = Re ln B(x) + add_x,  add_x = pp2_lemma.fJ_fixed (mode 'fix' by default: no-kept rows -> node -69, else c sign(f_J) min(|f_J|, f0)).
    cfg: uint64 words (pack64 convention) or (M, 64) +-1 arrays.  sign_fn: (M, 64) +-1 int8 -> (M,) +-1 (the stored s_G; a test stub is fine).  Only the amplitude is returned;
    the sign enters through f_J (kept / blocked bonds) and through the partners' blocking.  No normalisation (the offset c of the fit absorbs it).
    _break (TEST ONLY): 'c_sign' (c -> -c), 'no_cap' (f0 = inf), 'sign_other' (kept/blocked decided by another sign_fn: sign_fn_alt)."""

    def __init__(self, guide, sign_fn, EP, c=4.0, f0=None, mode='fix', dtype='f64', chunk=256, _break=None, sign_fn_alt=None):
        self.guide, self.sign_fn, self.EP, self.c, self.f0, self.mode, self.dtype, self.chunk = guide, sign_fn, float(EP), float(c), f0, mode, dtype, int(chunk)
        self._break, self.sign_fn_alt = _break, sign_fn_alt
        if mode == 'fix': assert f0 is not None, "mode 'fix' needs f0"

    def __call__(self, cfg, orb=None):
        import pp2_lemma as PL
        cfg = np.asarray(cfg); X = unpack64(cfg.astype(np.uint64)) if cfg.ndim == 1 else cfg
        X = X.astype(np.float64); out = np.empty(len(X)); c = -self.c if self._break == 'c_sign' else self.c
        f0 = np.inf if self._break == 'no_cap' else self.f0; sf = self.sign_fn_alt if self._break == 'sign_other' else self.sign_fn
        for i in range(0, len(X), self.chunk):
            xb = X[i:i + self.chunk]
            o = PL.fJ_fixed(self.guide, xb, sf, self.EP, c, f0, self.mode, at_y=False, dtype=self.dtype)
            lB = o['logB'] if 'logB' in o else PL.logB(self.guide, xb, self.dtype)
            out[i:i + len(xb)] = np.asarray(np.real(lB), np.float64) + np.asarray(o['add_x'], np.float64)
        return out


class Neighbours8:
    """neighbours_fn for build_pool at 8x8: all NB = 256 one-swap neighbours of each x under the guide B (pp2_lemma.one_swap_ratios), blocked = anti-aligned bond with
    s(x) s(y) > 0 under sign_fn.  Returns the build_pool tuple (y words (S, NB), hw, valid, blocked, dlnB = ln|B(y)/B(x)|, y orbit ids = 0 (labels_fn ignores them))."""

    def __init__(self, guide, sign_fn, dtype='f64', sign_chunk=16384, walker_chunk=256):
        self.guide, self.sign_fn, self.dtype, self.sc, self.wc = guide, sign_fn, dtype, sign_chunk, walker_chunk; lt = lat8()
        self.BI, self.BJ, self.hw = lt['BI'], lt['BJ'], lt['HW']; self.MASK = _BIT64[self.BI] | _BIT64[self.BJ]

    def sign(self, X):
        o = []
        for i in range(0, len(X), self.sc): o.append(np.asarray(self.sign_fn(X[i:i + self.sc])))
        s = np.concatenate(o).astype(np.float64) if o else np.zeros(0); assert np.all(np.abs(s) == 1), 'sign_fn must return +-1'; return s

    def __call__(self, cfg, orb=None, sub=4096):
        w = np.asarray(cfg, np.uint64)
        if len(w) > sub:                                                                                                          # memory: (S, NB) arrays in sub-blocks
            parts = [self(w[i:i + sub]) for i in range(0, len(w), sub)]
            return (np.concatenate([q[0] for q in parts]), self.hw, np.concatenate([q[2] for q in parts]), np.concatenate([q[3] for q in parts]),
                    np.concatenate([q[4] for q in parts]), np.zeros((len(w), len(self.BI)), np.int64))
        import pp2_lemma as PL
        X = unpack64(w); S = len(w)
        r = np.asarray(PL.one_swap_ratios(self.guide, X.astype(np.float64), self.dtype, walker_chunk=self.wc))                    # (S, NB) complex, 0 if aligned
        valid = X[:, self.BI] != X[:, self.BJ]; ar = np.abs(r)
        with np.errstate(divide='ignore'): dl = np.where(valid & (ar > 0), np.log(np.where(ar > 0, ar, 1.0)), -700.0)
        y = np.where(valid, w[:, None] ^ self.MASK[None, :], w[:, None])
        sx = self.sign(X.astype(np.int8)); sy = np.ones((S, len(self.BI)))
        iv = np.nonzero(valid); sy[iv] = self.sign(unpack64(y[iv]))
        blocked = valid & (sx[:, None] * sy > 0)
        return y, self.hw, valid, blocked, dl, np.zeros(y.shape, np.int64)


def draw_from_candidates(cand_words, cand_lnB):
    """DEV / TEST sampler only (real 8x8 pools come from pool8.py): tempered resampling from a fixed candidate set drawn UNIFORMLY over S^z = 0 (importance weights
    |B|^(2 beta), multinomial resampling).  Returns the draw_fn hook (beta, S, rng) -> (words, lnB, orbit ids = canon64 words as int64)."""
    cw = np.asarray(cand_words, np.uint64); cl = np.asarray(cand_lnB, np.float64); cor = canon64(cw).view(np.int64)

    def draw_fn(beta, S, rng):
        lw = 2.0 * beta * (cl - cl.max()); p = np.exp(lw); p /= p.sum(); i = rng.choice(len(cl), size=S, p=p)
        return cw[i], cl[i], cor[i]
    return draw_fn


def build8(OUT, SP, draw_fn, sign_fn):
    """8x8 pool: x ~ draw_fn (tempered ladder of |B_k|^(2 beta)), K blocked partners, labels ln T = Labels8 (guide = theta8 of SP['guide_theta8'] or the SP warm start).
    SP: S, betas, K, seed, decs_rel, EP, c, f0, mode, guide_theta8 | warm spec.  Writes OUT/pool.npz + pool_summary.json (+ .sha256)."""
    os.makedirs(OUT, exist_ok=True); T0 = time.time(); log = lambda m: print(f'[{time.time() - T0:6.0f}s] {m}', flush=True)
    th = np.load(SP['guide_theta8']) if SP.get('guide_theta8') else warm_theta8(dict(SP)); guide = theta8_to_pp2(th, int(SP.get('rbm_M', 16)))
    lab = Labels8(guide, sign_fn, SP['EP'], SP.get('c', 4.0), SP.get('f0'), SP.get('mode', 'fix'))
    P = build_pool(draw_fn, Neighbours8(guide, sign_fn), lab, int(SP['S']), [float(b) for b in SP.get('betas', [1.0, 0.5, 0.25])], int(SP.get('K', 2)),
                   int(SP.get('seed', 1)), [int(d) for d in SP.get('decs_rel', list(range(2, -7, -1)))], ho_fn=ho_hash, log=log)
    path = os.path.join(OUT, 'pool.npz'); np.savez(path, **P); summ = pool_summary(P)
    summ.update(spec={k: v for k, v in SP.items()}, pool_sha256=sha(path), code_sha256=sha(os.path.abspath(__file__)), sec=time.time() - T0)
    jdump(os.path.join(OUT, 'pool_summary.json'), summ); open(path + '.sha256', 'w').write(summ['pool_sha256'] + '  pool.npz\n'); log(f'pool written {path}'); return P


# ============================================================================================ fit
def fit(OUT, SP):
    import jax, jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    os.makedirs(OUT, exist_ok=True); F = os.path.join(OUT, 'pool_fit.json'); PROG = os.path.join(OUT, 'progress.txt'); T00 = time.time()

    def progress(msg):
        open(PROG, 'a').write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}\n'); print(msg, flush=True)
    res = dict(spec=SP, code_sha256={f: sha(os.path.join(HERE, f)) for f in ('pool_fit.py', 'cx_lib.py', 'rbmpp_lib.py')})
    if SP.get('code_sha256'):
        bad = [f for f, h in SP['code_sha256'].items() if res['code_sha256'].get(f) != h]
        if bad: res['stop'] = f'CODE DRIFT {bad}'; jdump(F, res); progress(res['stop']); sys.exit(3)
    if SP.get('pool_sha256') and sha(SP['pool']) != SP['pool_sha256']:
        res['stop'] = 'POOL sha256 mismatch'; jdump(F, res); progress(res['stop']); sys.exit(3)
    P = load_pool(SP['pool']); res['pool'] = SP['pool']; res['pool_summary'] = pool_summary(P)
    B = int(SP.get('B', 8208)); B_ACC = int(SP.get('B_acc', B)); B_HO = int(SP.get('B_ho', 4096)); LAM_B = float(SP.get('lam_b', 1.0))
    LAM0 = float(SP.get('lam0', 1e-2)); LAM_CAP = float(SP.get('lam_cap', 1.0)); LAM_MIN = float(SP.get('lam_floor', 1e-6))
    TAU = float(SP.get('tau', 0.05)); TAU_MIN = float(SP.get('tau_min', TAU)); ANNEAL = int(SP.get('anneal_steps', 20)); Z_ACC = float(SP.get('z_acc', 2.0))
    HO_EVERY = int(SP.get('ho_every', 10)); FLAT_TOL = float(SP.get('flat_tol', 0.02)); MAX_STEPS = int(SP['max_steps']); FIT_SEC = float(SP.get('fit_sec', 1e12))
    SAVE_AT = sorted({int(q) for q in SP.get('save_theta_at', [])}); SEED = int(SP.get('seed', 0)); NEG = bool(SP.get('break_negate_step', False))
    mdl = make_model8(SP) if SP.get('arm') == 'pp8' else make_model(SP.get('arm', 'pp2'), int(SP.get('rbm_alpha', 4))); net = Net(mdl); per_sample, grad_from_G, solve_dir = _jfuns()
    theta = jnp.asarray(warm_theta(SP, mdl)); res['npar'] = int(theta.size); res['warm_theta_sha256'] = sha(SP.get('warm_theta') or SP.get('warm_theta8') or SP['warm_npz'])
    if SP.get('test_perturb_sigma'):                  # TEST ONLY (check c): start from warm + sigma N(0,1) (seed 0), far enough that GN must reduce the loss
        theta = theta + float(SP['test_perturb_sigma']) * jnp.asarray(np.random.default_rng(0).standard_normal(theta.size)); res['test_perturb_sigma'] = float(SP['test_perturb_sigma'])
    S_tr = PoolSampler(P, False, 1000 + SEED, LAM_B); S_ho = PoolSampler(P, True, 5000 + SEED, LAM_B)
    HOB = S_ho.batch(B_HO); TRB = PoolSampler(P, False, 7000 + SEED, LAM_B).batch(B_HO)
    S_of = lambda th, bt: net.fwd(th, jnp.concatenate([bt['x'], bt['y'].reshape(-1)]))
    bl = lambda th, c, bt: per_sample(S_of(th, bt), bt['lx'], bt['ly'], bt['wb'], c)
    c = float(np.mean(np.asarray(net.fwd(theta, HOB['x'])) - np.asarray(HOB['lx'])))
    L_ho0 = float(jnp.mean(bl(theta, c, HOB)[0])); L_tr0 = float(jnp.mean(bl(theta, c, TRB)[0]))
    curve = [dict(step=0, t=0.0, L_ho=L_ho0, L_tr=L_tr0, c=c)]; progress(f'[pool_fit] npar {theta.size} start held-out {L_ho0:.4e} train(fixed) {L_tr0:.4e}  pool {res["pool_summary"]["S"]}')
    tau = TAU; tau_changed = 0; tau_hist = [dict(step=0, tau=tau)]; hist = []; lam_rel = max(LAM0, LAM_MIN); n = 0; t0 = time.time(); key = None
    saved = []
    while n < MAX_STEPS and time.time() - t0 < FIT_SEC:
        n += 1; ts = time.time()
        bt = S_tr.batch(B); Xall = jnp.concatenate([bt['x'], bt['y'].reshape(-1)])
        S = net.fwd(theta, Xall); G = net.jac(theta, Xall)
        dP, dc, prms, sc, rr = solve_dir(G, S, bt['lx'], bt['ly'], bt['wb'], c, lam_rel, tau); del G
        if NEG: dP, dc = -dP, -dc                       # TEST ONLY
        dc = float(dc); ab = S_tr.batch(B_ACC)
        L_old = bl(theta, c, ab)[0]; cand = theta + dP; L_new = bl(cand, c + dc, ab)[0]
        dmean, dse = strat_mean_se(L_new - L_old, ab['g'], ab['ng']); lam_used = lam_rel
        if not np.isfinite(dmean) or dmean > Z_ACC * dse: ok = False; verdict = 'worse'; lam_rel = min(lam_rel * 4, LAM_CAP)
        elif dmean < -Z_ACC * dse: ok = True; verdict = 'better'; lam_rel = max(lam_rel / 2, LAM_MIN)
        else: ok = True; verdict = 'noise'
        if ok: theta = cand; c = c + dc
        if n in SAVE_AT:
            np.save(os.path.join(OUT, f'theta_step{n}.npy'), np.asarray(theta)); np.save(os.path.join(OUT, f'offset_c_step{n}.npy'), np.asarray(c)); saved.append(n)
        hist.append(dict(step=n, ok=ok, verdict=verdict, lam_rel=lam_used, lam_next=lam_rel, dmean=dmean, dse=dse, pred_rms=float(prms), trust_scale=float(sc), tau=tau,
                         loss_batch=float(rr), c=c, sec=time.time() - ts))
        if n % HO_EVERY == 0:
            curve.append(dict(step=n, t=time.time() - t0, L_ho=float(jnp.mean(bl(theta, c, HOB)[0])), L_tr=float(jnp.mean(bl(theta, c, TRB)[0])), c=c))
            progress(f'  step {n}: held-out {curve[-1]["L_ho"]:.4e} train {curve[-1]["L_tr"]:.4e} acc {np.mean([h["ok"] for h in hist[-HO_EVERY:]]):.2f} lam_rel {lam_rel:.1e} '
                     f'tau {tau:.4f} trust {float(sc):.2f} {hist[-1]["sec"]:.1f}s/step')
            if tau > TAU_MIN and n - tau_changed >= ANNEAL:          # cx_r1fit T1s-2 anneal
                cb = [q for q in curve if q['step'] <= n - ANNEAL]
                if cb and abs(cb[-1]['L_ho'] - curve[-1]['L_ho']) / curve[-1]['L_ho'] < FLAT_TOL:
                    tau = max(tau / 2, TAU_MIN); tau_changed = n; tau_hist.append(dict(step=n, tau=tau)); progress(f'  step {n}: held-out flat -> tau {tau:.4f}')
            res['fit'] = dict(hist=hist, curve=curve, tau_hist=tau_hist); jdump(F, res)
    np.save(os.path.join(OUT, 'theta.npy'), np.asarray(theta)); np.save(os.path.join(OUT, 'offset_c.npy'), np.asarray(c))
    ths = [np.load(os.path.join(OUT, f'theta_step{k}.npy')) for k in saved] + [np.asarray(theta)]
    tbar = np.mean(np.stack(ths), 0); np.save(os.path.join(OUT, 'theta_bar.npy'), tbar)
    Lf = float(jnp.mean(bl(theta, c, HOB)[0]))
    res['fit'] = dict(hist=hist, curve=curve, tau_hist=tau_hist, steps=n, fit_sec=time.time() - t0, acceptance_all=float(np.mean([h['ok'] for h in hist])) if hist else None,
                      median_sec_per_step=float(np.median([h['sec'] for h in hist])) if hist else None, L_ho_final=Lf, saved_steps=saved,
                      theta_bar_of=[f'theta_step{k}.npy' for k in saved] + ['theta.npy'], theta_bar_sha256=sha(os.path.join(OUT, 'theta_bar.npy')))
    res['sec'] = time.time() - T00; res['done'] = True; jdump(F, res)
    progress(f'DONE pool_fit steps {n} held-out {L_ho0:.4e} -> {Lf:.4e}  theta_bar of {len(ths)} iterates  {res["sec"]:.0f}s')
    return res


# ============================================================================================ build (6x6, exact tables)
def build6(OUT, SP):
    import jax
    jax.config.update('jax_enable_x64', True)
    os.makedirs(OUT, exist_ok=True); T0 = time.time()
    log = lambda m: print(f'[{time.time() - T0:6.0f}s] {m}', flush=True)
    lB = np.load(SP['la_B']).astype(np.float64); sg = np.load(SP['sign'])
    ex = Exact6(lB, sg); lab = TableLabels(ex, SP['target'], normalise=True)
    labels_fn = lambda cfg, orb=None: lab(cfg, orb)
    P = build_pool(ex.draw_fn, ex.neighbours_fn, labels_fn, int(SP['S']), [float(b) for b in SP.get('betas', [1.0, 0.5, 0.25])], int(SP.get('K', 2)),
                   int(SP.get('seed', 1)), [int(d) for d in SP.get('decs_rel', list(range(2, -7, -1)))], ho_fn=ho_hash, log=log)
    if SP.get('check_canon', True):                    # the pool's x configs are reps: canon must return their own orbit index (labels_fn path)
        k = min(20000, P['x'].size); o = ex.canon(P['x'][:k]); assert np.array_equal(o, P['x_orb'][:k]), 'canon(x) != drawn orbit'
        oy = ex.canon(P['y'][:k // 4].reshape(-1)); assert np.array_equal(oy, P['y_orb'][:k // 4].reshape(-1)), 'canon(y) != partner orbit'
    path = os.path.join(OUT, 'pool.npz'); np.savez(path, **P)
    summ = pool_summary(P); summ.update(spec=SP, la_B_sha256=sha(SP['la_B']), sign_sha256=sha(SP['sign']), target_sha256=lab.sha256, pool_sha256=sha(path),
                                        code_sha256=sha(os.path.abspath(__file__)), sec=time.time() - T0)
    jdump(os.path.join(OUT, 'pool_summary.json'), summ); open(path + '.sha256', 'w').write(summ['pool_sha256'] + '  pool.npz\n')
    log(f'pool written {path} sha {summ["pool_sha256"][:12]}; counts ' + ' '.join(f'{d}:{v["all"]}' for d, v in summ['counts'].items()))
    return P, ex, lab


if __name__ == '__main__':
    sys.path.insert(0, HERE); sys.path.append('/project/theorie/a/A.Otaifi/Chatty/runs/compress6/code')
    mode, OUT, SPF = sys.argv[1:4]; SP = json.load(open(SPF))
    if mode == 'build': build6(OUT, SP)
    elif mode == 'fit': fit(OUT, SP)
    else: raise SystemExit(f'unknown mode {mode}')
