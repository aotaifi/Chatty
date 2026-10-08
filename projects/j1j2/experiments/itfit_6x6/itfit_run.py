"""Stage 1 of experiments/itfit_6x6 (README.md): projected imaginary-time evolution with the frozen FN Hamiltonian as
the amplitude write-back of the FN/Krylov loop, exact 6x6 sector, residual factor b = |psi_P| exp(f_theta).

Arm kinds (SPEC['arm']):
  'ite'  : per loop iteration k, K steps; step n: exact table of b_n, target t_n = scheme(b_n) on F_k = H_FN[a_k, s_k]
           (scheme 'semi': (b + tau K b)/(1 + tau (D - E)); 'explicit': (1 - tau (F - E)) b; 'oracle': phi_k), then
           n_in Gauss-Newton (Levenberg-Marquardt, minSR form) iterations on fresh batches:
             fit 'Q': energy metric of the target, rows sqrt(c_xy) [(f - g)_x - (f - g)_y] on K_b kept bonds of x,
                      c_xy = iw_x |H_xy| t_y/t_x (n_kept / K_b);
             fit 'P': pointwise infidelity, rows sqrt(iw_x) [(f - g)_x - <f - g>]   (g = log t_n - log|psi_P|).
           distributions: 'i' x ~ b_n^2, 'ii' x ~ a_k phi_k (both unweighted), 'iii' x ~ b_n with weights b_n (-> b_n^2).
  'vmc'  : same-capacity fixed-sign VMC: minSR on <H>(b, s_k), local energies over all valid bonds, samples x ~ b^2
           ('i') or b (iii, weights b) from an exact table refreshed every 'refresh' steps (reweighted in between).
Every loop iteration: Krylov sign step on the new amplitude, exact <H>, E_FN, frac_k, lambda_max(F_k), wall summary.
  python itfit_run.py OUT SPEC.json
"""
import json, os, sys, time
import numpy as np
from itfit_common import Exact, log, dump, N, f64, SS, E0_SITE
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.flatten_util import ravel_pytree
jax.config.update('jax_default_matmul_precision', 'highest')   # full fp32

OUT = sys.argv[1]
SPEC = json.load(open(sys.argv[2]))
os.makedirs(OUT, exist_ok=True)
F_OUT = os.path.join(OUT, 'itfit.json')
T00 = time.time()
AR = jnp.arange(N, dtype=jnp.uint64)


def bits(S):
    return (((S[:, None] >> AR[None, :]) & jnp.uint64(1)).astype(jnp.float32) * 2 - 1)


# ============================================================================================ network (writeback net)
NBR = jnp.asarray(np.array([[((x + dx) % 6) + 6 * ((y + dy) % 6) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
                            for y in range(6) for x in range(6)], np.int32))


class ResCNN(nn.Module):
    C: int = 32
    layers: int = 4

    @nn.compact
    def __call__(self, x):
        kw = dict(dtype=jnp.float32, param_dtype=jnp.float32)

        def conv(h):
            g = h[:, NBR, :]
            return nn.Dense(self.C, **kw)(g.reshape(g.shape[0], N, -1))
        h = nn.gelu(conv(x[:, :, None]))
        for _ in range(1, self.layers):
            h = h + nn.gelu(conv(nn.LayerNorm(**kw)(h)))
        h = h.sum(axis=1) / 6.0
        return nn.Dense(1, kernel_init=nn.initializers.zeros, **kw)(h)[:, 0]


INV = jnp.asarray(np.argsort(SS.space_group()[:8], axis=1))           # D4 images (inverse permutations)
SG = jnp.asarray([1.0, -1.0], jnp.float32)


class Model:
    """f(x) = 1/16 sum_{g in D4 x flip} net(g x): exactly invariant under the space group x spin flip."""

    def __init__(self, C, layers, seed=0, head_init=0.0):
        self.net = ResCNN(C, layers)
        params = self.net.init(jax.random.PRNGKey(seed), jnp.zeros((1, N), jnp.float32))['params']
        if head_init:     # tiny random head: f ~ 1e-4 at start, but every layer has a non-zero Jacobian (GN escapes the
            hd = f'Dense_{layers}'; k = params[hd]['kernel']            # zero-head subspace, where only 32 directions move)
            params[hd]['kernel'] = head_init * jax.random.normal(jax.random.PRNGKey(seed + 7), k.shape, k.dtype)
        self.flat0, self.unravel = ravel_pytree(params)
        self.npar = int(self.flat0.size)
        net, unravel = self.net, self.unravel

        def f(flat, X):                                   # X (R, 36) +-1 float32 -> (R,)
            R = X.shape[0]
            Xg = jnp.stack([X[:, INV[k // 2]] * SG[k % 2] for k in range(16)], 1).reshape(R * 16, N)
            return net.apply({'params': unravel(flat)}, Xg).reshape(R, 16).mean(1)
        def f_aug(flat, X, G):                            # one D4 x flip image per row (training; by convexity of the
            Xg = jnp.take_along_axis(X, INV[G // 2], axis=1) * SG[G % 2][:, None]   # quadratic fit losses in f the
            return net.apply({'params': unravel(flat)}, Xg)                      # symmetrised f does at least as well)
        self.f = f; self.f_aug = f_aug
        self._fS = jax.jit(lambda flat, S: f(flat, bits(S)))
        self._fSa = jax.jit(lambda flat, S, G: f_aug(flat, bits(S), G))
        self._jac = jax.jit(jax.vmap(jax.grad(lambda flat, xb: f(flat, xb[None])[0]), in_axes=(None, 0)))
        self._jaca = jax.jit(jax.vmap(jax.grad(lambda flat, xb, g: f_aug(flat, xb[None], g[None])[0]), in_axes=(None, 0, 0)))

    def fS(self, flat, S, chunk=2048):
        out = []
        for i in range(0, S.shape[0], chunk):
            s = S[i:i + chunk]; m = s.shape[0]
            if m < chunk: s = jnp.concatenate([s, jnp.repeat(s[:1], chunk - m)])
            out.append(self._fS(flat, s)[:m])
        return jnp.concatenate(out).astype(f64)

    def table(self, flat, reps):
        return self.fS(flat, reps)

    def jac(self, flat, S, chunk=256):
        out = []
        Xb = bits(S)
        for i in range(0, S.shape[0], chunk):
            x = Xb[i:i + chunk]; m = x.shape[0]
            if m < chunk: x = jnp.concatenate([x, jnp.repeat(x[:1], chunk - m, 0)])
            out.append(self._jac(flat, x)[:m])
        return jnp.concatenate(out)

    def jac_aug(self, flat, S, G, chunk=2048):
        out = []
        Xb = bits(S)
        for i in range(0, S.shape[0], chunk):
            x = Xb[i:i + chunk]; g = G[i:i + chunk]; m = x.shape[0]
            if m < chunk:
                x = jnp.concatenate([x, jnp.repeat(x[:1], chunk - m, 0)]); g = jnp.concatenate([g, jnp.repeat(g[:1], chunk - m)])
            out.append(self._jaca(flat, x, g)[:m])
        return jnp.concatenate(out)

    def fS_aug(self, flat, S, G):
        return self._fSa(flat, S, G).astype(f64)


# ============================================================================================ exact setup
ex = Exact(); sec = ex.sec
lP, sP0 = ex.lP, ex.sP
MASKS = jnp.asarray(SS.MASKS); BI = jnp.asarray(SS.BI); BJ = jnp.asarray(SS.BJ); JB = jnp.asarray(SS.JB)
NB = int(MASKS.shape[0])
lnn = jnp.log(sec.n)
arm = SPEC['arm']


@jax.jit
def _nval_max(x):
    return jnp.max((((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(jnp.int32).sum(1))


NVM = int(max(int(_nval_max(sec.reps[i:i + (1 << 20)])) for i in range(0, sec.D, 1 << 20)))
log(f'max number of valid bonds over the sector: {NVM}')
model = Model(SPEC.get('C', 32), SPEC.get('layers', 4), seed=SPEC.get('seed', 0), head_init=SPEC.get('head_init', 1e-4))
flat = model.flat0
res = dict(spec=SPEC, npar=model.npar, loops=[])
EV = dict(total=0.0)                                            # evaluation counter (see README)


def esite(E):
    return E / N - E0_SITE


def shells(idx_or_x):
    """valid-bond mask, all neighbour configs and their rep indices for configs x (B,)."""
    x = idx_or_x
    valid = (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(bool)
    yall = jnp.where(valid, x[:, None] ^ MASKS[None, :], x[:, None])
    iya = SS.canon(sec.T, sec.reps, yall.reshape(-1)).reshape(x.shape[0], NB)
    return valid, yall, iya


def sample_idx(key, cdf, B):
    uu = jax.random.uniform(key, (B,), f64) * cdf[-1]
    return jnp.clip(jnp.searchsorted(cdf, uu), 0, sec.D - 1)


def make_qbatch(B, Kb):
    @jax.jit
    def batch(key, cdf, liw, s_tab, lt, g):
        k1, k2 = jax.random.split(key)
        idx = sample_idx(k1, cdf, B)
        lw = liw[idx]; iw = jnp.exp(lw - jnp.max(lw)); iw = iw / jnp.sum(iw)
        x = sec.reps[idx]
        valid, yall, iya = shells(x)
        kept = valid & (s_tab[idx][:, None] * s_tab[iya] < 0)
        nk = kept.sum(1)
        sc = jnp.where(kept, jax.random.uniform(k2, kept.shape), -1.0)
        _, bsel = jax.lax.top_k(sc, Kb)
        ok = jnp.take_along_axis(kept, bsel, 1)
        iy = jnp.take_along_axis(iya, bsel, 1)
        y = x[:, None] ^ MASKS[bsel]
        fac = (nk / jnp.maximum(jnp.minimum(nk, Kb), 1)).astype(f64)
        c = jnp.where(ok, 0.5 * JB[bsel] * jnp.exp(lt[iy] - lt[idx][:, None]), 0.0) * fac[:, None] * iw[:, None]
        nval = valid.sum(1)
        return x, y, g[idx], g[iy], c, iw, nval
    return batch


def make_pbatch(B):
    @jax.jit
    def batch(key, cdf, liw, s_tab, lt, g):
        idx = sample_idx(key, cdf, B)
        lw = liw[idx]; iw = jnp.exp(lw - jnp.max(lw)); iw = iw / jnp.sum(iw)
        x = sec.reps[idx]
        valid = (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(bool)
        return x, g[idx], iw, valid.sum(1)
    return batch


def q_loss(fx, fy, gx, gy, c):
    d = (fx - gx)[:, None] - (fy - gy)
    return 0.5 * jnp.sum(c * d * d) / N


def p_loss(fx, gx, iw):
    e = fx - gx; m = jnp.sum(iw * e)
    return jnp.sum(iw * (e - m) ** 2)


@jax.jit
def gn_prep(J):
    """Levenberg-Marquardt in minSR form with Marquardt column scaling (Jacobian columns normalised to unit rms so the
    damping does not freeze small-gradient layers).  Returns scale, scaled Jacobian and the R x R Gram matrix."""
    sc = jnp.sqrt(jnp.mean(J * J, 0)); sc = sc + 1e-8 * jnp.max(sc) + 1e-30
    Js = J / sc[None, :]
    return sc, Js, (Js @ Js.T).astype(f64)


@jax.jit
def gn_dir(sc, Js, M, r, lam):
    """d = -S^2 J^T (J S^2 J^T + lam tr/R I)^-1 r."""
    R = M.shape[0]
    al = jnp.linalg.solve(M + (lam * jnp.trace(M) / R + 1e-30) * jnp.eye(R, dtype=f64), r)
    return -(Js.T @ al.astype(jnp.float32)) / sc


@jax.jit
def minsr_dir(Y, eps, lam):
    """standard minSR (Chen & Heyl): d = -Y^T (Y Y^T + lam tr/B I)^-1 eps, final product in float64."""
    M = (Y @ Y.T).astype(f64); R = M.shape[0]
    al = jnp.linalg.solve(M + (lam * jnp.trace(M) / R + 1e-30) * jnp.eye(R, dtype=f64), eps)
    return (-(Y.astype(f64).T @ al)).astype(jnp.float32)


# ============================================================================================ exact per-loop helpers
def loop_start(la, s):
    """exact guide quantities of the loop iteration (guide log amplitude la, sign s): FN diagonal, phi, E_FN, G,
    lambda_max, wall."""
    t0 = time.time()
    Efn, u, info = sec.fn_solve(la, s)
    D, w, Kw = ex.fn_diag(la, s)
    b = w / jnp.linalg.norm(w)
    Eg, Kb = ex.Ef(b, D, s, Kw / jnp.linalg.norm(w))
    out = dict(E_FN_dE_site=esite(Efn), Ef_guide_dE_site=esite(Eg), G_site=(Eg - Efn) / N,
               H_guide_dE_site=esite(ex.energy_H(la, s)), fn_info=info)
    if SPEC.get('lanczos_m', 40):
        lan = ex.lanczos_ext(D, s, m=SPEC.get('lanczos_m', 40))
        out['lambda_max'] = lan[-1]['lmax']
    Wall = D - ex.Hdiag; p = u * u
    out['wall'] = dict(mean_phi2=float(jnp.sum(p * Wall)), max=float(jnp.max(Wall)),
                       phi2_gt10=float(jnp.sum(jnp.where(Wall > 10, p, 0.0))),
                       phi2_gt100=float(jnp.sum(jnp.where(Wall > 100, p, 0.0))))
    out['sec'] = time.time() - t0
    return dict(la=la, Efn=Efn, u=u, D=D, w=w, Eg=Eg, s=s, info=out)


def ef_of_table(ftab, D, s):
    b = ex.pos_vec(lP + ftab)
    E, Kb = ex.Ef(b, D, s)
    return E, b, Kb


def hop(la_b, s, tau):
    """one semi-implicit FN step of b on its OWN frozen FN Hamiltonian F[b, s] (one hop of b): returns log c."""
    D, w, Kw = ex.fn_diag(la_b, s)
    b = w / jnp.linalg.norm(w); Kb = Kw / jnp.linalg.norm(w); del w, Kw
    E = float(b @ (D * b) - b @ Kb)
    c = (b + tau * Kb) / (1.0 + tau * (D - E))
    return jnp.log(jnp.maximum(c / ex.sqn, 1e-300))


def loop_end(flat, L, k):
    """exact end-of-iteration scores, the Krylov sign step, and the one-hop composite guide c = T_{F[b, s']}[b]."""
    t0 = time.time()
    ftab = model.table(flat, sec.reps)
    la_b = lP + ftab
    E_end, b, Kb = ef_of_table(ftab, L['D'], L['s'])
    G = L['Eg'] - L['Efn']
    frac = (L['Eg'] - E_end) / G
    H_own = ex.energy_H(la_b, L['s'])
    tau = SPEC.get('tau', 1.0)
    t = (b + tau * Kb) / (1.0 + tau * (L['D'] - E_end)); del Kb
    E_t, _ = ex.Ef(t, L['D'], L['s']); del t
    s_new, E_kry, kinfo = sec.krylov(b, L['s'])
    la_c = hop(la_b, s_new, tau)
    E_c, _ = ex.Ef(ex.pos_vec(la_c), L['D'], L['s'])
    out = dict(it=k + 1, frac=frac, Ef_end_dE_site=esite(E_end), Ef_end=E_end, H_oldsign_dE_site=esite(H_own),
               H_kry_dE_site=esite(E_kry), kry=kinfo,
               rms_f=float(jnp.sqrt(jnp.sum(b * b * (ftab - jnp.sum(b * b * ftab)) ** 2))),
               frac_hop_Fk=(L['Eg'] - E_t) / G,                       # one exact F_k step on top of the fitted net
               frac_composite=(L['Eg'] - E_c) / G,                    # composite guide c (own-F hop, new sign)
               H_composite_dE_site=esite(ex.energy_H(la_c, s_new)))
    if not SPEC.get('composite'):
        _, _, ic = sec.fn_solve(la_c, s_new); out['E_FN_composite_dE_site'] = ic['dE_FN_site']
    out['sec'] = time.time() - t0
    return out, s_new, la_b, la_c


# ============================================================================================ arms
def run_ite(flat, L, k, lrec):
    sch, dist, fit = SPEC['scheme'], SPEC['dist'], SPEC['fit']
    tau = SPEC.get('tau', 0.3); K = SPEC['K']; n_in = SPEC['n_in']
    B = SPEC.get('B', 1024); Kb = SPEC.get('Kb', 4); Bv = SPEC.get('Bv', 2048)
    D, s, u = L['D'], L['s'], L['u']
    bq = make_qbatch(B, Kb) if fit == 'Q' else make_pbatch(B)
    bqv = make_qbatch(Bv, Kb) if fit == 'Q' else make_pbatch(Bv)
    lphi = jnp.log(jnp.maximum(u / ex.sqn, 1e-300))
    lam = lrec.get('lam', SPEC.get('lam', 1e-3 if SPEC.get('inner', 'sng') == 'sng' else 1.0))
    steps = []; cum_fitloss = 0.0
    key = jax.random.PRNGKey(1000 * (k + 1) + SPEC.get('seed', 0))
    E_prev_target = None
    for n in range(K):
        t0 = time.time()
        la = L['la'] if n == 0 else lP + model.table(flat, sec.reps)      # step 0 starts from the loop's guide
        b = ex.pos_vec(la)
        E_b, Kbv = ex.Ef(b, D, s)
        if sch == 'semi':
            t = (b + tau * Kbv) / (1.0 + tau * (D - E_b))
        elif sch == 'explicit':
            t = b - tau * (D * b - Kbv - E_b * b)
            t = jnp.maximum(t, 1e-300 * jnp.max(t))
        elif sch == 'oracle':
            t = u
        E_t, _ = ex.Ef(t, D, s)
        lt = jnp.log(jnp.maximum(t / ex.sqn, 1e-300))
        g = lt - lP
        del Kbv
        if E_prev_target is not None: cum_fitloss += E_b - E_prev_target
        # distribution
        if dist == 'i':
            lw = 2 * la + lnn; liw = jnp.zeros_like(la)
        elif dist == 'ii':
            lw = jnp.log(jnp.maximum(L['w'], 1e-300)) + jnp.log(jnp.maximum(u, 1e-300)); liw = jnp.zeros_like(la)
        elif dist == 'iii':
            lw = la + lnn; liw = la
        cdf = jnp.cumsum(jnp.exp(lw - jnp.max(lw))); del lw
        liw = liw - jnp.max(liw)
        key, kv = jax.random.split(key)
        vb = bqv(kv, cdf, liw, s, lt, g)
        sec.offload()                                              # H off the device during the fit

        def val_loss(fl):
            if fit == 'Q':
                x, y, gx, gy, c, iw, nval = vb
                fa = model.fS(fl, jnp.concatenate([x, y.reshape(-1)]))
                return float(q_loss(fa[:Bv], fa[Bv:].reshape(Bv, Kb), gx, gy, c))
            x, gx, iw, nval = vb
            return float(p_loss(model.fS(fl, x), gx, iw))
        nv_mean = float(jnp.mean(vb[-1]))
        pts_v = Bv * (1 + Kb) if fit == 'Q' else Bv
        EV['total'] += 2 * pts_v * (1 + nv_mean)                 # targets of the validation batch (two shells)
        L0 = val_loss(flat); Lc = L0; EV['total'] += pts_v
        n_acc = 0; n_rej_run = 0; vhist = [L0]; best_flat = flat
        for it in range(n_in):
            key, kb = jax.random.split(key)
            bd = bq(kb, cdf, liw, s, lt, g)
            key, kg = jax.random.split(key)
            if fit == 'Q':
                x, y, gx, gy, c, iw, nval = bd
                S = jnp.concatenate([x, y.reshape(-1)])
                Gx = jax.random.randint(kg, (B,), 0, 16); GG = jnp.concatenate([Gx, jnp.repeat(Gx, Kb)])
                Jall = model.jac_aug(flat, S, GG); fa = model.fS_aug(flat, S, GG)
                Jx, Jy = Jall[:B], Jall[B:].reshape(B, Kb, -1)
                sq = jnp.sqrt(c).astype(jnp.float32)
                J = (sq[:, :, None] * (Jx[:, None, :] - Jy)).reshape(B * Kb, -1)
                r = (jnp.sqrt(c) * ((fa[:B] - gx)[:, None] - (fa[B:].reshape(B, Kb) - gy))).reshape(-1)
                del Jall, Jx, Jy
                pts = B * (1 + Kb)
            else:
                x, gx, iw, nval = bd
                Gx = jax.random.randint(kg, (B,), 0, 16)
                Jx = model.jac_aug(flat, x, Gx); fx = model.fS_aug(flat, x, Gx)
                w32 = iw.astype(jnp.float32)
                J = jnp.sqrt(w32)[:, None] * (Jx - (w32 @ Jx)[None, :])
                e = fx - gx
                r = jnp.sqrt(iw) * (e - jnp.sum(iw * e))
                del Jx
                pts = B
            nb1 = 1 + float(jnp.mean(nval))
            EV['total'] += (2 * pts * nb1 * (nb1 if (SPEC.get('composite') and k > 0) else 1)) + pts
            if SPEC.get('inner', 'sng') == 'sng':
                # stochastic natural-gradient (minSR form) iterations with a fixed step eta, as in p-tVMC inner loops:
                # every step accepted; the symmetrised validation loss is logged every val_every iterations and the
                # best-validation parameters are kept (selection by the arm's own fit objective).
                flat = flat + SPEC.get('eta', 0.1) * minsr_dir(J, r, lam); del J
                if (it + 1) % SPEC.get('val_every', 25) == 0 or it == n_in - 1:
                    Lt = val_loss(flat); EV['total'] += pts_v; vhist.append(Lt)
                    if np.isfinite(Lt) and Lt < Lc: best_flat, Lc = flat, Lt; n_acc += 1
                    if not np.isfinite(Lt): break
                continue
            sc, Js, M = gn_prep(J)
            del J
            ok = False
            for tr_ in range(SPEC.get('lm_tries', 4)):                 # LM: re-solve on the same batch with more damping
                d = gn_dir(sc, Js, M, r, lam)
                ftry = flat + SPEC.get('eta', 1.0) * d
                Lt = val_loss(ftry); EV['total'] += pts_v
                if np.isfinite(Lt) and Lt < Lc:
                    ok = True; break
                lam = min(lam * 4, 1e4)
            del sc, Js, M
            if ok:
                flat, Lc = ftry, Lt; n_acc += 1; n_rej_run = 0
                if tr_ == 0: lam = max(lam / 3, 1e-6)
            else:
                n_rej_run += 1
            vhist.append(Lc)
            if n_rej_run >= SPEC.get('max_rej', 3): break
            pw = SPEC.get('plateau_window', 25)
            if len(vhist) > pw + 15 and vhist[-1] > (1 - SPEC.get('plateau_tol', 1e-3)) * vhist[-1 - pw]: break
        if SPEC.get('inner', 'sng') == 'sng': flat = best_flat
        st = dict(n=n, E_b_dE_site=esite(E_b), frac_b=(L['Eg'] - E_b) / (L['Eg'] - L['Efn']),
                  E_t_dE_site=esite(E_t), frac_t=(L['Eg'] - E_t) / (L['Eg'] - L['Efn']),
                  step_gain_exact_site=(E_b - E_t) / N, val0=L0, val_end=Lc, val_ratio=Lc / max(L0, 1e-300),
                  n_acc=n_acc, n_it=it + 1, lam=lam, val_hist=vhist, nval_mean=nv_mean, evals=EV['total'],
                  sec=time.time() - t0)
        if steps and E_b > steps[-1]['_E'] + 1e-12 * abs(E_b): st['energy_increase'] = True
        st['_E'] = E_b
        steps.append(st)
        E_prev_target = E_t
        log(f'  step {n}: frac_b {st["frac_b"]:.4f} frac_t {st["frac_t"]:.4f} val {L0:.3e}->{Lc:.3e} '
            f'acc {n_acc}/{it + 1} lam {lam:.1e} evals {EV["total"]:.3e} {st["sec"]:.0f}s')
        lrec['steps'] = steps; dump(F_OUT, res)
    lrec['lam'] = lam
    lrec['cum_fitloss_site_through_step_K-1'] = cum_fitloss / N
    lrec['E_last_target'] = E_prev_target
    return flat


def run_vmc(flat, L, k, lrec, budget):
    dist = SPEC['dist']; B = SPEC.get('B', 1024); eta = SPEC['eta']; lam = SPEC.get('lam', 1e-3)
    beta = 1.0 if dist == 'i' else 0.5
    s = L['s']; D = L['D']
    s_f = s.astype(f64)

    @jax.jit
    def vstep(fl, key, cdf, liw, fref):
        idx = sample_idx(key, cdf, B)
        x = sec.reps[idx]
        valid, yall, iya = shells(x)
        _, vsel = jax.lax.top_k(valid.astype(jnp.int32), NVM)          # the valid bonds first (NVM >= max n_valid)
        vv = jnp.take_along_axis(valid, vsel, 1); yv = jnp.take_along_axis(yall, vsel, 1)
        iyv = jnp.take_along_axis(iya, vsel, 1)
        fx = model.f(fl, bits(x)).astype(f64)
        fy = jax.lax.map(lambda yk: model.f(fl, bits(yk)), yv.T).T.astype(f64)       # (B, NVM), one slot at a time
        lx = lP[idx] + fx; ly = lP[iyv] + fy
        d0 = 0.25 * JB.sum() - 0.5 * (valid * JB[None, :]).sum(1)
        EL = d0 + jnp.sum(jnp.where(vv, 0.5 * JB[vsel] * s_f[idx][:, None] * s_f[iyv]
                                    * jnp.exp(jnp.clip(ly - lx[:, None], -80, 80)), 0.0), 1)
        lr = liw[idx] + 2 * (fx - fref[idx]); rho = jnp.exp(lr - jnp.max(lr)); rho = rho / jnp.sum(rho)
        Eb = jnp.sum(rho * EL)
        return x, rho, EL, Eb, valid.sum(1), 1.0 / jnp.sum(rho * rho) / B

    @jax.jit
    def vupdate(fl, O, rho, EL, Eb):
        r32 = rho.astype(jnp.float32)
        Y = jnp.sqrt(r32)[:, None] * (O - (r32 @ O)[None, :])
        eps = jnp.sqrt(rho) * (EL - Eb)
        return fl + eta * minsr_dir(Y, eps, lam)
    key = jax.random.PRNGKey(5000 + 17 * k + SPEC.get('seed', 0))
    trace = []; ev0 = EV['total']; it = 0; t0 = time.time()
    while EV['total'] - ev0 < budget and time.time() - t0 < SPEC.get('max_sec', 1e9):
        if it % SPEC.get('refresh', 200) == 0:
            fref = model.table(flat, sec.reps); la = lP + fref
            if it > 0 or k > 0:
                E_now, _, _ = ef_of_table(fref, D, s)
                trace.append(dict(it=it, frac=(L['Eg'] - E_now) / (L['Eg'] - L['Efn']),
                                  H_dE_site=esite(ex.energy_H(la, s)), evals=EV['total'] - ev0, sec=time.time() - t0))
                log(f'  vmc it {it}: frac {trace[-1]["frac"]:.4f} <H> {trace[-1]["H_dE_site"]:.4e}')
            lw = 2 * beta * la + lnn
            cdf = jnp.cumsum(jnp.exp(lw - jnp.max(lw))); del lw
            liw = (2 - 2 * beta) * la; liw = liw - jnp.max(liw)
            sec.offload()
        key, kk = jax.random.split(key)
        x, rho, EL, Eb, nval, ess = vstep(flat, kk, cdf, liw, fref)
        flat = vupdate(flat, model.jac(flat, x), rho, EL, Eb)          # symmetrised Jacobian, chunked
        if it % 50 == 0: trace.append(dict(it=it, E_batch_dE_site=esite(float(Eb)), ess=float(ess)))
        EV['total'] += B * (1 + float(jnp.mean(nval))) + B
        it += 1
        if not np.isfinite(float(Eb)): log('  non-finite VMC energy'); break
    fend = model.table(flat, sec.reps); E_now, _, _ = ef_of_table(fend, D, s)
    trace.append(dict(it=it, frac=(L['Eg'] - E_now) / (L['Eg'] - L['Efn']), H_dE_site=esite(ex.energy_H(lP + fend, s)),
                      evals=EV['total'] - ev0, sec=time.time() - t0)); del fend
    lrec['vmc_trace'] = trace; lrec['vmc_steps'] = it; lrec['vmc_capped'] = EV['total'] - ev0 < budget
    return flat


# ============================================================================================ main loop
s = sP0
L = loop_start(lP + model.table(flat, sec.reps), s)
log('LOOP START', L['info'])
if arm == 'vmc_scan':                                          # step-size scan of the VMC control (smoke)
    res['scan'] = []
    for eta in SPEC['etas']:
        SPEC['eta'] = eta; rec = dict(eta=eta); EV['total'] = 0.0; t0 = time.time()
        fl = run_vmc(model.flat0, L, 0, rec, SPEC['evals_per_loop'])
        ftab = model.table(fl, sec.reps)
        E_end, _, _ = ef_of_table(ftab, L['D'], L['s'])
        rec.update(frac=(L['Eg'] - E_end) / (L['Eg'] - L['Efn']), H_dE_site=esite(ex.energy_H(lP + ftab, L['s'])),
                   evals=EV['total'], sec=time.time() - t0)
        log('SCAN', {k: v for k, v in rec.items() if k != 'vmc_trace'})
        res['scan'].append(rec); dump(F_OUT, res)
    SPEC['n_loop'] = 0
if arm == 'fit_scan':                                          # inner-optimiser scan on the first target (smoke)
    res['scan'] = []
    for eta, lam_ in SPEC['combos']:
        SPEC['eta'] = eta; SPEC['lam'] = lam_; rec = dict(eta=eta, lam=lam_); EV['total'] = 0.0; t0 = time.time()
        fl = run_ite(model.flat0, L, 0, rec)
        ftab = model.table(fl, sec.reps)
        E_end, _, _ = ef_of_table(ftab, L['D'], L['s'])
        rec.update(frac=(L['Eg'] - E_end) / (L['Eg'] - L['Efn']), evals=EV['total'], sec=time.time() - t0)
        log('SCAN', {k: v for k, v in rec.items() if k != 'steps'}, rec['steps'][0]['val_hist'])
        res['scan'].append(rec); dump(F_OUT, res)
    SPEC['n_loop'] = 0
L0 = L
BASE = dict(SPEC)
res['arms'] = []
for sub in (BASE.get('arms') or [{}]) if BASE['arm'] in ('ite', 'vmc', 'multi') else []:
    SPEC.clear(); SPEC.update(BASE); SPEC.pop('arms', None); SPEC.update(sub)
    arm = SPEC['arm']; tag = SPEC.get('tag', arm)
    log(f'==== arm {tag}: {json.dumps(sub)}')
    ares = dict(tag=tag, spec=dict(SPEC), loops=[]); res['arms'].append(ares); res['loops'] = ares['loops']
    flat = model.flat0; s = sP0; L = L0; EV['total'] = 0.0; ta = time.time()
    for k in range(SPEC.get('n_loop', 3)):
        lrec = dict(it=k + 1, start=L['info'], evals_start=EV['total'])
        ares['loops'].append(lrec); dump(F_OUT, res)
        t0 = time.time()
        if arm == 'ite':
            flat = run_ite(flat, L, k, lrec)
        elif arm == 'vmc':
            flat = run_vmc(flat, L, k, lrec, SPEC['evals_per_loop'])
        lrec['evals_loop'] = EV['total'] - lrec['evals_start']
        lrec['train_sec'] = time.time() - t0
        end, s, la_b, la_c = loop_end(flat, L, k)
        lrec['end'] = end
        if arm == 'ite' and lrec.get('E_last_target') is not None:
            st = lrec['steps']
            lrec['cum_fitloss_site'] = lrec['cum_fitloss_site_through_step_K-1'] + (end['Ef_end'] - lrec['E_last_target']) / N
            Eb = [q['_E'] for q in st] + [end['Ef_end']]
            Et = [(q['E_t_dE_site'] + E0_SITE) * N for q in st]
            lrec['kept_per_step'] = [(Eb[i] - Eb[i + 1]) / max(Eb[i] - Et[i], 1e-300) for i in range(len(st))]
            lrec['energy_increase_events'] = int(sum(Eb[i + 1] > Eb[i] + 1e-12 * abs(Eb[i]) for i in range(len(st))))
        np.save(os.path.join(OUT, f'params_{tag}_it{k + 1}.npy'), np.asarray(flat))
        fr_ = end['frac_composite'] if SPEC.get('composite') else end['frac']
        stop = arm == 'ite' and k == 0 and fr_ < SPEC.get('stop_frac', 0.10)
        if k < SPEC.get('n_loop', 3) - 1 and not stop:
            L = loop_start(la_c if SPEC.get('composite') else la_b, s)
            lrec['next_E_FN_dE_site'] = L['info']['E_FN_dE_site']; lrec['next'] = L['info']
        elif SPEC.get('composite'):
            _, _, ic = sec.fn_solve(la_c, s); lrec['next_E_FN_dE_site'] = ic['dE_FN_site']
        else:
            _, _, ib = sec.fn_solve(la_b, s); lrec['next_E_FN_dE_site'] = ib['dE_FN_site']
        del la_b, la_c
        log(f'== {tag} loop it {k + 1}: frac {end["frac"]:.4f} (+F_k hop {end["frac_hop_Fk"]:.4f}, composite {end["frac_composite"]:.4f}) '
            f'<H>(old s) {end["H_oldsign_dE_site"]:.4e}  <H>(Krylov) {end["H_kry_dE_site"]:.4e}  <H>(composite) '
            f'{end["H_composite_dE_site"]:.4e}  next E_FN {lrec["next_E_FN_dE_site"]:.4e}  evals {lrec["evals_loop"]:.3e}')
        dump(F_OUT, res)
        if stop:
            log('  frac < stop_frac after iteration 1: arm stops (pre-registered)'); break
    ares['sec'] = time.time() - ta
    dump(F_OUT, res)
res['sec'] = time.time() - T00
dump(F_OUT, res)
log('DONE', res['sec'])
