"""Test L8 refit library: fit the RBM+PP v_theta = Re[w psi_P^theta] to the fixed-node ground state phi_0 of the guide g_0 using
walker samples only (README compress_6x6, "Test L8"; design in l8/STATUS.md).

Walkers of the p = 1 FN-GFMC with guide g_0 sample the mixed distribution pi = g_0 phi_0 (>= 0): per block b, walkers x_bi with weights
W_bi = exp(logw_bi) (pre-resampling, from gfmc_run.py --dump).  Each block is normalised (p_bi = W_bi / sum_i W_bi) and blocks are weighted
equally, as in the mixed energy estimator.  Then for any amplitude a
    E_pi(a / g_0) = sum_b sum_i p_bi a(x_bi) / g_0(x_bi) / n_blocks   estimates   <a|phi_0> / <g_0|phi_0>.
g_0(x) = Re[w psi_P0(x) (1 + alpha (E_L0(x) - E))] = exp(Re l0) cg(x),  cg = Re[w e^{i Im l0} h0],  h0 = 1 + alpha (E_L0 - E)  (SIGNED).
v_theta(x) = Re[w psi_P^theta(x)] = exp(Re l) vt(x),  vt = Re[w e^{i Im l}]  (w = g_0's phase, kept fixed in the fit).
Relative fidelity (phi_0's norm cancels):  R(theta) = F(v_theta)/F(v_0) = Q(theta)^2 * Ninv(theta),
    Q(theta)   = E_pi(v_theta/g_0) / E_pi(v_0/g_0)                       (walkers)
    Ninv(theta)= <v_0|v_0>/<v_theta|v_theta> = E_{|psi_theta|^2}[e^{2(Re l0 - Re l)} vt0^2] / E_{|psi_theta|^2}[vt^2]   (VMC of psi_theta)
R(theta_0) = 1 exactly (identical numbers in numerator and denominator).
Gradient of (1/2) log R:  G = E_w~[D] - E_{v^2}[D],  D = d log|v_theta| / d theta (real parameters f, Re/Im b, Re/Im W),
    w~_i = p_i r_i / sum p r,  r = v_theta / g_0  (signed, self-normalised; computed as one reverse pass of sum_i w~_i log|v(x_i)|),
SR (natural gradient): S = (Cov_{v^2}(D) + Cov_walkers(D))/2 (see sr_step);  delta = lr (S + shift diag S)^{-1} G (diagonally rescaled Cholesky as in
rbmpp_train.sr_solve), capped at sqrt(delta S delta) <= maxn.  Ascent on R.
Broken variant for the controls: sign_broken=True uses |g_0| in place of g_0 (weights a/|g_0|).
"""
import time, glob, os, json
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
jax.config.update('jax_enable_x64', True)
from rbmpp_lib import make_fns, make_sampler, random_configs, to32

KEYS = ('f', 'bre', 'bim', 'Wre', 'Wim')


class Model:
    def __init__(self, lat, params0, use_rbm=True, chunk=256):
        self.lat = lat; self.N = lat.N; self.chunk = chunk; self.use_rbm = use_rbm
        p0 = {k: jnp.asarray(np.asarray(params0[k])) for k in KEYS}
        self.flat0, self.unravel = ravel_pytree(p0)
        self.P = int(self.flat0.size)
        f64 = make_fns(lat, jnp.float64, use_rbm)
        self.sweep = make_sampler(lat, make_fns(lat, jnp.float32, use_rbm))
        lp, el = f64['logpsi'], f64['eloc']
        un = self.unravel
        self._lp = jax.jit(jax.vmap(lambda fl, s: lp(un(fl), s), in_axes=(None, 0)))
        self._el = jax.jit(jax.vmap(lambda fl, s: el(un(fl), s), in_axes=(None, 0)))

        def logabsv(fl, s, w):
            l = lp(un(fl), s)
            return jnp.real(l) + jnp.log(jnp.abs(jnp.real(w * jnp.exp(1j * jnp.imag(l)))))
        self._jac = jax.jit(jax.vmap(jax.grad(logabsv), in_axes=(None, 0, None)))
        self._wgrad = jax.jit(jax.grad(lambda fl, S, c, w: jnp.sum(c * jax.vmap(lambda s: logabsv(fl, s, w))(S))))

    def params(self, flat):
        q = self.unravel(jnp.asarray(flat))
        return {k: np.asarray(q[k]) for k in KEYS}

    def _pad(self, Z, i):
        c = self.chunk; blk = Z[i:i + c]; n = blk.shape[0]
        if n < c:
            blk = np.concatenate([blk, np.repeat(blk[:1], c - n, 0)])
        return blk, n

    def logpsi(self, flat, Z):
        out = np.empty(Z.shape[0], np.complex128)
        for i in range(0, Z.shape[0], self.chunk):
            blk, n = self._pad(Z, i); out[i:i + n] = np.asarray(self._lp(flat, jnp.asarray(blk)))[:n]
        return out

    def eloc(self, flat, Z):
        E = np.empty(Z.shape[0], np.complex128); l = np.empty(Z.shape[0], np.complex128)
        for i in range(0, Z.shape[0], self.chunk):
            blk, n = self._pad(Z, i); e_, l_ = self._el(flat, jnp.asarray(blk)); E[i:i + n] = np.asarray(e_)[:n]; l[i:i + n] = np.asarray(l_)[:n]
        return E, l

    def jac(self, flat, Z, w):
        out = np.empty((Z.shape[0], self.P))
        for i in range(0, Z.shape[0], self.chunk):
            blk, n = self._pad(Z, i); out[i:i + n] = np.asarray(self._jac(flat, jnp.asarray(blk), w))[:n]
        return out

    def wgrad(self, flat, Z, c, w):
        g = np.zeros(self.P)
        for i in range(0, Z.shape[0], self.chunk):
            blk, n = self._pad(Z, i); cc = np.zeros(blk.shape[0]); cc[:n] = c[i:i + n]
            g += np.asarray(self._wgrad(flat, jnp.asarray(blk), jnp.asarray(cc), w))
        return g


# ----------------------------------------------------------------------------------------------- walkers
def load_dumps(dirs, holdout=0.2, first_block=None):
    """dirs: run dirs with dump/blk_*.npz (one per chain).  Returns dict of arrays over all walkers: X (n, N) float64, pw (n,) (block-normalised
    weights / blocks in the same set), chain, block, held (bool: last `holdout` fraction of each chain's blocks)."""
    Xs, pws, ch, bl, ho, EL = [], [], [], [], [], []
    for c, d in enumerate(dirs):
        files = sorted(glob.glob(os.path.join(d, 'dump', 'blk_*.npz')))
        if first_block is not None:
            files = [f for f in files if int(os.path.basename(f)[4:9]) >= first_block]
        if not files:
            raise FileNotFoundError(f'no dump files in {d}/dump')
        nb = len(files); nho = int(round(holdout * nb)); cut = nb - nho
        for k, f in enumerate(files):
            z = np.load(f)
            lw = z['logw']; w = np.exp(lw - lw.max()); p = w / w.sum()
            Xs.append(z['X'].astype(np.float64)); pws.append(p); EL.append(z['EL'])
            ch.append(np.full(len(p), c)); bl.append(np.full(len(p), int(z['block']))); ho.append(np.full(len(p), k >= cut))
    out = dict(X=np.concatenate(Xs), p=np.concatenate(pws), chain=np.concatenate(ch), block=np.concatenate(bl), held=np.concatenate(ho),
               EL=np.concatenate(EL))
    return out


def guide_at(model, flat0, X, alpha, E, w):
    """Re l0, cg (signed g_0 / e^{Re l0}), vt0 (v_0 / e^{Re l0}) at walkers X."""
    El, l = model.eloc(flat0, X)
    h = 1.0 + alpha * (El - E)
    u = w * np.exp(1j * l.imag)
    return dict(lr0=l.real, cg=(u * h).real, vt0=u.real, h=h.real, El=El.real)


class WalkerSet:
    """subset of walkers (train or held-out): per-bin sums for the jackknife (bin = chain x `binb` consecutive blocks)."""

    def __init__(self, D, G, mask, w, binb=2, sign_broken=False):
        self.w = complex(w)
        self.X = D['X'][mask]; blocks = D['block'][mask]; chain = D['chain'][mask]
        self.nblocks = sum(len(np.unique(blocks[chain == c])) for c in np.unique(chain))
        self.pw = D['p'][mask] / self.nblocks
        self.lr0 = G['lr0'][mask]; cg = G['cg'][mask]
        self.cg = np.abs(cg) if sign_broken else cg
        self.vt0 = G['vt0'][mask]
        lab = np.zeros(len(blocks), np.int64)
        for c in np.unique(chain):
            m = chain == c; _, j = np.unique(blocks[m], return_inverse=True); lab[m] = c * 10 ** 6 + j // binb
        _, self.bin = np.unique(lab, return_inverse=True)
        self.nbin = int(self.bin.max()) + 1
        self.r0 = self.vt0 / self.cg
        self.B = np.bincount(self.bin, weights=self.pw * self.r0, minlength=self.nbin)

    def ratios(self, l):
        """r = v_theta / g_0 at the walkers (common factor e^{-Re l0} cancels in every ratio) from log psi_theta."""
        return np.exp(l.real - self.lr0) * (self.w * np.exp(1j * l.imag)).real / self.cg

    def Q(self, r):
        A = np.bincount(self.bin, weights=self.pw * r, minlength=self.nbin)
        q = A.sum() / self.B.sum()
        jk = (A.sum() - A) / (self.B.sum() - self.B)
        err = float(np.sqrt((self.nbin - 1) / self.nbin * np.sum((jk - jk.mean()) ** 2)))
        return float(q), err

    def health(self, r):
        """self-normalised walker weights x v/g: n_eff/n and the largest single weight's share (absolute values)."""
        c = np.abs(self.pw * r); s = c.sum()
        return dict(n=int(c.size), neff_over_n=float(s * s / np.sum(c * c) / c.size), max_share=float(c.max() / s),
                    neg_frac_weight=float(np.sum(np.abs(self.pw * r)[self.pw * r < 0]) / s),
                    plain_neff_over_n=float(self.pw.sum() ** 2 / np.sum(self.pw ** 2) / c.size))


# ----------------------------------------------------------------------------------------------- VMC (norm ratio, metric)
class VMC:
    def __init__(self, model, C, seed, nsweep=None, therm=50, groups=16):
        self.m = model; self.C = C; self.G = groups; self.nsweep = nsweep or model.N
        self.key = jax.random.PRNGKey(4242 + seed); self.S = random_configs(model.N, C, 777 + seed); self.therm = therm
        self.started = False

    def sample(self, flat):
        p32 = to32(self.m.unravel(jnp.asarray(flat)))
        n = (self.therm if not self.started else 0) + 1; self.started = True
        for _ in range(n):
            self.key, k = jax.random.split(self.key); self.S, acc = self.m.sweep(p32, self.S, k, self.nsweep)
        self.acc = float(acc)
        return np.asarray(self.S).astype(np.float64)


def ninv(model, flat, flat0, S, w, groups=16):
    """Ninv = <v0|v0>/<v|v> from samples S of |psi_theta|^2, jackknife over chain groups; also returns l and vt at S."""
    l = model.logpsi(flat, S)
    if np.array_equal(np.asarray(flat), np.asarray(flat0)):
        l0 = l
    else:
        l0 = model.logpsi(flat0, S)
    vt = (w * np.exp(1j * l.imag)).real; vt0 = (w * np.exp(1j * l0.imag)).real
    num = np.exp(2 * (l0.real - l.real)) * vt0 ** 2; den = vt ** 2
    g = np.arange(len(S)) % groups
    a = np.bincount(g, weights=num, minlength=groups); b = np.bincount(g, weights=den, minlength=groups)
    val = a.sum() / b.sum(); jk = (a.sum() - a) / (b.sum() - b)
    err = float(np.sqrt((groups - 1) / groups * np.sum((jk - jk.mean()) ** 2)))
    return float(val), err, l, vt


def logabsv(model, flat, Z, w):
    l = model.logpsi(flat, Z)
    return l.real + np.log(np.abs((w * np.exp(1j * l.imag)).real))


def sr_step(model, flat, S, vt, wset_tr, r_tr, w, lr, shift, maxn, rng, nsub=None, dfloor=1e-2, max_halve=8):
    """One SR ascent step on R.  Metric S = (S_v + S_w)/2: S_v = Cov_{v^2}(D) from the VMC samples, S_w = Cov(D) over nsub training walkers
    resampled by |w~| (the gradient's walker term lives on the walkers' configurations; a metric from the VMC samples alone has rank <= #samples
    and left those directions uncontrolled: 4x4 and 8x8 steps blew R up by orders of magnitude, 2026-10-25).  Diagonal floor dfloor x median
    diag (parameters that barely move log|v| on typical samples, e.g. Im W for a nearly real psi).  Trust region on function values: the
    realised change of log|v| (centred RMS over VMC samples + resampled walkers) must stay <= 2 maxn, else the step is halved."""
    import scipy.linalg as sl
    D = model.jac(flat, S, w)
    wt = vt ** 2; wt = wt / wt.sum()
    Dbar = wt @ D
    Sv = ((D - Dbar) * wt[:, None]).T @ (D - Dbar)
    c = wset_tr.pw * r_tr; c = c / c.sum()
    gw = model.wgrad(flat, wset_tr.X, c, w)
    G = gw - Dbar
    nsub = nsub or S.shape[0]
    ac = np.abs(c); idx = rng.choice(len(c), nsub, p=ac / ac.sum())
    Xw = wset_tr.X[idx]; Dw = model.jac(flat, Xw, w); Dwm = Dw - Dw.mean(0)
    Sw = Dwm.T @ Dwm / nsub
    Sm = 0.5 * (Sv + Sw)
    d = np.diag(Sm).copy(); pos = d[d > 0]
    d = np.maximum(d, dfloor * (np.median(pos) if pos.size else 1.0)); sq = np.sqrt(d)
    Ss = Sm / (sq[:, None] * sq[None, :]) + shift * np.eye(len(d))
    y = sl.cho_solve(sl.cho_factor(Ss), G / sq)
    dl = lr * y / sq
    nrm = float(np.sqrt(max(dl @ Sm @ dl, 0.0)))
    if nrm > maxn: dl *= maxn / nrm
    Z = np.concatenate([S, Xw]); l0 = logabsv(model, flat, Z, w)
    nh = 0
    for nh in range(max_halve + 1):
        dlz = logabsv(model, flat + dl, Z, w) - l0
        rms = float(np.sqrt(np.mean((dlz - dlz.mean()) ** 2)))
        if rms <= 2 * maxn: break
        dl = 0.5 * dl
    return dl, dict(dnorm=nrm, rms_dlogv=rms, halvings=nh, gnorm=float(np.sqrt(G @ G)), gS=float(np.sqrt(max(G / sq @ (G / sq), 0.0))))


def save_params(path, model, flat, src_npz):
    z = np.load(src_npz); q = model.params(flat)
    extra = {k: z[k] for k in z.files if k not in KEYS}
    np.savez(path, **q, **extra)
