"""Sampled 'hop-aware' objective: E_1(theta) = variational energy of g = (1 + alpha (H - E)) v, v = Re[w psi_P], alpha by Ritz in
span{v, (H-E)v}, and its gradient from x ~ |psi_P|^2 samples (the moments of vmc_moments.py plus their theta-derivatives).

Per configuration x (u = w psi_P(x)/|psi_P(x)|, E_L = (H psi_P)/psi_P, O = d log psi_P / d theta, H' = H - E_ref):
  vt  = v/|psi|            = Re u                    dvt = (dv)/|psi|        = Re[u O]
  h1  = (Hv)/|psi|         = Re[u E_L]               dh1 = (H dv)/|psi|      = Re[u (O E_L + dE_L)]   (d(Hv)(x) = (H dv)(x))
  h2  = (H^2 v)/|psi|      = diag h1 + sum_b (J_b/2) Re[u psi(y_b)/psi(x) E_L(y_b)]      (E_L at the antiparallel neighbours)
Moments (means over |psi|^2):  N0 = <vt^2> = <v|v>, N1 = <vt h1'>, N2 = <h1'^2>, N3 = <h1' h2'> = <v|H'^3|v>  (all divided by the norm sum)
Gradients (H real symmetric):  dN0 = 2<dvt vt>,  dN1 = <dvt h1' + vt dh1'>,  dN2 = 2<dh1' h1'>,  dN3 = 2<dh1' h2'>   (2<dv|H'^k v>)
E_1 = Phi(N): m_k = N_k/N0, E = E_ref + m1, mu2 = m2 - m1^2, mu3 = m3 - 3 m1 m2 + 2 m1^3, r = mu3/mu2, E_1 = E + (r - sqrt(r^2 + 4 mu2))/2;
grad E_1 = sum_k dPhi/dN_k dN_k.  alpha = (mu3 - sqrt(mu3^2 + 4 mu2^3))/(2 mu2^2).
Gradient used: the zero-variance g-form (see gform; equal to the moment chain rule 'mom' in exact mode, far less noisy).
Broken variants: 'noDE' (dE_L dropped from dh1), 'p0' (alpha = 0: gradient of E_v). 'alpha frozen' is NOT a broken variant here:
E_1 is the Ritz minimum over alpha, so dE_1/dalpha = 0 and the alpha-frozen gradient IS the gradient (the g-form uses it); the moment
chain rule ('mom', which differentiates alpha) agreeing with it in exact mode is the identity check.
Errors: jackknife over groups of chains.  Exact mode: all configurations with weights |psi|^2 / sum (wt argument).
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for d in (os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'rbmpp_6x6')): sys.path.insert(0, d)
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from jax.flatten_util import ravel_pytree
from rbmpp_lib import Lattice, make_fns, make_sampler, random_configs, to32

KEYS = ('f', 'bre', 'bim', 'Wre', 'Wim')


def phi_E1(N, E_ref):
    m1, m2, m3 = N[1] / N[0], N[2] / N[0], N[3] / N[0]
    mu2 = m2 - m1 ** 2; mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3; r = mu3 / mu2
    return E_ref + m1 + 0.5 * (r - jnp.sqrt(r * r + 4 * mu2))


def phi_E0(N, E_ref):
    return E_ref + N[1] / N[0]


def phi_afrozen(N, E_ref, al):
    m1, m2, m3 = N[1] / N[0], N[2] / N[0], N[3] / N[0]
    mu2 = m2 - m1 ** 2; mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    return E_ref + m1 + (2 * al * mu2 + al ** 2 * mu3) / (1 + al ** 2 * mu2)


def moments_info(N, E_ref):
    N = np.asarray(N, float); m1, m2, m3 = N[1] / N[0], N[2] / N[0], N[3] / N[0]
    mu2 = m2 - m1 ** 2; mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    al = (mu3 - np.sqrt(mu3 ** 2 + 4 * mu2 ** 3)) / (2 * mu2 ** 2)
    return dict(E=E_ref + m1, var=mu2, mu3=mu3, alpha=al, E1=float(phi_E1(N, E_ref)))


_dphi = {'E1': jax.jit(jax.grad(phi_E1)), 'E0': jax.jit(jax.grad(phi_E0)), 'afrozen': jax.jit(jax.grad(phi_afrozen))}


class HopEstimator:
    def __init__(self, lat, use_rbm, unravel, chunk=256, jchunk=64):
        self.lat = lat; self.N = lat.N; self.chunk = chunk; self.jchunk = jchunk; self.unravel = unravel
        self.BI = np.asarray(lat.BI); self.BJ = np.asarray(lat.BJ); self.JB = np.asarray(lat.JB, float)
        f64 = make_fns(lat, jnp.float64, use_rbm)

        def four(fl, x):
            E, lP, rP = f64['eloc_full'](unravel(fl), x)
            return jnp.stack([jnp.real(lP), jnp.imag(lP), jnp.real(E), jnp.imag(E)]), (E, lP)
        self._jac = jax.jit(jax.vmap(jax.jacrev(four, has_aux=True), in_axes=(None, 0)))
        self._fe = jax.jit(jax.vmap(lambda fl, y: f64['eloc'](unravel(fl), y), in_axes=(None, 0)))
        self.t = {}

    def _pad(self, Z, c, i):
        n = min(c, Z.shape[0] - i); blk = Z[i:i + n]
        if n < c: blk = np.concatenate([blk, np.repeat(blk[:1], c - n, 0)])
        return n, jnp.asarray(blk)

    def at_x(self, fl, X):
        """J (ns, 4, P) jacobian of (Re lP, Im lP, Re E_L, Im E_L); E_L, lP (ns,)."""
        Js, Es, ls = [], [], []
        for i in range(0, X.shape[0], self.jchunk):
            n, blk = self._pad(X, self.jchunk, i)
            J, (E, l) = self._jac(fl, blk)
            Js.append(J[:n]); Es.append(E[:n]); ls.append(l[:n])
        return jnp.concatenate(Js), jnp.concatenate(Es), jnp.concatenate(ls)

    def at_y(self, fl, Y):
        Es, ls = [], []
        for i in range(0, Y.shape[0], self.chunk):
            n, blk = self._pad(Y, self.chunk, i)
            E, l = self._fe(fl, blk); Es.append(np.asarray(E)[:n]); ls.append(np.asarray(l)[:n])
        return np.concatenate(Es), np.concatenate(ls)

    def neighbours(self, X):
        anti = X[:, self.BI] != X[:, self.BJ]
        wi, bi = np.nonzero(anti)
        Y = X[wi].copy(); r = np.arange(len(wi))
        Y[r, self.BI[bi]] = X[wi, self.BJ[bi]]; Y[r, self.BJ[bi]] = X[wi, self.BI[bi]]
        return wi, bi, Y

    def estimate(self, fl, X, w, E_ref, wt=None, groups=16, variants=(), with_O=True):
        """X (ns, N) +-1 float64. wt: per-sample weights (None = uniform MC samples). Returns dict with E1, grad (P,), jackknife errors,
        moments, O (ns, P complex, for SR) and gradients of the requested broken variants."""
        fl = jnp.asarray(fl); ns = X.shape[0]; t0 = time.time()
        J, E, l = self.at_x(fl, X); J.block_until_ready(); t1 = time.time()
        En = np.asarray(E); ln = np.asarray(l)
        wi, bi, Y = self.neighbours(X)
        Ey, ly = self.at_y(fl, Y); t2 = time.time()
        u = w * np.exp(1j * ln.imag)
        vt = u.real; h1 = (u * En).real
        ratio = np.exp(ly - ln[wi])
        Hvy = (u[wi] * ratio * Ey).real
        diag = 0.25 * np.sum(self.JB[None] * X[:, self.BI] * X[:, self.BJ], 1)
        h2 = diag * h1 + np.bincount(wi, weights=0.5 * self.JB[bi] * Hvy, minlength=ns)
        h1s = h1 - E_ref * vt; h2s = h2 - 2 * E_ref * h1 + E_ref ** 2 * vt
        q = np.stack([vt * vt, vt * h1s, h1s * h1s, h1s * h2s, vt * h2s], 1)          # last: <vt h2'> (Hermiticity check vs N2)
        wt = np.full(ns, 1.0 / ns) if wt is None else np.asarray(wt, float) / np.sum(wt)
        grp = np.arange(ns) % groups if wt is None or groups > 1 else np.zeros(ns, int)
        G = groups
        oh = np.zeros((G, ns)); oh[grp, np.arange(ns)] = wt               # weighted one-hot
        Qg = oh @ q                                                       # (G, 5) group sums of weighted q
        uj = jnp.asarray(u); Ej = jnp.asarray(En)
        O = J[:, 0, :] + 1j * J[:, 1, :]; dE = J[:, 2, :] + 1j * J[:, 3, :]
        dvt = jnp.real(uj[:, None] * O)
        dh1 = jnp.real(uj[:, None] * (O * Ej[:, None] + dE)) - E_ref * dvt
        dh1_noDE = jnp.real(uj[:, None] * (O * Ej[:, None])) - E_ref * dvt
        ohj = jnp.asarray(oh); vtj, h1j, h2j = jnp.asarray(vt), jnp.asarray(h1s), jnp.asarray(h2s)

        def vsums(dh):   # (G, 2, 3, P): <dX_i Y_j>, dX = (dvt, dh1'), Y = (vt, h1', h2')
            return jnp.stack([jnp.stack([(ohj * y) @ dx for y in (vtj, h1j, h2j)]) for dx in (dvt, dh)]).transpose(2, 0, 1, 3)
        Vg = np.asarray(vsums(dh1))
        out = dict(ns=ns, t_x=t1 - t0, t_y=t2 - t1, n_y=int(len(wi)))
        tot = Qg.sum(0)
        info = moments_info(tot[:4], E_ref); out.update(info); out['check_h2'] = float(tot[2] / tot[4])

        def gform(Q, V, al):
            """zero-variance form: grad E_1 = 2 <dg~ (H'g~ - e g~)> / <g~^2>, g~ = a0 vt + a1 h1', H'g~ = a0 h1' + a1 h2', dg~ = a0 dvt + a1 dh1',
            a0 = 1 - al m1, a1 = al (coefficients frozen: exact by Ritz stationarity); e = <g~ H'g~>/<g~^2>.  Returns (grad, E_1)."""
            m1 = Q[1] / Q[0]; a = np.array([1 - al * m1, al])
            ZZ = np.array([[Q[0], Q[1]], [Q[1], Q[2]]])                      # <Z_i Z_j>, Z = (vt, h1')
            ZY = np.array([[Q[1], Q[4]], [Q[2], Q[3]]])                      # <Z_i Y'_j>, Y' = (h1', h2')
            gg = a @ ZZ @ a; e = (a @ ZY @ a) / gg
            # sum_ij a_i a_j (<dX_i Y'_j> - e <dX_i Z_j>);  V[i, j]: j = 0 vt, 1 h1', 2 h2'
            T = sum(a[i] * a[j] * (V[i, j + 1] - e * V[i, j]) for i in range(2) for j in range(2))
            return 2 * T / gg, E_ref + e

        def mom(Q, V):   # moment chain rule (first version): dN0 = 2<dvt vt>, dN1 = <dvt h1' + vt dh1'>, dN2 = 2<dh1' h1'>, dN3 = 2<dh1' h2'>
            Gk = np.stack([2 * V[0, 0], V[0, 1] + V[1, 0], 2 * V[1, 1], 2 * V[1, 2]])
            return np.asarray(_dphi['E1'](jnp.asarray(Q[:4]), E_ref)) @ Gk

        jobs = {'E1': lambda Q, V: gform(Q, V, moments_info(Q[:4], E_ref)['alpha'])[0],
                'p0': lambda Q, V: gform(Q, V, 0.0)[0],
                'mom': mom}
        Vb = None
        if 'noDE' in variants:
            Vb = np.asarray(vsums(dh1_noDE))
            jobs['noDE'] = 'b'
        for name, fn in jobs.items():
            VV = Vg
            if fn == 'b':
                VV = Vb; fn = jobs['E1']
            g_full = fn(tot, VV.sum(0))
            if G > 1:
                Vt = VV.sum(0)
                jk = np.array([fn(tot - Qg[k], Vt - VV[k]) for k in range(G)])
                err = np.sqrt((G - 1) / G * np.sum((jk - jk.mean(0)) ** 2, 0))
            else:
                err = np.zeros_like(g_full)
            out['grad' if name == 'E1' else f'grad_{name}'] = g_full
            out['err' if name == 'E1' else f'err_{name}'] = err
        out['E1_gform'] = gform(tot, Vg.sum(0), info['alpha'])[1]
        if G > 1:
            vals = np.array([[moments_info((tot - Qg[k])[:4], E_ref)[key] for key in ('E', 'E1', 'alpha', 'var')] for k in range(G)])
            errs = np.sqrt((G - 1) / G * np.sum((vals - vals.mean(0)) ** 2, 0))
            out.update(E_err=errs[0], E1_err=errs[1], alpha_err=errs[2], var_err=errs[3])
        if with_O: out['O'] = O
        out['t_red'] = time.time() - t2
        return out


def make_sr(mask):
    maskj = jnp.asarray(mask)

    @jax.jit
    def sr_solve(O, grad, lr, shift, maxn):
        """d = -lr (S + shift diag S)^-1 F, F = grad/2 (the usual Re cov(O, E_L) = dE/2 convention), S = Re cov(O, O), diag-rescaled."""
        Om = O - jnp.mean(O, 0); ns = O.shape[0]
        S = jnp.real(Om.conj().T @ Om) / ns
        F = 0.5 * grad
        S = jnp.where(maskj[:, None] & maskj[None, :], S, 0.0); F = jnp.where(maskj, F, 0.0)
        d = jnp.maximum(jnp.diag(S), 1e-12); sq = jnp.sqrt(d)
        Ss = S / (sq[:, None] * sq[None, :]) + shift * jnp.eye(S.shape[0])
        y = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(Ss), F / sq)
        dl = jnp.where(maskj, -lr * y / sq, 0.0)
        nrm = jnp.sqrt(jnp.maximum(dl @ S @ dl, 0.0))
        dl = dl * jnp.minimum(1.0, maxn / jnp.maximum(nrm, 1e-30))
        return dl, nrm, jnp.sqrt(jnp.sum(F * F))
    return sr_solve


def class_mask(params, cls):
    """cls: 'all' | 'f' | 'rbm' (ravel_pytree order is alphabetical: Wim, Wre, bim, bre, f)."""
    sel = {'all': set(KEYS), 'f': {'f'}, 'rbm': {'bre', 'bim', 'Wre', 'Wim'}}[cls]
    return np.asarray(ravel_pytree({k: jnp.full(jnp.shape(params[k]), k in sel) for k in KEYS})[0]) > 0.5


def phase_w(est, fl, X):
    """w from the 2x2 problem in span{Re psi, Im psi} (vmc_moments.phase_from_moments) on samples X."""
    from vmc_moments import phase_from_moments
    E, l = est.at_y(jnp.asarray(fl), X)
    e2 = np.exp(2j * l.imag)
    return phase_from_moments(e2.mean(), (e2 * E).mean(), E.real.mean())
