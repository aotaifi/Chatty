"""Exact full-sum variational optimiser on a small J1-J2 lattice (PLAN_ansatz.md, "Shared tool + C2-T2 + C5-T0").

  python exact_vmc.py check  OUTDIR [LX LY]          checks + timing (gradient vs finite differences, scipy energy, J2=0 Jastrow)
  python exact_vmc.py run    OUTDIR SPEC.json        train the arms of a spec (list of runs), one JSON per run
  python exact_vmc.py verdict OUTDIR                 C2-T2 / C5-T0 verdict lines from the run JSONs in OUTDIR

Basis: Sz=0 configurations of LX x LY (PBC), lattice/bonds/Marshall sign as ansatz/code/c2_t1_small.py.
H u = sum_b J_b (zz_b u + 0.5 anti_b u[nbr_b]) as per-bond gathers on the device (custom VJP: H real symmetric).
Every ansatz amplitude is evaluated on ALL configurations each step (no sampling); E = Re<psi|H psi>/<psi|psi>,
gradients by autodiff, optimiser optax.lbfgs (zoom line search) on the deterministic objective.
Bases f_theta (log amplitude, positive part):
  J   : Jastrow, f = sum_d v_d sum_i s_i s_{i+d}, d over translation displacement classes (d ~ -d), no reflections.
  R<a>: translation-symmetric real RBM, f = sum_k sum_t logcosh(b_k + sum_i W_k[i - t] s_i), k = 1..a.
Forms: plain s_N e^f; layered (1 + a K) s_N e^f with K = H or H_J2 (a trained with theta);
  C5 standard s_N e^{f + i pi sigmoid(g)}; C5 coupled s_N e^f tanh(g); g = c0 + sum_k c_k sum_t logcosh(...)
  (same RBM size as f plus a+1 output weights; without signed output weights g >= 0 and neither form could flip a sign).
"""
import os, sys, json, time, math, gc
import numpy as np

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import lax
import optax
from jax.flatten_util import ravel_pytree
import scipy.sparse.linalg as sla

T0 = time.time()
LOGF = None
def log(*a):
    s = f'[{time.time() - T0:8.1f}s] ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    if LOGF is not None:
        with open(LOGF, 'a') as fh: fh.write(s + '\n')

E0_REF_4x6 = -0.52252493537

# ------------------------------------------------------------------ lattice (as c2_t1_small.py)
class Lattice:
    def __init__(self, LX, LY, J2=0.5):
        self.LX, self.LY, self.N, self.J2 = LX, LY, LX * LY, J2
        N = self.N; site = self.site
        self.NN = [(site(x, y), site(x + 1, y)) for y in range(LY) for x in range(LX)] + \
                  [(site(x, y), site(x, y + 1)) for y in range(LY) for x in range(LX)]
        self.NNN = [(site(x, y), site(x + 1, y + 1)) for y in range(LY) for x in range(LX)] + \
                   [(site(x, y), site(x + 1, y - 1)) for y in range(LY) for x in range(LX)]
        allx = np.arange(1 << N, dtype=np.int64); pc = np.zeros(1 << N, np.int8)
        for i in range(N): pc += ((allx >> i) & 1).astype(np.int8)
        self.basis = allx[pc == N // 2]; del allx, pc
        self.D = D = self.basis.size
        MA = sum(1 << site(x, y) for y in range(LY) for x in range(LX) if (x + y) % 2 == 0)
        cnt = np.zeros(D, np.int64)
        for i in range(N):
            if (MA >> i) & 1: cnt += (self.basis >> i) & 1
        self.sN = (1 - 2 * (cnt & 1)).astype(np.float64)
        self.sigma = np.stack([2.0 * ((self.basis >> i) & 1) - 1.0 for i in range(N)], axis=1)   # (D, N) +-1
        # translations: TIDX[t, i] = site index of (x_i - tx, y_i - ty)
        self.TIDX = np.array([[site(i % LX - tx, i // LX - ty) for i in range(N)]
                              for ty in range(LY) for tx in range(LX)], dtype=np.int64)
        # displacement classes d ~ -d (d != 0)
        cls, seen = [], set()
        for dy in range(LY):
            for dx in range(LX):
                if (dx, dy) == (0, 0) or (dx, dy) in seen: continue
                seen.add((dx, dy)); seen.add(((-dx) % LX, (-dy) % LY)); cls.append((dx, dy))
        self.dclasses = cls

    def site(self, x, y): return (x % self.LX) + self.LX * (y % self.LY)

    def translation_perms(self):
        """P[t, x] = basis index of the configuration translated by t (all LX*LY translations, a group)."""
        P = np.empty((self.TIDX.shape[0], self.D), np.int32)
        for t in range(self.TIDX.shape[0]):
            y = np.zeros(self.D, np.int64)
            for i in range(self.N): y |= ((self.basis >> i) & 1) << int(self.TIDX[t, i])
            k = np.searchsorted(self.basis, y); assert np.all(self.basis[k] == y)
            P[t] = k
        return P

    def bond_data(self, i, j):
        b = self.basis
        anti = ((b >> i) & 1) != ((b >> j) & 1)
        nbr = np.minimum(np.searchsorted(b, b ^ ((1 << i) | (1 << j))), self.D - 1)   # parallel bonds masked by anti
        return anti, nbr

    def jastrow_features(self):
        s = self.sigma; F = np.zeros((self.D, len(self.dclasses)))
        for c, (dx, dy) in enumerate(self.dclasses):
            for y in range(self.LY):
                for x in range(self.LX):
                    F[:, c] += s[:, self.site(x, y)] * s[:, self.site(x + dx, y + dy)]
        return F

# ------------------------------------------------------------------ device Hamiltonian
# All large device arrays travel as explicit jit arguments (a dict "c"), never as closed-over constants
# (closed-over arrays are embedded into every compiled executable: 2.1 GB per executable at 4x6).
F0 = jax.dtypes.float0
def _f0(x): return np.zeros(x.shape, dtype=F0)

def _off(u, nb, an, J):
    def body(acc, xs):
        n, a = xs
        return acc + jnp.where(a, 0.5 * J, 0.0) * u[n], None
    return lax.scan(body, jnp.zeros_like(u), (nb, an))[0]

class Ops:
    """per-bond neighbour index arrays (int32) and antiparallel masks (bool) on the device; diagonals in f64.
    Matvecs H(u, c), HJ2(u, c), HJ1(u, c) with c = ops.consts() (or a dict that contains it)."""
    def __init__(self, lat, J1=1.0, J2=None):
        J2 = lat.J2 if J2 is None else J2
        self.D, self.J1, self.J2 = lat.D, float(J1), float(J2)
        def pack(bonds, J):
            nb = np.empty((len(bonds), lat.D), np.int32); an = np.empty((len(bonds), lat.D), bool); dg = np.zeros(lat.D)
            for k, (i, j) in enumerate(bonds):
                anti, nbr = lat.bond_data(i, j); nb[k] = nbr; an[k] = anti; dg += J * np.where(anti, -0.25, 0.25)
            return jnp.asarray(nb), jnp.asarray(an), jnp.asarray(dg)
        nb1, an1, dg1 = pack(lat.NN, J1)
        if J2 != 0.0: nb2, an2, dg2 = pack(lat.NNN, J2)
        else: nb2, an2, dg2 = jnp.zeros((0, lat.D), jnp.int32), jnp.zeros((0, lat.D), bool), jnp.zeros(lat.D)
        self.c = dict(nb1=nb1, an1=an1, dg1=dg1, nb2=nb2, an2=an2, dg2=dg2)
        J1_, J2_ = self.J1, self.J2

        @jax.custom_vjp
        def hfull(u, nb1, an1, dg1, nb2, an2, dg2):
            return (dg1 + dg2) * u + _off(u, nb1, an1, J1_) + _off(u, nb2, an2, J2_)
        def hf_fwd(u, *r): return hfull(u, *r), r
        def hf_bwd(r, g):            # H real symmetric: VJP of u -> H u is g -> H g
            return (hfull(g, *r), _f0(r[0]), _f0(r[1]), jnp.zeros_like(r[2]), _f0(r[3]), _f0(r[4]), jnp.zeros_like(r[5]))
        hfull.defvjp(hf_fwd, hf_bwd)

        def mk1(J):
            @jax.custom_vjp
            def h1(u, nb, an, dg): return dg * u + _off(u, nb, an, J)
            h1.defvjp(lambda u, *r: (h1(u, *r), r),
                      lambda r, g: (h1(g, *r), _f0(r[0]), _f0(r[1]), jnp.zeros_like(r[2])))
            return h1
        h2, h1 = mk1(J2_), mk1(J1_)
        self.H = lambda u, c: hfull(u, c['nb1'], c['an1'], c['dg1'], c['nb2'], c['an2'], c['dg2'])
        self.HJ2 = lambda u, c: h2(u, c['nb2'], c['an2'], c['dg2'])
        self.HJ1 = lambda u, c: h1(u, c['nb1'], c['an1'], c['dg1'])
        self._Hjit = jax.jit(self.H)

    def consts(self): return dict(self.c)
    def Hnp(self, v): return np.asarray(self._Hjit(jnp.asarray(v), self.c))

# ------------------------------------------------------------------ ansatz
@jax.custom_vjp
def symsum(h, P):
    """S(h)(x) = sum_t h(P[t, x]); the translations form a group, so S is self-adjoint (VJP = S)."""
    return lax.fori_loop(0, P.shape[0], lambda t, acc: acc + h[P[t]], jnp.zeros_like(h))
symsum.defvjp(lambda h, P: (symsum(h, P), P), lambda P, g: (symsum(g, P), _f0(P)))

def logcosh(x):
    ax = jnp.abs(x)
    return ax + jnp.log1p(jnp.exp(-2.0 * ax)) - math.log(2.0)

class Model:
    """builds psi(params, c) on all configurations for one arm; c = model.consts() holds every large array."""
    def __init__(self, lat, ops, base, form, alpha=None, layer=None):
        self.lat, self.ops, self.base, self.form, self.alpha, self.layer = lat, ops, base, form, alpha, layer
        if not hasattr(lat, 'dev'): lat.dev = {}          # device copies shared by all models of this lattice
        dv = lat.dev
        if 'sN' not in dv: dv['sN'] = jnp.asarray(lat.sN)
        c = ops.consts(); c['sN'] = dv['sN']
        if base == 'J':
            if 'F' not in dv: dv['F'] = jnp.asarray(lat.jastrow_features())
            c['F'] = dv['F']
        else:
            if 'sigma' not in dv: dv['sigma'] = jnp.asarray(lat.sigma); dv['P'] = jnp.asarray(lat.translation_perms())
            c['sigma'] = dv['sigma']; c['P'] = dv['P']
        self.c = c; self._step = None
        self._psi_jit = jax.jit(self.psi)
        self._E_jit = jax.jit(lambda p, c: energy_of(self.psi(p, c), lambda u: ops.H(u, c)))

    def consts(self): return self.c

    # RBM sum_t sum_k logcosh(b_k + sum_i W_k[i - t] s_i) = sum_t h(T_t x), h(x) = sum_k logcosh(b_k + W_k . s(x)):
    # h is evaluated once per configuration, the translation sum is a sum of gathers over the translation permutations.
    def _rbm_feats(self, W, b, c):
        """per-configuration per-feature sums  S_k(x) = sum_t logcosh(b_k + W_k . s(T_t x)),  shape (D, a)."""
        h = logcosh(c['sigma'] @ W.T + b)                    # (D, a)
        return jax.vmap(symsum, in_axes=(1, None), out_axes=1)(h, c['P'])

    def f_of(self, p, c):
        if self.base == 'J': return c['F'] @ p['v']
        return self._rbm_feats(p['W'], p['b'], c).sum(axis=1)

    def g_of(self, p, c):
        return p['c0'] + self._rbm_feats(p['Wg'], p['bg'], c) @ p['cg']

    def init(self, seed, a0=0.0):
        rng = np.random.default_rng(seed); N = self.lat.N; p = {}
        if self.base == 'J':
            p['v'] = 0.01 * rng.standard_normal(len(self.lat.dclasses))
        else:
            al = self.alpha
            p['W'] = 0.05 * rng.standard_normal((al, N)); p['b'] = 0.01 * rng.standard_normal(al)
        if self.form in ('c5std', 'c5cpl'):
            al = self.alpha
            p['Wg'] = 0.05 * rng.standard_normal((al, N)); p['bg'] = 0.01 * rng.standard_normal(al)
            p['cg'] = 0.1 * rng.standard_normal(al)
            p['c0'] = np.array(0.0 if self.form == 'c5std' else 1.0)   # both start near the Marshall state
        if self.layer is not None:
            p['a'] = np.array(float(a0))
        return {k: jnp.asarray(v, dtype=jnp.float64) for k, v in p.items()}

    def psi(self, p, c):
        f = self.f_of(p, c); f = f - lax.stop_gradient(jnp.max(f))
        amp = c['sN'] * jnp.exp(f)
        if self.form == 'c5std':
            return amp * jnp.exp(1j * jnp.pi * jax.nn.sigmoid(self.g_of(p, c)))
        if self.form == 'c5cpl':
            return amp * jnp.tanh(self.g_of(p, c))
        if self.layer == 'H': return amp + p['a'] * self.ops.H(amp, c)
        if self.layer == 'HJ2': return amp + p['a'] * self.ops.HJ2(amp, c)
        return amp

    def psi_np(self, p): return np.asarray(self._psi_jit(p, self.c))
    def energy(self, p): return float(self._E_jit(p, self.c))

def energy_of(psi, Hmv):
    Hp = Hmv(psi)
    return jnp.real(jnp.vdot(psi, Hp)) / jnp.real(jnp.vdot(psi, psi))

# ------------------------------------------------------------------ training
def train(model, p0, max_steps=3000, tol_site=1e-10, window=50, progress_every=100, tag=''):
    N = model.lat.N; H = model.ops.H
    flat0, unravel = ravel_pytree(p0)
    opt = optax.lbfgs()
    if model._step is None or model._step[0] != flat0.size:
        def step(x, st, c):
            Efun = lambda y: energy_of(model.psi(unravel(y), c), lambda u: H(u, c))
            val, g = optax.value_and_grad_from_state(Efun)(x, state=st)
            upd, st = opt.update(g, st, x, value=val, grad=g, value_fn=Efun)
            return optax.apply_updates(x, upd), st, val, jnp.linalg.norm(g)
        model._step = (flat0.size, jax.jit(step))
    step = model._step[1]; c = model.c
    x, st = flat0, opt.init(flat0)
    curve, t0, conv, gn = [], time.time(), False, float('nan')
    for k in range(max_steps):
        x_new, st_new, val, gn = step(x, st, c)
        val = float(val)
        if not np.isfinite(val):
            log(f'PROGRESS {tag} step {k} non-finite energy; stop'); break
        curve.append(val); x, st = x_new, st_new
        if k >= window and abs(curve[-1] - curve[-1 - window]) / N < tol_site:
            conv = True; break
        if progress_every and k % progress_every == 0:
            log(f'PROGRESS {tag} step {k} E/N {val / N:.12f} |g| {float(gn):.3e}')
    pf = unravel(x)
    Ef = model.energy(pf); curve.append(Ef)
    return dict(params=pf, flat=np.asarray(x), curve=curve, steps=len(curve) - 1, converged=conv,
                E=Ef, sec=time.time() - t0, gnorm=float(gn), n_params=int(flat0.size))

# ------------------------------------------------------------------ diagnostics
def lanczos_ground(matvec_np, D, tol=1e-12, v0=None):
    op = sla.LinearOperator((D, D), matvec=lambda v: matvec_np(np.ravel(v)), dtype=np.float64)
    E, V = sla.eigsh(op, k=1, which='SA', tol=tol, v0=v0)
    return float(E[0]), V[:, 0]

def real_sign_frame(psi):
    """real representative of a (possibly complex) state: rotate by the global phase that maximises sum Re^2."""
    psi = np.asarray(psi)
    if not np.iscomplexobj(psi): return psi
    th = 0.5 * np.angle(np.sum(psi ** 2))
    return np.real(np.exp(-1j * th) * psi)

def WM(psi, sN):
    v = real_sign_frame(psi); w = np.abs(np.asarray(psi)) ** 2; w = w / w.sum()
    glob = np.sign(np.sum(v * np.abs(v) * sN)) or 1.0
    return float(np.sum(w * ((np.sign(v) * glob) != sN)))

RCAP = 1e12      # cap on the guide ratio g(y)/g(x) in the FN sign-flip term (diag ~1e12 => that amplitude ~1e-12; avoids overflow)

def _fn_viol_diag(s, g, nb, an, J):
    def body(acc, xs):
        n, a = xs; viol = a & ((s * s[n]) > 0)
        return acc + jnp.where(viol, 0.5 * J * jnp.minimum(g[n] / g, RCAP), 0.0), None
    return lax.scan(body, jnp.zeros_like(g), (nb, an))[0]

def _fn_off(u, s, nb, an, J):
    def body(acc, xs):
        n, a = xs; keep = a & ((s * s[n]) < 0)
        return acc + jnp.where(keep, 0.5 * J * u[n], 0.0), None
    return lax.scan(body, jnp.zeros_like(u), (nb, an))[0]

def ground_lobpcg(matvec_np, diag, x0, tol=1e-7, maxiter=4000):
    """lowest eigenpair of a symmetric operator by LOBPCG with a Jacobi preconditioner 1/(|diag - e| + 1):
    robust when the diagonal has huge entries (FN sign-flip terms with a tiny guide), where plain Lanczos stalls."""
    D = diag.size; x0 = x0 / np.linalg.norm(x0); _mv = matvec_np; matvec_np = lambda x: _mv(np.ravel(x))
    e = float(x0 @ matvec_np(x0)); pre = 1.0 / (np.abs(diag - e) + 1.0)
    mm = lambda X: np.stack([matvec_np(X[:, k]) for k in range(X.shape[1])], axis=1) if X.ndim == 2 else matvec_np(X)
    A = sla.LinearOperator((D, D), matvec=matvec_np, matmat=mm, dtype=np.float64)
    M = sla.LinearOperator((D, D), matvec=lambda r: pre * np.ravel(r),
                           matmat=lambda R: pre[:, None] * R, dtype=np.float64)
    w, V, rh = sla.lobpcg(A, x0[:, None], M=M, largest=False, tol=tol, maxiter=maxiter, retResidualNormsHistory=True)
    v = V[:, 0]; r = np.linalg.norm(matvec_np(v) - w[0] * v)
    return float(w[0]), v, dict(iters=len(rh), resid=float(r))

def fn_energy(ops, psi, tol=1e-7, info=None):
    """lattice FN (ten Haaf), sign of the (real-frame) state, guide |psi|; per-bond matvec on the device; LOBPCG."""
    v = real_sign_frame(psi); c = ops.c; J1, J2 = ops.J1, ops.J2
    s = jnp.asarray(np.where(v >= 0, 1.0, -1.0)); g = jnp.asarray(np.maximum(np.abs(np.asarray(psi)), 1e-300))
    dF = jax.jit(lambda s, g, c: c['dg1'] + c['dg2'] + _fn_viol_diag(s, g, c['nb1'], c['an1'], J1)
                 + _fn_viol_diag(s, g, c['nb2'], c['an2'], J2))(s, g, c)
    mvF = jax.jit(lambda u, s, dF, c: dF * u + _fn_off(u, s, c['nb1'], c['an1'], J1) + _fn_off(u, s, c['nb2'], c['an2'], J2))
    x0 = np.asarray(g) * np.asarray(s)
    E, _, inf = ground_lobpcg(lambda x: np.asarray(mvF(jnp.asarray(x), s, dF, c)), np.asarray(dF), x0, tol=tol)
    if info is not None: info.update(inf)
    return E

def diagnostics(model, params, psi0, E0):
    N = model.lat.N
    psi = model.psi_np(params)
    E = model.energy(params)
    nrm = np.vdot(psi, psi).real
    ov2 = abs(np.vdot(psi0, psi)) ** 2 / nrm
    efn = fn_energy(model.ops, psi)
    return dict(dE_var_site=(E - E0) / N, E_var_site=E / N, infidelity=float(1 - ov2), W_M=WM(psi, model.lat.sN),
                dE_FN_site=(efn - E0) / N, E_FN_site=efn / N)

# ------------------------------------------------------------------ setup helpers
def setup(LX, LY, J2=0.5, outdir=None):
    lat = Lattice(LX, LY, J2); log('lattice', LX, LY, 'D', lat.D, 'J2', J2, 'classes', len(lat.dclasses))
    ops = Ops(lat); log('ops on device', jax.devices()[0])
    p0f = os.path.join(outdir, f'psi0_{LX}x{LY}_J2{J2}.npy') if outdir else None
    if p0f and os.path.exists(p0f) and os.path.exists(p0f + '.E'):
        psi0 = np.load(p0f); E0 = float(open(p0f + '.E').read())
    else:
        E0, psi0 = lanczos_ground(ops.Hnp, lat.D)
        if np.sum(psi0 * np.abs(psi0) * lat.sN) < 0: psi0 = -psi0
        if p0f: np.save(p0f, psi0); open(p0f + '.E', 'w').write(repr(E0))
    log('E0/N', repr(E0 / lat.N))
    return lat, ops, psi0, E0

def arm_model(lat, ops, arm):
    """arm names: J, J_H, J_HJ2, R2, R2_H, R2_HJ2, C5std_R1, C5cpl_R4, ..."""
    if arm.startswith('C5'):
        form = 'c5std' if arm.startswith('C5std') else 'c5cpl'
        al = int(arm.split('_R')[1]); return Model(lat, ops, 'R', form, alpha=al)
    parts = arm.split('_'); b = parts[0]; layer = parts[1] if len(parts) > 1 else None
    if b == 'J': return Model(lat, ops, 'J', 'plain', layer=layer)
    return Model(lat, ops, 'R', 'plain', alpha=int(b[1:]), layer=layer)

# ------------------------------------------------------------------ check mode
def run_checks(outdir, LX=4, LY=6):
    import scipy.sparse as sp
    res = {}
    lat, ops, psi0, E0 = setup(LX, LY, 0.5, outdir); N, D = lat.N, lat.D
    res['E0_site'] = E0 / N
    if (LX, LY) == (4, 6):
        res['E0_check'] = dict(ref=E0_REF_4x6, diff=E0 / N - E0_REF_4x6, ok=abs(E0 / N - E0_REF_4x6) < 1e-9)
        log('CHECK E0/N', res['E0_check'])
    # (0) translation-permutation RBM equals the direct formula sum_t sum_k logcosh(b_k + sum_i W_k[i - t] s_i)
    m = arm_model(lat, ops, 'R2'); p = m.init(11); p['b'] = p['b'] + 0.3
    f_dev = np.asarray(jax.jit(m.f_of)(p, m.c)); W, b = np.asarray(p['W']), np.asarray(p['b'])
    rows = np.random.default_rng(2).choice(D, 2000, replace=False); sg = lat.sigma[rows]; f_dir = np.zeros(rows.size)
    for ty in range(LY):
        for tx in range(LX):
            for k in range(W.shape[0]):
                Wt = np.array([W[k, lat.site(i % LX - tx, i // LX - ty)] for i in range(N)])   # W_k[i - t]
                th = sg @ Wt + b[k]; f_dir += np.abs(th) + np.log1p(np.exp(-2 * np.abs(th))) - np.log(2)
    Wbad = W.copy(); Wbad[:, 1] += 0.2; f_bad = np.zeros(rows.size)
    for ty in range(LY):
        for tx in range(LX):
            for k in range(W.shape[0]):
                Wt = np.array([Wbad[k, lat.site(i % LX - tx, i // LX - ty)] for i in range(N)])
                th = sg @ Wt + b[k]; f_bad += np.abs(th) + np.log1p(np.exp(-2 * np.abs(th))) - np.log(2)
    err = float(np.max(np.abs(f_dev[rows] - f_dir))); errb = float(np.max(np.abs(f_dev[rows] - f_bad)))
    res['rbm_direct'] = dict(max_abs_err=err, max_abs_err_broken=errb, ok=bool(err < 1e-10 and errb > 1e-4))
    log('CHECK RBM perm vs direct', res['rbm_direct'])
    # (2) scipy evaluation of H (c2_t1_small construction) vs device matvec and the energy of an ansatz state
    def ham_scipy(J2):
        rows, cols, vals = [], [], []; dg = np.zeros(D)
        for bonds, J in ((lat.NN, 1.0), (lat.NNN, J2)):
            for (i, j) in bonds:
                anti, nbr = lat.bond_data(i, j); dg += J * np.where(anti, -0.25, 0.25)
                r = np.nonzero(anti)[0]; rows.append(r); cols.append(nbr[r]); vals.append(np.full(r.size, 0.5 * J))
        M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(D, D))
        return (M + sp.diags(dg)).tocsr()
    Hs = ham_scipy(0.5); Hbad = ham_scipy(0.45)
    rng = np.random.default_rng(1); v = rng.standard_normal(D)
    Hv = ops.Hnp(v)
    rel = np.linalg.norm(Hv - Hs @ v) / np.linalg.norm(Hs @ v); relbad = np.linalg.norm(Hv - Hbad @ v) / np.linalg.norm(Hbad @ v)
    res['matvec_vs_scipy'] = dict(rel=rel, rel_broken_J2_0p45=relbad, ok=bool(rel < 1e-12 and relbad > 1e-6))
    log('CHECK matvec vs scipy', res['matvec_vs_scipy'])
    res['energy_vs_scipy'] = {}
    for arm in ('J', 'R1', 'J_HJ2', 'C5std_R1', 'C5cpl_R1'):
        m = arm_model(lat, ops, arm); p = m.init(7, a0=-0.05)
        psi = m.psi_np(p); Ej = m.energy(p)
        Es = float(np.real(np.vdot(psi, Hs @ psi)) / np.real(np.vdot(psi, psi)))
        Eb = float(np.real(np.vdot(psi, Hbad @ psi)) / np.real(np.vdot(psi, psi)))
        res['energy_vs_scipy'][arm] = dict(E_jax=Ej, E_scipy=Es, diff=Ej - Es, diff_broken=Ej - Eb,
                                            ok=bool(abs(Ej - Es) < 1e-10 * abs(Es) and abs(Ej - Eb) > 1e-6))
        log('CHECK energy vs scipy', arm, res['energy_vs_scipy'][arm])
    # psi0 by scipy CSR eigsh
    Es0 = float(sla.eigsh(Hs, k=1, which='SA', tol=1e-12)[0][0])
    res['E0_scipy_site'] = Es0 / N; log('CHECK E0 scipy/N', Es0 / N, 'device/N', E0 / N)
    # lattice FN: exact sign -> E0; Marshall sign, |psi0| guide -> above E0; device FN vs scipy fn (c2_t1_small)
    del Hbad
    t = time.time(); i1, i2 = {}, {}
    efn_exact = fn_energy(ops, psi0, info=i1); efn_marsh = fn_energy(ops, np.abs(psi0) * lat.sN, info=i2)
    log('FN solves', i1, i2, f'{time.time() - t:.1f}s')
    s_ = lat.sN; g = np.abs(psi0); rows, cols, vals = [], [], []; dgF = Hs.diagonal().copy(); del Hs
    for bonds, J in ((lat.NN, 1.0), (lat.NNN, 0.5)):
        for (i, j) in bonds:
            anti, nbr = lat.bond_data(i, j); r = np.nonzero(anti)[0]; cc = nbr[r]; viol = s_[r] * s_[cc] > 0
            rows.append(r[~viol]); cols.append(cc[~viol]); vals.append(np.full((~viol).sum(), 0.5 * J))
            dgF += np.bincount(r[viol], weights=0.5 * J * np.minimum(g[cc[viol]] / np.maximum(g[r[viol]], 1e-300), RCAP), minlength=D)
    HF = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(D, D)) + sp.diags(dgF)
    del rows, cols, vals
    efn_marsh_scipy = ground_lobpcg(lambda x: HF @ x, HF.diagonal(), g * s_)[0]; del HF
    res['FN'] = dict(exact_sign_minus_E0_site=(efn_exact - E0) / N, marshall_minus_E0_site=(efn_marsh - E0) / N,
                     marshall_scipy_minus_E0_site=(efn_marsh_scipy - E0) / N,
                     ok=bool(abs(efn_exact - E0) / N < 1e-8 and (efn_marsh - E0) / N > 1e-6
                             and abs(efn_marsh - efn_marsh_scipy) / N < 1e-8))
    log('CHECK FN', res['FN'])
    # (1) autodiff gradient vs central finite differences on 5 parameters; broken control: FD of -E
    res['grad_fd'] = {}
    for arm in ('J', 'J_H', 'R1_HJ2', 'C5std_R1', 'C5cpl_R1'):
        m = arm_model(lat, ops, arm); p = m.init(3, a0=-0.05)
        flat, unr = ravel_pytree(p)
        Efc = lambda x, c: energy_of(m.psi(unr(x), c), lambda u: ops.H(u, c))
        Ejit = jax.jit(Efc); Ef = lambda x: Ejit(x, m.c)
        g = np.asarray(jax.jit(jax.grad(Efc))(flat, m.c))
        idx = np.random.default_rng(5).choice(flat.size, size=min(5, flat.size), replace=False)
        h = 1e-5; fd = []
        for i in idx:
            e = np.zeros(flat.size); e[i] = h
            fd.append((float(Ef(flat + e)) - float(Ef(flat - e))) / (2 * h))
        fd = np.array(fd); ga = g[idx]
        err = float(np.max(np.abs(ga - fd)) / max(np.max(np.abs(fd)), 1e-12))
        err_bad = float(np.max(np.abs(ga - (-fd))) / max(np.max(np.abs(fd)), 1e-12))
        res['grad_fd'][arm] = dict(idx=idx.tolist(), autodiff=ga.tolist(), fd=fd.tolist(), rel_err=err,
                                   rel_err_flipped=err_bad, ok=bool(err < 1e-5 and err_bad > 1e-2))
        log('CHECK grad vs FD', arm, res['grad_fd'][arm])
    # timing per arm type (30 steps)
    res['timing'] = {}
    for arm in ('J', 'J_H', 'J_HJ2', 'R2', 'R2_H', 'R2_HJ2', 'C5std_R1', 'C5cpl_R1', 'C5std_R2', 'C5std_R4', 'C5cpl_R4'):
        m = arm_model(lat, ops, arm); p = m.init(0, a0=-0.05)
        t = time.time(); r_ = train(m, p, max_steps=1, progress_every=0); tc = time.time() - t
        t = time.time(); r_ = train(m, p, max_steps=30, progress_every=0, window=10 ** 9); ts = (time.time() - t) / 30
        res['timing'][arm] = dict(first_step_s=tc, s_per_step=ts, E_site_after30=r_['E'] / N)
        log('TIMING', arm, res['timing'][arm])
        json.dump(res, open(os.path.join(outdir, 'checks.json'), 'w'), indent=1, default=float)
    # (3) J2 = 0 Marshall-Jastrow within 1% of E0(J2=0); broken control: same Jastrow without the Marshall sign
    lat0 = Lattice(LX, LY, 0.0); ops0 = Ops(lat0, J2=0.0)
    E00, _ = lanczos_ground(ops0.Hnp, lat0.D)
    m0 = Model(lat0, ops0, 'J', 'plain'); r0 = train(m0, m0.init(0), progress_every=500, tag='J2=0 Jastrow')
    m0.c['sN'] = jnp.ones(lat0.D); r0b = train(m0, m0.init(0), max_steps=500, progress_every=0, tag='J2=0 no-sign')
    res['J2zero_jastrow'] = dict(E0_site=E00 / N, E_var_site=r0['E'] / N, rel=(r0['E'] - E00) / abs(E00),
                                 steps=r0['steps'], E_var_nosign_site=r0b['E'] / N, rel_nosign=(r0b['E'] - E00) / abs(E00))
    res['J2zero_jastrow']['ok'] = bool(res['J2zero_jastrow']['rel'] < 0.01 and res['J2zero_jastrow']['rel_nosign'] > 0.01)
    log('CHECK J2=0 Jastrow', res['J2zero_jastrow'])
    oks = [res.get('E0_check', {}).get('ok', True), res['matvec_vs_scipy']['ok'], res['FN']['ok'], res['J2zero_jastrow']['ok'], res['rbm_direct']['ok']] + \
          [v['ok'] for v in res['energy_vs_scipy'].values()] + [v['ok'] for v in res['grad_fd'].values()]
    res['all_ok'] = bool(all(oks))
    json.dump(res, open(os.path.join(outdir, 'checks.json'), 'w'), indent=1, default=float)
    log(f'VERDICT exact_vmc checks all_ok={res["all_ok"]} ({sum(oks)}/{len(oks)})')
    return res

# ------------------------------------------------------------------ run mode
def run_spec(outdir, spec_path):
    spec = json.load(open(spec_path))
    LX, LY = spec.get('LX', 4), spec.get('LY', 6)
    lat, ops, psi0, E0 = setup(LX, LY, spec.get('J2', 0.5), spec.get('psi0_dir', outdir))
    if (LX, LY) == (4, 6) and abs(E0 / lat.N - E0_REF_4x6) > 1e-9:
        log(f'ABORT E0/N {E0 / lat.N} != {E0_REF_4x6}'); sys.exit(2)
    os.makedirs(os.path.join(outdir, 'runs'), exist_ok=True)
    max_steps = spec.get('max_steps', 3000); screen = spec.get('a_screen_steps', 0)
    for job in spec['runs']:
        arm, seed = job['arm'], job['seed']; name = f'{arm}_s{seed}'
        fout = os.path.join(outdir, 'runs', name + '.json')
        if os.path.exists(fout): log('skip existing', name); continue
        m = arm_model(lat, ops, arm); tr = {}
        if m.layer is not None:
            a0s = job.get('a0s', [0.0, -0.05, 0.05]); cand = []
            for a0 in a0s:
                st = screen if screen else max_steps
                r = train(m, m.init(seed, a0), max_steps=st, progress_every=500, tag=f'{name} a0={a0}')
                tr[f'a0={a0}'] = dict(E_site=r['E'] / lat.N, steps=r['steps'], a=float(r['params']['a']),
                                      converged=r['converged'], curve=r['curve'], sec=r['sec'])
                cand.append((r['E'], a0, r))
                log(f'PROGRESS {name} a0={a0} E/N {r["E"] / lat.N:.12f} a {float(r["params"]["a"]):.5f} steps {r["steps"]}')
            cand.sort(key=lambda z: z[0]); best_a0, r = cand[0][1], cand[0][2]
            if screen:     # continue the best screened start from its parameters
                r2 = train(m, r['params'], max_steps=max_steps - r['steps'], progress_every=500, tag=f'{name} cont')
                r2['curve'] = r['curve'][:-1] + r2['curve']; r2['steps'] += r['steps']; r2['sec'] += sum(c[2]['sec'] for c in cand)
                r = r2
        else:
            best_a0 = None
            r = train(m, m.init(seed), max_steps=max_steps, progress_every=500, tag=name)
        d = diagnostics(m, r['params'], psi0, E0)
        out = dict(arm=arm, seed=seed, n_params=r['n_params'], steps=r['steps'], converged=r['converged'], sec=r['sec'],
                   best_a0=best_a0, a=float(r['params']['a']) if 'a' in r['params'] else None,
                   E0_site=E0 / lat.N, **d, curve=r['curve'], a_screen=tr)
        json.dump(out, open(fout, 'w'), default=float)
        np.save(os.path.join(outdir, 'runs', name + '_params.npy'), r['flat'])
        del m; gc.collect()
        log(f'RESULT {name} dE_var/N {d["dE_var_site"]:.4e} infid {d["infidelity"]:.4e} W_M {d["W_M"]:.4e} '
            f'dE_FN/N {d["dE_FN_site"]:.4e} npar {r["n_params"]} steps {r["steps"]} conv {r["converged"]} {r["sec"]:.0f}s')
    log(f'VERDICT exact_vmc spec {os.path.basename(spec_path)} done')

# ------------------------------------------------------------------ verdicts
def verdict(outdir):
    import glob
    R = {}
    for f in glob.glob(os.path.join(outdir, '**', 'runs', '*.json'), recursive=True) + glob.glob(os.path.join(outdir, 'runs', '*.json')):
        d = json.load(open(f)); R.setdefault(d['arm'], {})[d['seed']] = d
    tab = {}
    for arm, seeds in sorted(R.items()):
        best = min(seeds.values(), key=lambda d: d['dE_var_site'])
        dv = [d['dE_var_site'] for d in seeds.values()]
        tab[arm] = dict(nseeds=len(seeds), best_seed=best['seed'], dE_var=best['dE_var_site'], spread=max(dv) - min(dv),
                        infid=best['infidelity'], W_M=best['W_M'], dE_FN=best['dE_FN_site'], npar=best['n_params'],
                        steps=best['steps'], conv=best['converged'], a=best['a'])
        t = tab[arm]
        print(f'{arm:12s} n={t["nseeds"]} dE_var/N {t["dE_var"]:.4e} (spread {t["spread"]:.1e}) infid {t["infid"]:.3e} '
              f'W_M {t["W_M"]:.3e} dE_FN/N {t["dE_FN"]:.4e} npar {t["npar"]} steps {t["steps"]} conv {t["conv"]} a {t["a"]}')
    out = dict(table=tab)
    # C2-T2: Pass: J2 layer E_var <= 0.5 x full-H layer for J and R2.  Kill: J2 layer >= full-H layer for both.
    if all(k in tab for k in ('J_H', 'J_HJ2', 'R2_H', 'R2_HJ2')):
        r = {b: tab[f'{b}_HJ2']['dE_var'] / tab[f'{b}_H']['dE_var'] for b in ('J', 'R2')}
        v = 'PASS' if all(x <= 0.5 for x in r.values()) else 'KILL' if all(x >= 1.0 for x in r.values()) else 'INCONCLUSIVE'
        out['C2-T2'] = dict(ratio_HJ2_over_H=r, verdict=v)
        print(f'VERDICT C2-T2 {v}: (E_var-E0) ratio HJ2/H layer J {r["J"]:.3f}, R2 {r["R2"]:.3f}')
    # C5-T0: Pass: coupled E_var <= 0.5 x standard for 2 of 3 alphas and lower E_FN. Kill: coupled >= standard for all 3.
    al = [a for a in (1, 2, 4) if f'C5std_R{a}' in tab and f'C5cpl_R{a}' in tab]
    if len(al) == 3:
        r = {a: tab[f'C5cpl_R{a}']['dE_var'] / tab[f'C5std_R{a}']['dE_var'] for a in al}
        fn = {a: tab[f'C5cpl_R{a}']['dE_FN'] < tab[f'C5std_R{a}']['dE_FN'] for a in al}
        npass = sum(1 for a in al if r[a] <= 0.5 and fn[a])
        v = 'PASS' if npass >= 2 else 'KILL' if all(r[a] >= 1.0 for a in al) else 'INCONCLUSIVE'
        out['C5-T0'] = dict(ratio_cpl_over_std=r, cpl_lower_FN=fn, verdict=v)
        print(f'VERDICT C5-T0 {v}: (E_var-E0) ratio coupled/standard ' + ', '.join(f'R{a} {r[a]:.3f}' for a in al) +
              '; coupled lower E_FN ' + ', '.join(f'R{a} {fn[a]}' for a in al))
    json.dump(out, open(os.path.join(outdir, 'verdict.json'), 'w'), indent=1, default=float)
    return out

if __name__ == '__main__':
    mode, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    LOGF = os.path.join(outdir, f'exact_vmc_{mode}.log')
    log('jax', jax.__version__, 'optax', optax.__version__, 'devices', jax.devices())
    if mode == 'check':
        LX, LY = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (4, 6)
        r = run_checks(outdir, LX, LY)
        sys.exit(0 if r['all_ok'] else 3)
    elif mode == 'run':
        run_spec(outdir, sys.argv[3])
    elif mode == 'verdict':
        verdict(outdir)
