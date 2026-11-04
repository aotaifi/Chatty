"""pp2_lemma.py: batched JAX evaluator of B = P_1 + lam P_2 (two symmetry-projected pair products, optional RBM factor) and of the
augmented guide G = s_G B e^{add}, add = c f_J[B] (fixed), with all one- and two-swap ratios from the rank-2 / rank-4 determinant
lemma on G_g = M_g(x)^-1 per image (PLAN_pfaffian.md "B design", build item (i)).  General L (6, 8), f64 or f32.

Wave function (sigma_i = +-1, site = x + L y; images z_g = FL_g sigma[IMG_g], IMG/FL as rbmpp_lib.Lattice: 'full' = 8 L^2 x flip,
'translations' = L^2 x flip):
  per image g:  a_j(g) = par(z_g) det f_j[Up(z_g), Dn(z_g)]          (j = 1, 2; f_j complex N x N)
  rbm = 'none'  : B = sum_g [a_1(g) + lam a_2(g)]
  rbm = 'inner' : B = sum_g phi(z_g) [a_1(g) + lam a_2(g)],  phi(z) = prod_k cosh(b_k + z.W[:, k])     (rbmpp_lib.make_fns; 8x8 production)
  rbm = 'outer' : B = exp(sum_g sum_k lncosh(b_k + z_g.W[:, k])) sum_g [a_1(g) + lam a_2(g)]           (cx_r1fit.py arm 'pp2'; 6x6 round 5)
  ('inner' with P_2 puts the RBM on both PPs: B = sum_g phi(z_g) [a_1 + lam a_2].  A design choice, see the report.)
Lemma (walker_cost.py CSRC 'B-form', verified there to 1e-12): a net move removes k up sites (slots P) and k down sites (slots Q) of an
image, k <= 2; ratio_g = (-1)^e det [[f[A,R] - W[Q,P], -T[Q,P]], [S[Q,P], G[Q,P]]] with T = f[Do,Do] G, S = G f[Uo,Uo], W = T f[Uo,Uo].
A two-swap endpoint z = swap_b(swap_a(x)) has net k = 2 (disjoint bonds), 1 (bonds share a site) or 0 (b = a).

API (params = a PP2 object; sigma, x = (B, N) arrays of +-1):
  logB(params, sigma, dtype='f64')                       -> complex (B,)  (direct: slogdet of every image)
  one_swap_ratios(params, x, dtype='f64')                -> complex (B, NB)  B(y_b)/B(x) for every bond b (J1 then J2; 0 if aligned)
  two_swap_ratios(params, x, dtype='f64', dmax=None)     -> dict  a (B, dA) bond of the first swap (-1 pad), am mask,
        r1 (B, dA) = B(y_a)/B(x);  b (B, dA, dB) bond of the second swap on y_a (-1 pad), bm mask, r2 (B, dA, dB) = B(z_ab)/B(x),
        ru, rd (B, dA, dB, 2) net removed up / down sites of z_ab vs x, k (B, dA, dB) net swap count; dB = dA + 15.
  fJ_fixed(params, x, sign_fn, EP, c, f0=None, mode='old', at_y=True, ...) -> dict  fx (B,), fy (B, dA) raw f_J (= r5_bare.fJ_of
        arithmetic; -inf where no kept bond), add_x, add_y (the c-multiplied term after the fix), sx, sy, plus two_swap_ratios' fields.
        mode 'old': non-finite -> 0 (= fJ_of);  'a1': non-finite -> add = node (default -69, fj_fix's stand-in; -inf is exact);
             'fix' (= the FIX: a1 + b1): non-finite -> node, else add = c sign(f) min(|f|, f0)  (c_eff = c min(1, f0/|f_J|)).
        sign_fn: callable, (M, N) +-1 array -> (M,) +-1 (numpy or jax; called outside jit, in chunks of sign_chunk).
  aug_ratios(params, x, sign_fn, EP, c, f0=None, mode='fix', ...) -> dict  ratio (B, NB) = G(y_b)/G(x) (real, signed; 0 if aligned),
        logG_x = Re logB(x) + add_x, s_x, and fJ_fixed's fields.
Evaluation paths (2026-11-04, piece (3) savings; DEFAULT_PATH = 'old' keeps every output of the functions above unchanged):
  eval_path = 'old'   : padded (dA, dB = dA + 15) layout, every (a, b) path evaluated (the A40-benchmarked code);
              'pack'  : same outputs, but only the valid (a, b) entries are evaluated (per-walker move lists packed in buckets of QUANT moves, walkers sorted by count);
              'dedup' : like 'pack', and every distinct endpoint z (x ^ flip-mask) once per walker: z = x (b = a), the (a, b) / (b, a)
                        pairs of disjoint bonds, and any other path reaching the same configuration share one evaluation.
  two_swap_ratios / fJ_fixed / aug_ratios take eval_path=None (-> DEFAULT_PATH); fJ_fixed / aug_ratios also cnn_dedup=None (-> True iff
  eval_path == 'dedup'): the sign_fn is then called once per distinct configuration among x, all y_a and all z_ab (across walkers).
  make_sign_fn(size='L'|'XL', seed, net) wraps sign8.SignNet as a jax_native sign_fn.  STATS collects counts / seconds of the new paths.
Lemma-path sign (2026-11-05, savings item (4)): sign_fn may be a PhaseSign (phase_sign_from_theta(theta, L)): the sign angle's 1-PP complex-phase class,
  S(x) = M(x) sgn(l(x)),  l(x) = Re(e^{-i theta0} P(x)) / |P(x)|,  P(x) = sum over the 8 L^2 x 2 images g of par(z_g) det f[Up(z_g), Dn(z_g)]   (f complex, no RBM),
  M = Marshall (-1)^{#up on the sublattice x + y even}, sgn(0) = +1.   fJ_fixed / aug_ratios then get sx, sy, sz from a SECOND lemma evaluation (its own per-image
  inverse), P(y)/P(x), P(z)/P(x) of every valid endpoint, with the same eval_path ('old' / 'pack' / 'dedup') and the same move lists as the amplitude's; no sign_fn call
  on a full configuration.  M(endpoint) = M(x) (-1)^{popcount(flip mask & A-sublattice mask)}.  Outputs have the same keys / shapes as with any other sign_fn.
Constructors: PP2.from_cx_theta(theta, L=6, NA=4) (6x6 round-5 theta, 'outer', full projection); PP2.from_rbmpp_npz(path)
(production rbmpp params: f real, complex RBM 'inner', projection read from the file); p.with_p2(f2, lam).
"""
import functools
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

NODE_ADD = -69.0


# ------------------------------------------------------------------------------------------ lattice
class Lat:
    def __init__(self, L, proj, J2=0.5):
        self.L = L; self.N = N = L * L; self.n = N // 2; self.proj = proj
        site = lambda x, y: (x % L) + L * (y % L)
        nn_, nnn = [], []
        for y in range(L):
            for x in range(L):
                i = site(x, y)
                nn_ += [(i, site(x + 1, y)), (i, site(x, y + 1))]
                nnn += [(i, site(x + 1, y + 1)), (i, site(x + 1, y - 1))]
        bl = nn_ + nnn
        self.BI = np.array([b[0] for b in bl], np.int32); self.BJ = np.array([b[1] for b in bl], np.int32)
        self.JB = np.array([1.0] * len(nn_) + [J2] * len(nnn)); self.HW = self.JB / 2; self.NB = len(bl)
        inc = [[] for _ in range(N)]
        for b in range(self.NB): inc[self.BI[b]].append(b); inc[self.BJ[b]].append(b)
        assert all(len(v) == 8 for v in inc)
        self.INC = np.array(inc, np.int32)                                  # (N, 8) bonds incident to each site

        def pm(f):
            p = np.empty(N, np.int64)
            for y in range(L):
                for x in range(L):
                    xx, yy = f(x, y); p[site(x, y)] = site(xx, yy)
            return p
        c4 = pm(lambda x, y: (-y, x)); sx = pm(lambda x, y: (-x, y)); r = np.arange(N); pts = []
        for k in range(4):
            for refl in (False, True):
                q = r.copy()
                for _ in range(k): q = c4[q]
                if refl: q = sx[q]
                pts.append(q)
        G = []
        for ty in range(L):
            for tx in range(L):
                t = pm(lambda x, y: (x + tx, y + ty))
                for q in pts: G.append(t[q])
        G = np.array(G)                                                     # rbmpp_lib.Lattice.space_group order
        if proj == 'full': P = G
        elif proj == 'translations': P = np.stack([G[8 * t] for t in range(N)])
        elif proj == 'none': P = G[:1]
        elif proj == 'full_noflip': P = G                                   # BROKEN-variant hook (PhaseSign broken='flip'): the spin-flip images dropped
        else: raise ValueError(proj)
        fl = [1.0, -1.0] if proj not in ('none', 'full_noflip') else [1.0]
        self.IMG = np.concatenate([P for _ in fl]).astype(np.int32)
        self.FL = np.concatenate([[f] * len(P) for f in fl])
        self.INV = np.argsort(self.IMG, axis=1).astype(np.int32)             # z_g[INV[g, s]] = FL_g sigma[s]
        self.nI = len(self.IMG)


@functools.lru_cache(None)
def lattice(L, proj):
    return Lat(L, proj)


# ------------------------------------------------------------------------------------------ parameters
@jax.tree_util.register_pytree_node_class
class PP2:
    """f1, f2 (N, N) complex; lam complex; b (M,), W (N, M) complex (rbm != 'none').  Static: L, proj, rbm, has_p2."""
    def __init__(self, f1, f2=None, lam=0.0, b=None, W=None, *, L, proj, rbm='none'):
        self.f1 = jnp.asarray(f1); self.f2 = None if f2 is None else jnp.asarray(f2); self.lam = jnp.asarray(lam)
        self.b = None if b is None else jnp.asarray(b); self.W = None if W is None else jnp.asarray(W)
        self.L = int(L); self.proj = proj; self.rbm = rbm
        assert rbm in ('none', 'inner', 'outer') and (rbm == 'none') == (W is None)

    has_p2 = property(lambda s: s.f2 is not None)
    lat = property(lambda s: lattice(s.L, s.proj))

    def tree_flatten(self):
        return (self.f1, self.f2, self.lam, self.b, self.W), (self.L, self.proj, self.rbm)

    @classmethod
    def tree_unflatten(cls, aux, ch):
        o = object.__new__(cls); o.f1, o.f2, o.lam, o.b, o.W = ch; o.L, o.proj, o.rbm = aux; return o

    def with_p2(self, f2, lam):
        return PP2(self.f1, f2, lam, self.b, self.W, L=self.L, proj=self.proj, rbm=self.rbm)

    def cast(self, dtype):
        cd = jnp.complex128 if dtype in ('f64', jnp.float64) else jnp.complex64
        c = lambda a: None if a is None else jnp.asarray(a).astype(cd)
        return PP2(c(self.f1), c(self.f2), c(self.lam), c(self.b), c(self.W), L=self.L, proj=self.proj, rbm=self.rbm)

    @staticmethod
    def from_cx_theta(theta, L=6, NA=4, proj='full'):
        """cx_r1fit.py arm 'pp2' theta = [Re f1, Im f1, RBM (NA, N+1) real | Re f2, Im f2 | Re lam, Im lam]."""
        th = np.asarray(theta, np.float64); N = L * L; NP1 = 2 * N * N + NA * (N + 1)
        assert th.size == NP1 + 2 * N * N + 2, th.size
        f1 = (th[:N * N] + 1j * th[N * N:2 * N * N]).reshape(N, N); wb = th[2 * N * N:NP1].reshape(NA, N + 1)
        f2 = (th[NP1:NP1 + N * N] + 1j * th[NP1 + N * N:NP1 + 2 * N * N]).reshape(N, N); lam = th[-2] + 1j * th[-1]
        return PP2(f1, f2, lam, wb[:, N].astype(complex), wb[:, :N].T.astype(complex), L=L, proj=proj, rbm='outer' if NA else 'none')

    @staticmethod
    def from_rbmpp_npz(path, proj=None):
        z = np.load(path); N = z['f'].shape[0]; L = int(round(np.sqrt(N)))
        pr = proj or str(z['proj']) if 'proj' in z.files else (proj or 'translations')
        if bool(z['rbm']):
            return PP2(z['f'].astype(complex), L=L, proj=pr, rbm='inner', b=z['bre'] + 1j * z['bim'], W=z['Wre'] + 1j * z['Wim'])
        return PP2(z['f'].astype(complex), L=L, proj=pr)


def _logcosh(z):
    s = jnp.where(jnp.real(z) >= 0, 1.0, -1.0).astype(jnp.real(z).dtype)
    return s * z + jnp.log1p(jnp.exp(-2.0 * s * z)) - jnp.log(2.0).astype(jnp.real(z).dtype)


# ------------------------------------------------------------------------------------------ per-walker preparation
def _prep_one(p, sig, full):
    """per image quantities of one configuration sig (N,) (real dtype).  full: also T, S, W (needed by the lemma)."""
    lat = p.lat; n = lat.n; rd = jnp.real(p.f1).dtype; cd = p.f1.dtype
    IMG = jnp.asarray(lat.IMG); FL = jnp.asarray(lat.FL, rd)
    Z = FL[:, None] * sig.astype(rd)[IMG]                                   # (nI, N)
    order = jnp.argsort(-Z, axis=1, stable=True); Uo = order[:, :n]; Do = order[:, n:]
    up = (Z > 0).astype(jnp.int32); dn = 1 - up
    cU = jnp.cumsum(up, 1); cD = jnp.cumsum(dn, 1)
    out = dict(Uo=Uo, Do=Do, pos=jnp.where(up > 0, cU - 1, cD - 1), cntUp=cU - up, cntDn=cD - dn, Z=Z)
    par = 1.0 - 2.0 * (jnp.sum(up * (cD - dn), 1) % 2).astype(rd)
    lr = 0.0
    if p.rbm != 'none':
        th = p.b[None, :] + Z.astype(cd) @ p.W                            # (nI, M)
        out['theta'] = th; lr = jnp.sum(_logcosh(th), 1)                   # (nI,)
    for j, f in ((1, p.f1), (2, p.f2)):
        if f is None: continue
        M = f[Uo[:, :, None], Do[:, None, :]]
        sg, ld = jnp.linalg.slogdet(M)
        lg = ld.astype(cd) + jnp.log(sg * par)                              # complex log of par det
        if p.rbm == 'inner': lg = lg + lr
        out[f'lg{j}'] = lg
        if full:
            G = jnp.linalg.inv(M); fUU = f[Uo[:, :, None], Uo[:, None, :]]
            T = f[Do[:, :, None], Do[:, None, :]] @ G
            out[f'G{j}'] = G; out[f'T{j}'] = T; out[f'S{j}'] = G @ fUU; out[f'W{j}'] = T @ fUU
    out['R'] = jnp.sum(lr) if p.rbm == 'outer' else jnp.zeros((), cd)
    return out


def _amps(p, pr):
    """image weights a_j(g) / exp(m) and log B = m + log S (+ R)."""
    m = jnp.max(jnp.real(pr['lg1']))
    if p.has_p2: m = jnp.maximum(m, jnp.max(jnp.real(pr['lg2'])))
    a1 = jnp.exp(pr['lg1'] - m); a2 = p.lam * jnp.exp(pr['lg2'] - m) if p.has_p2 else None
    S = jnp.sum(a1) + (jnp.sum(a2) if p.has_p2 else 0.0)
    return a1, a2, S, m + jnp.log(S) + pr['R']


@functools.partial(jax.jit, static_argnums=())
def _logB_j(p, sig):
    def one(s):
        pr = _prep_one(p, s, False); return _amps(p, pr)[3]
    return jax.vmap(one)(sig)


def logB(params, sigma, dtype='f64', chunk=256):
    p = params.cast(dtype); rd = jnp.float64 if dtype == 'f64' else jnp.float32
    S = np.asarray(sigma); out = []
    for i in range(0, len(S), chunk):
        blk = S[i:i + chunk]; m = len(blk)
        if m < chunk and len(S) > chunk: blk = np.concatenate([blk, np.repeat(blk[:1], chunk - m, 0)])   # one compile
        out.append(np.asarray(_logB_j(p, jnp.asarray(blk, rd)))[:m])
    return jnp.asarray(np.concatenate(out))


# ------------------------------------------------------------------------------------------ lemma kernel for one image
def _img_ratio(lat, f, G, T, S, W, Uo, Do, pos, cntUp, cntDn, fl, inv, ru, rd_, k, broken):
    """rho_g for K net moves in one image.  ru, rd_ (K, 2): sigma-frame removed up / down sites (slot 1 a duplicate when k < 2)."""
    n = lat.n
    Rs = jnp.where(fl > 0, inv[ru], inv[rd_]); As = jnp.where(fl > 0, inv[rd_], inv[ru])     # image frame removed up / down sites
    P = pos[Rs]; Q = pos[As]
    two = (k == 2)[:, None]
    sw = two & (P[:, 0] > P[:, 1])[:, None]; P = jnp.where(sw, P[:, ::-1], P)
    sq = two & (Q[:, 0] > Q[:, 1])[:, None]; Q = jnp.where(sq, Q[:, ::-1], Q)
    R = Uo[P]; A = Do[Q]                                                    # (K, 2) sites
    msk = jnp.stack([k >= 1, k >= 2], 1)
    e1 = (0 if broken else P + Q) + (n - cntUp[A]) + (n - cntDn[R]) + A - R
    e = jnp.sum(jnp.where(msk, e1, 0), 1)
    cmp = (R[:, None, :] > A[:, :, None]).astype(jnp.int32) + (A[:, :, None] > R[:, None, :]).astype(jnp.int32)   # [a, b]
    e = e - jnp.sum(jnp.where(msk[:, :, None] & msk[:, None, :], cmp, 0), (1, 2))
    ix = Q[:, :, None] * n + P[:, None, :]                                 # (K, 2, 2) [a, b]
    Gf = G.reshape(-1)[ix]; Tf = T.reshape(-1)[ix]; Sf = S.reshape(-1)[ix]; Wf = W.reshape(-1)[ix]
    Ff = f[A[:, :, None], R[:, None, :]]
    K11 = Ff - Wf; K12 = -Tf; K21 = Sf; K22 = Gf
    d2 = K11[:, 0, 0] * K22[:, 0, 0] - K12[:, 0, 0] * K21[:, 0, 0]
    # 4 x 4 det: rows (K11 K12; K21 K22), Laplace over the top two rows
    Km = jnp.concatenate([jnp.concatenate([K11, K12], 2), jnp.concatenate([K21, K22], 2)], 1)   # (K, 4, 4)
    m2 = lambda r0, r1, c0, c1: Km[:, r0, c0] * Km[:, r1, c1] - Km[:, r0, c1] * Km[:, r1, c0]
    d4 = (m2(0, 1, 0, 1) * m2(2, 3, 2, 3) - m2(0, 1, 0, 2) * m2(2, 3, 1, 3) + m2(0, 1, 0, 3) * m2(2, 3, 1, 2)
          + m2(0, 1, 1, 2) * m2(2, 3, 0, 3) - m2(0, 1, 1, 3) * m2(2, 3, 0, 2) + m2(0, 1, 2, 3) * m2(2, 3, 0, 1))
    det = jnp.where(k == 0, 1.0, jnp.where(k == 1, d2, d4))
    return jnp.where(e % 2 == 1, -det, det)


def _rbm_dl(th, Wimg, fl, ru, rd_, k):
    """sum_k [lncosh(theta + Delta) - lncosh(theta)] for K net moves in one image; Delta = FL 2 (sum_rd W - sum_ru W)."""
    msk = jnp.stack([k >= 1, k >= 2], 1)[:, :, None]
    D = 2.0 * fl * jnp.sum(jnp.where(msk, Wimg[rd_] - Wimg[ru], 0.0), 1)      # (K, M)
    return jnp.sum(_logcosh(th[None, :] + D) - _logcosh(th)[None, :], 1)


def _ratios_batch(p, X, ru, rd_, k, img_chunk=1, broken=False):
    """B(x')/B(x) for K net moves per walker.  X (B, N); ru, rd_ (B, K, 2); k (B, K).  Returns (logB(x) (B,), ratios (B, K))."""
    lat = p.lat; cd = p.f1.dtype
    prs = jax.vmap(lambda s: _prep_one(p, s, True))(X)
    a1, a2, Ssum, lB = jax.vmap(lambda pr: _amps(p, pr))(prs)               # a (B, nI)
    FL = jnp.asarray(lat.FL, jnp.real(p.f1).dtype); INV = jnp.asarray(lat.INV)
    Wimg = p.W[INV] if p.rbm != 'none' else None                            # (nI, N, M)
    nI = lat.nI; gc = img_chunk; assert nI % gc == 0
    keys = ['Uo', 'Do', 'pos', 'cntUp', 'cntDn', 'G1', 'T1', 'S1', 'W1'] + (['G2', 'T2', 'S2', 'W2'] if p.has_p2 else []) + \
           (['theta'] if p.rbm != 'none' else [])
    xs = {kk: jnp.moveaxis(prs[kk], 1, 0).reshape((nI // gc, gc) + prs[kk].shape[:1] + prs[kk].shape[2:]) for kk in keys}
    xs['a1'] = a1.T.reshape(nI // gc, gc, -1)
    if p.has_p2: xs['a2'] = a2.T.reshape(nI // gc, gc, -1)
    xs['fl'] = FL.reshape(nI // gc, gc); xs['inv'] = INV.reshape(nI // gc, gc, -1)
    if Wimg is not None: xs['Wimg'] = Wimg.reshape((nI // gc, gc) + Wimg.shape[1:])
    Bn, K = k.shape

    def per_walker(d, ru1, rd1, k1):                                        # one image, one walker
        r1 = _img_ratio(lat, p.f1, d['G1'], d['T1'], d['S1'], d['W1'], d['Uo'], d['Do'], d['pos'], d['cntUp'], d['cntDn'],
                        d['fl'], d['inv'], ru1, rd1, k1, broken)
        t = d['a1'] * r1
        if p.has_p2:
            r2 = _img_ratio(lat, p.f2, d['G2'], d['T2'], d['S2'], d['W2'], d['Uo'], d['Do'], d['pos'], d['cntUp'], d['cntDn'],
                            d['fl'], d['inv'], ru1, rd1, k1, broken)
            t = t + d['a2'] * r2
        dl = _rbm_dl(d['theta'], d['Wimg'], d['fl'], ru1, rd1, k1) if p.rbm != 'none' else None
        if p.rbm == 'inner': t = t * jnp.exp(dl)
        return t, dl

    def per_image(d):
        wk = {kk: d[kk] for kk in keys + ['a1'] + (['a2'] if p.has_p2 else [])}
        sh = dict(fl=d['fl'], inv=d['inv'], Wimg=d.get('Wimg'))
        def f1(w):
            return per_walker({**w, **sh}, *w['_mv'])
        return jax.vmap(lambda w, a, b_, c: per_walker({**w, **sh}, a, b_, c))(wk, ru, rd_, k)

    def body(carry, d):
        acc, accl = carry
        t, dl = jax.vmap(per_image)(d)                                     # (gc, B, K)
        acc = acc + jnp.sum(t, 0)
        if p.rbm == 'outer': accl = accl + jnp.sum(dl, 0)
        return (acc, accl), None
    init = (jnp.zeros((Bn, K), cd), jnp.zeros((Bn, K), cd))
    (acc, accl), _ = jax.lax.scan(body, init, xs)
    rat = acc / Ssum[:, None]
    if p.rbm == 'outer': rat = rat * jnp.exp(accl)
    return lB, rat


# ------------------------------------------------------------------------------------------ move lists
def _one_moves(lat, X):
    BI = jnp.asarray(lat.BI); BJ = jnp.asarray(lat.BJ)
    si = X[:, BI]; sj = X[:, BJ]; valid = si != sj
    up = jnp.where(si > 0, BI, BJ); dn = jnp.where(si > 0, BJ, BI)
    ru = jnp.stack([up, up], -1); rd_ = jnp.stack([dn, dn], -1)
    return ru, rd_, valid.astype(jnp.int32), valid


def _two_moves(lat, X, dmax):
    """compact first swaps a (dmax) and second swaps b on y_a (dmax + 15), with the net (ru, rd, k) of z_ab vs x."""
    BI = jnp.asarray(lat.BI); BJ = jnp.asarray(lat.BJ); INC = jnp.asarray(lat.INC)
    def one(x):
        valid = x[BI] != x[BJ]
        A = jnp.argsort(~valid, stable=True)[:dmax].astype(jnp.int32); Am = valid[A]
        s = BI[A]; t = BJ[A]
        upa = jnp.where(x[s] > 0, s, t); dna = jnp.where(x[s] > 0, t, s)     # (dA,)
        # candidates: part 1 = compact x-valid bonds not incident to {s, t}; part 2 = bonds incident to {s, t} (a once)
        Bs = BI[A]; Bt = BJ[A]
        inc1 = (Bs[None, :] == s[:, None]) | (Bs[None, :] == t[:, None]) | (Bt[None, :] == s[:, None]) | (Bt[None, :] == t[:, None])
        m1 = Am[None, :] & Am[:, None] & ~inc1
        it = INC[t]; it = jnp.take_along_axis(it, jnp.argsort(it == A[:, None], axis=1, stable=True), 1)[:, :7]
        C2 = jnp.concatenate([INC[s], it], 1)                                # (dA, 15)
        Bb = jnp.concatenate([jnp.broadcast_to(A[None, :], (dmax, dmax)), C2], 1)   # (dA, dB)
        # y_a values at the bond ends of b
        def yval(site):
            return jnp.where(site == s[:, None], x[t][:, None], jnp.where(site == t[:, None], x[s][:, None], x[site]))
        bi = BI[Bb]; bj = BJ[Bb]; ybi = yval(bi); ybj = yval(bj)
        m2 = Am[:, None] & (ybi != ybj)
        bm = jnp.concatenate([m1, m2[:, dmax:]], 1)
        upb = jnp.where(ybi > 0, bi, bj); dnb = jnp.where(ybi > 0, bj, bi)   # orientation in y_a
        same = Bb == A[:, None]
        sh_up = (bi == upa[:, None]) | (bj == upa[:, None])                 # b touches up_a (now down in y)
        sh_dn = (bi == dna[:, None]) | (bj == dna[:, None])                 # b touches dn_a (now up in y)
        kk = jnp.where(same, 0, jnp.where(sh_up | sh_dn, 1, 2))
        UA = jnp.broadcast_to(upa[:, None], Bb.shape); DA = jnp.broadcast_to(dna[:, None], Bb.shape)
        # k = 2: ru = {up_a, up_b}, rd = {dn_a, dn_b};  k = 1 via up_a: ru = {up_b (= other end, up in y)}, rd = {dn_a};
        # k = 1 via dn_a: ru = {up_a}, rd = {dn_b (other end, down in y)}
        r0 = jnp.where(kk == 1, jnp.where(sh_up, upb, UA), UA); d0 = jnp.where(kk == 1, jnp.where(sh_up, DA, dnb), DA)
        r1 = jnp.where(kk == 2, upb, r0); d1 = jnp.where(kk == 2, dnb, d0)
        kk = jnp.where(bm, kk, 0)
        return A, Am, jnp.where(bm, Bb, -1), bm, jnp.stack([r0, r1], -1), jnp.stack([d0, d1], -1), kk, upa, dna
    return jax.vmap(one)(X)


def _dmax_of(lat, X):
    d = int(np.max(np.sum(np.asarray(X)[:, lat.BI] != np.asarray(X)[:, lat.BJ], 1)))
    return int(min(lat.NB, 8 * ((d + 7) // 8)))


@functools.partial(jax.jit, static_argnames=('img_chunk', 'broken'))
def _one_j(p, X, img_chunk=1, broken=False):
    ru, rd_, k, valid = _one_moves(p.lat, X)
    lB, r = _ratios_batch(p, X, ru, rd_, k, img_chunk, broken)
    return lB, jnp.where(valid, r, 0.0)


def one_swap_ratios(params, x, dtype='f64', img_chunk=1, walker_chunk=256, _broken=False, return_logB=False):
    p = params.cast(dtype); rd = jnp.float64 if dtype == 'f64' else jnp.float32; x = np.asarray(x)
    parts = [_one_j(p, jnp.asarray(x[i:i + walker_chunk], rd), img_chunk=img_chunk, broken=_broken) for i in range(0, len(x), walker_chunk)]
    lB = jnp.concatenate([q[0] for q in parts]); r = jnp.concatenate([q[1] for q in parts])
    return (lB, r) if return_logB else r


@functools.partial(jax.jit, static_argnames=('dmax', 'img_chunk', 'broken'))
def _two_j(p, X, dmax, img_chunk=1, broken=False):
    A, Am, Bb, bm, ru, rd_, kk, upa, dna = _two_moves(p.lat, X, dmax)
    Bn = X.shape[0]; dB = Bb.shape[2]
    ru1 = jnp.stack([upa, upa], -1)[:, :, None, :]; rd1 = jnp.stack([dna, dna], -1)[:, :, None, :]
    RU = jnp.concatenate([ru1, ru], 2).reshape(Bn, -1, 2); RD = jnp.concatenate([rd1, rd_], 2).reshape(Bn, -1, 2)
    K = jnp.concatenate([Am[:, :, None].astype(jnp.int32), kk], 2).reshape(Bn, -1)
    lB, r = _ratios_batch(p, X, RU, RD, K, img_chunk, broken)
    r = r.reshape(Bn, dmax, dB + 1)
    r1 = jnp.where(Am, r[:, :, 0], 0.0); r2 = jnp.where(bm, r[:, :, 1:], 0.0)
    return dict(logB=lB, a=jnp.where(Am, A, -1), am=Am, r1=r1, b=Bb, bm=bm, r2=r2, ru=ru, rd=rd_, k=kk, upa=upa, dna=dna)


def two_swap_ratios(params, x, dtype='f64', dmax=None, img_chunk=1, walker_chunk=256, _broken=False, eval_path=None):
    path = eval_path or DEFAULT_PATH
    if path != 'old':
        assert path in ('pack', 'dedup'), path
        return _two_new(params, x, dtype, dmax, walker_chunk, path, _broken)
    p = params.cast(dtype); rd = jnp.float64 if dtype == 'f64' else jnp.float32
    x = np.asarray(x); dm = dmax or _dmax_of(p.lat, x)
    if len(x) <= walker_chunk: return _two_j(p, jnp.asarray(x, rd), dm, img_chunk=img_chunk, broken=_broken)
    parts = []
    for i in range(0, len(x), walker_chunk):
        parts.append(_two_j(p, jnp.asarray(x[i:i + walker_chunk], rd), dm, img_chunk=img_chunk, broken=_broken))
        if PROGRESS is not None: jax.block_until_ready(parts[-1]['r2']); PROGRESS(i + walker_chunk, len(x))
    return {k: jnp.concatenate([q[k] for q in parts]) for k in parts[0]}


# ------------------------------------------------------------------------------------------ savings: flat move lists, endpoint dedup
DEFAULT_PATH = os.environ.get('PP2_DEFAULT_PATH', 'old')   # 'old' until the lead switches (PP2_DEFAULT_PATH=pack|dedup: test hook)
STATS = {}                            # counts and host seconds of the new paths (reset by the caller)
PROGRESS = None                       # optional callback (walkers done, walkers total) after every walker chunk of two_swap_ratios (job heartbeat)
_SENT = np.uint64(0xFFFFFFFFFFFFFFFF)


def _stat(k, v): STATS[k] = STATS.get(k, 0) + v


def _bit(s): return jnp.left_shift(jnp.uint64(1), s.astype(jnp.uint64))


@jax.jit
def _flip_masks(upa, dna, Am, ru, rd_, kk, bm):
    """flip mask (uint64, bit s = site s flipped) of every entry, flat layout E = dA + dA dB: the dA first swaps (y_a), then the (a, b)
    endpoints z_ab; z = x ^ mask.  Invalid entries (Am / bm false) get mask 0 (= z = x, as the old padded _configs does)."""
    Bn, dA = Am.shape
    m0 = jnp.where(Am, _bit(upa) | _bit(dna), jnp.uint64(0))
    mz = jnp.where(kk >= 1, _bit(ru[..., 0]) | _bit(rd_[..., 0]), jnp.uint64(0)) | \
         jnp.where(kk >= 2, _bit(ru[..., 1]) | _bit(rd_[..., 1]), jnp.uint64(0))
    mz = jnp.where(bm, mz, jnp.uint64(0))
    return jnp.concatenate([m0, mz.reshape(Bn, -1)], 1)


@jax.jit
def _flip_masks0(upa, dna, Am):
    return jnp.where(Am, _bit(upa) | _bit(dna), jnp.uint64(0))


@jax.jit
def _pair_keys(A, Bb, P):
    """BROKEN-variant key: the unordered bond pair {a mod P, b mod P} (P = 2N: the lattice index of the bond without its type J1 / J2)
    instead of the configuration; col 0: {a mod P, none}.  Flat layout as _flip_masks."""
    Bn, dA = A.shape; Pu = jnp.asarray(P).astype(jnp.uint64); a = A.astype(jnp.uint64) % Pu
    k0 = a * jnp.uint64(1024) + jnp.uint64(1000)
    b = jnp.where(Bb < 0, jnp.uint64(1000), Bb.astype(jnp.uint64) % Pu); a3 = jnp.broadcast_to(a[:, :, None], b.shape)
    kz = jnp.minimum(a3, b) * jnp.uint64(1024) + jnp.maximum(a3, b)
    return jnp.concatenate([k0, kz.reshape(Bn, -1)], 1)


def _row_unique(key, valid):
    """per-row unique of key over the valid entries (numpy).  Returns gid (B, E) group id within the row (-1 invalid), (rb, re) walker
    and entry index of the FIRST entry of every group (flat, row-major), cnt (B,) groups per row."""
    kk = np.where(valid, key, _SENT)
    order = np.argsort(kk, axis=1, kind='stable')
    sk = np.take_along_axis(kk, order, 1)
    new = np.empty(sk.shape, bool); new[:, 0] = True; new[:, 1:] = sk[:, 1:] != sk[:, :-1]
    new &= sk != _SENT
    gid_s = np.cumsum(new, 1) - 1
    gid = np.empty_like(gid_s); np.put_along_axis(gid, order, gid_s, 1)
    gid = np.where(valid, gid, -1)
    rb, rs = np.nonzero(new)
    return gid, rb, order[rb, rs], new.sum(1)


def _cfg_keys(Xn):
    Xn = np.asarray(Xn)
    return ((Xn > 0).astype(np.uint64) << np.arange(Xn.shape[1], dtype=np.uint64)[None]).sum(1, dtype=np.uint64)


_two_moves_j = jax.jit(_two_moves, static_argnums=(0, 2))


def _moves_chunk(p, Xc, m, dm, path, broken):
    """pass 1 for one walker chunk (Xc (Bc, N) device, the first m walkers real): the move lists of the new paths and the bookkeeping that maps
    them back.  Entry layout E = dA + dA dB (first swaps y_a, then the endpoints z_ab).  Returns (d, mask, rep) with rep = dict(w, ru, rd, k)
    the moves to evaluate (walker index within the chunk, row-major) and gidx (m, E) = index of each entry's move within rep (-1 invalid)."""
    A, Am, Bb, bm, ru, rd_, kk, upa, dna = _two_moves_j(p.lat, Xc, dm)
    Bc, dA = Am.shape; dB = Bb.shape[2]; E = dA + dA * dB
    RU = jnp.concatenate([jnp.stack([upa, upa], -1), ru.reshape(Bc, dA * dB, 2)], 1)
    RD = jnp.concatenate([jnp.stack([dna, dna], -1), rd_.reshape(Bc, dA * dB, 2)], 1)
    K = jnp.concatenate([Am.astype(jnp.int32), kk.reshape(Bc, -1)], 1)
    V = np.array(jnp.concatenate([Am, bm.reshape(Bc, -1)], 1)); V[m:] = False
    mask = np.asarray(_flip_masks(upa, dna, Am, ru, rd_, kk, bm))
    RUn = np.asarray(RU); RDn = np.asarray(RD); Kn = np.asarray(K)
    if path == 'dedup':
        key = np.asarray(_pair_keys(A, Bb, p.lat.NB // 2)) if broken == 'pair' else mask
        gid, mw, me, cnt = _row_unique(key, V)
        gidx = np.where(V, (np.cumsum(cnt) - cnt)[:, None] + gid, -1)
    else:
        mw, me = np.nonzero(V)
        gidx = np.where(V, np.cumsum(V.ravel()).reshape(V.shape) - 1, -1)
    if broken == 'trunc':                                                  # BROKEN: the last 10% of the moves never evaluated (their entries stay 0)
        keep = int(0.9 * len(mw)); mw, me = mw[:keep], me[:keep]; gidx = np.where(gidx < keep, gidx, -1)
    rep = dict(w=mw, ru=RUn[mw, me], rd=RDn[mw, me], k=Kn[mw, me])
    d = dict(logB=None, a=jnp.where(Am, A, -1), am=Am, b=Bb, bm=bm, ru=ru, rd=rd_, k=kk, upa=upa, dna=dna)
    return {k_: (v[:m] if v is not None else None) for k_, v in d.items()}, mask[:m], rep, gidx[:m], V[:m]


_UQ = {}                           # flip masks of the last new-path two_swap_ratios call, keyed by identity of its r2 array


QUANT = 512                           # move-list bucket quantum (K is rounded up to a multiple: few compiled shapes)


def _two_new(params, x, dtype, dmax, walker_chunk, path, broken):
    """two_swap_ratios on the 'pack' / 'dedup' paths: same output dict (a plain dict: exactly the old keys).  Pass 1 builds the move lists chunk by chunk;
    pass 2 sorts the walkers by their move count, packs them in chunks of walker_chunk with K = the chunk's max count rounded up to QUANT, and runs
    the old kernel (_ratios_batch: identical arithmetic per move) on the packed lists; pass 3 scatters the results back to the padded layout."""
    p = params.cast(dtype); rd = jnp.float64 if dtype == 'f64' else jnp.float32
    x = np.asarray(x); M = len(x); dm = dmax or _dmax_of(p.lat, x); t0 = time.perf_counter()
    P1 = []; off = 0; mv = dict(w=[], ru=[], rd=[], k=[])
    for i in range(0, M, walker_chunk):
        blk = x[i:i + walker_chunk]; m = len(blk)
        if m < walker_chunk and M > walker_chunk: blk = np.concatenate([blk, np.repeat(blk[:1], walker_chunk - m, 0)])
        d, mk, rep, gidx, V = _moves_chunk(p, jnp.asarray(blk, rd), m, dm, path, broken)
        P1.append((i, m, d, mk, gidx, V, off)); mv['w'].append(rep['w'] + i); mv['ru'].append(rep['ru']); mv['rd'].append(rep['rd']); mv['k'].append(rep['k'])
        off += len(rep['w'])
    mv = {k_: np.concatenate(v) for k_, v in mv.items()}; Mtot = len(mv['w'])
    u = np.bincount(mv['w'], minlength=M); start = np.cumsum(u) - u; pos = np.arange(Mtot) - start[mv['w']]
    order = np.argsort(u, kind='stable'); chunk_of = np.empty(M, np.int64); row_of = np.empty(M, np.int64)
    chunk_of[order] = np.arange(M) // walker_chunk; row_of[order] = np.arange(M) % walker_chunk
    res = np.zeros(Mtot, np.complex128 if dtype == 'f64' else np.complex64); lB = np.zeros(M, np.complex128 if dtype == 'f64' else np.complex64)
    t1 = time.perf_counter(); nsh = set()
    for c in range((M + walker_chunk - 1) // walker_chunk):
        ws = order[c * walker_chunk:(c + 1) * walker_chunk]; Bc = len(ws); Bp = walker_chunk if M > walker_chunk else Bc
        Kb = int(max(QUANT, QUANT * int(np.ceil(u[ws].max() / QUANT)))); nsh.add(Kb)
        sel = np.nonzero(chunk_of[mv['w']] == c)[0]; r_ = row_of[mv['w'][sel]]; q_ = pos[sel]
        RUp = np.zeros((Bp, Kb, 2), np.int32); RDp = np.zeros((Bp, Kb, 2), np.int32); Kp = np.zeros((Bp, Kb), np.int32)
        RUp[r_, q_] = mv['ru'][sel]; RDp[r_, q_] = mv['rd'][sel]; Kp[r_, q_] = mv['k'][sel]
        Xp = x[ws]
        if Bp > Bc: Xp = np.concatenate([Xp, np.repeat(Xp[:1], Bp - Bc, 0)])
        out = _rb_pack(p, jnp.asarray(Xp, rd), jnp.asarray(RUp), jnp.asarray(RDp), jnp.asarray(Kp), broken=(broken is True))
        lb_, r = np.asarray(out[0]), np.asarray(out[1]); res[sel] = r[r_, q_]; lB[ws] = lb_[:Bc]
        if PROGRESS is not None: PROGRESS(min((c + 1) * walker_chunk, M), M)
    t2 = time.perf_counter(); parts = []; masks = []
    for (i, m, d, mk, gidx, V, off_) in P1:
        dA = d['am'].shape[1]; dB = d['bm'].shape[2]
        r_e = np.where(V, res[np.clip(gidx + off_, 0, max(Mtot - 1, 0))], 0)
        d['r1'] = jnp.asarray(r_e[:, :dA]); d['r2'] = jnp.asarray(r_e[:, dA:].reshape(m, dA, dB)); d['logB'] = jnp.asarray(lB[i:i + m]); parts.append(d); masks.append(mk)
    out = {k_: jnp.concatenate([q[k_] for q in parts]) for k_ in parts[0]}      # a plain dict with exactly the old keys
    _UQ.clear(); _UQ['r2'] = out['r2']; _UQ['mask'] = np.concatenate(masks)     # side channel for the CNN dedup (valid while d['r2'] is this array)
    E = dA + dA * dB; nvalid = int(sum(V_.sum() for (_, _, _, _, _, V_, _) in P1))
    Kbs = [int(max(QUANT, QUANT * int(np.ceil(u[order[c * walker_chunk:(c + 1) * walker_chunk]].max() / QUANT)))) for c in range((M + walker_chunk - 1) // walker_chunk)]
    _stat('walkers', M); _stat('entries_old_padded', M * E); _stat('entries_valid', nvalid); _stat('entries_evaluated', Mtot)
    _stat('entries_packed_padded', int(sum(min(walker_chunk, M) * Kb_ for Kb_ in Kbs)))
    _stat('buckets', len(nsh)); _stat('sec_bookkeeping', (t1 - t0) + (time.perf_counter() - t2)); _stat('sec_lemma', t2 - t1)
    return out


@functools.partial(jax.jit, static_argnames=('broken',))
def _rb_pack(p, X, ru, rd_, k, broken=False):
    """_ratios_batch (the old kernel, unchanged arithmetic) on a packed move list: X (Bc, N), ru / rd_ (Bc, K, 2), k (Bc, K); K = bucket size."""
    return _ratios_batch(p, X, ru, rd_, k, 1, broken)


def _call_sign_keys(sign_fn, keys, N, chunk):
    """sign_fn on the configurations given by uint64 keys (bit s set = spin +1), fixed-size chunks (one compile per chunk shape)."""
    nat = getattr(sign_fn, 'jax_native', False); U = len(keys); outs = []; sh = np.arange(N, dtype=np.uint64)[None]
    for i in range(0, U, chunk):
        kb = keys[i:i + chunk]; m = len(kb)
        tgt = chunk if U > chunk else int(max(256, 1 << int(np.ceil(np.log2(max(m, 1))))))
        if m < tgt: kb = np.concatenate([kb, np.repeat(kb[:1], tgt - m)])
        cfg = (((kb[:, None] >> sh) & np.uint64(1)).astype(np.int8) * 2 - 1).astype(np.int8)
        o = sign_fn(jnp.asarray(cfg) if nat else cfg)
        outs.append((jnp.asarray(o) if nat else np.asarray(o))[:m])
    if nat: return jnp.concatenate(outs).astype(jnp.float64)
    out = np.concatenate(outs).astype(np.float64); assert np.all(np.abs(out) == 1), 'sign_fn must return +-1'
    return jnp.asarray(out)


def _signs_dedup(sign_fn, Xn, d, at_y, chunk, broken):
    """sx (M,), sy (M, dA), sz (M, dA, dB) with ONE sign_fn call per distinct configuration among x, all y_a, all z_ab of the whole batch.
    Entries that are not valid endpoints are z = x, as in the padded path (so every entry equals the old path's, valid or not)."""
    t0 = time.perf_counter(); M, N = Xn.shape; dA = d['am'].shape[1]
    xkey = _cfg_keys(Xn)
    if at_y:
        mask = _UQ['mask'] if (_UQ.get('r2') is not None and d.get('r2') is _UQ['r2']) else np.asarray(_flip_masks(d['upa'], d['dna'], d['am'], d['ru'], d['rd'], d['k'], d['bm']))
    else: mask = np.asarray(_flip_masks0(d['upa'], d['dna'], d['am']))
    allk = np.concatenate([xkey[:, None], xkey[:, None] ^ mask], 1)             # (M, 1 + E)
    if broken == 'pair':                                                     # BROKEN: group by the bond pair, not the configuration
        assert at_y; pk = np.asarray(_pair_keys(d['a'], d['b'], PP2_NB(N) // 2))
        gid, rb, re_, cnt = _row_unique(np.concatenate([np.full((M, 1), 2 ** 62, np.uint64), pk], 1), np.ones(allk.shape, bool))
        U = allk[rb, re_]; emap = (np.cumsum(cnt) - cnt)[:, None] + gid
    else:
        if broken == 'cnnkey': allk = allk & ~(np.uint64(3) << np.uint64(N - 2))   # BROKEN: sites N-2, N-1 dropped from the key
        gid, rb, re_, cnt = _row_unique(allk, np.ones(allk.shape, bool))
        U, inv_u = np.unique(allk[rb, re_], return_inverse=True); emap = inv_u[(np.cumsum(cnt) - cnt)[:, None] + gid]
        if broken == 'cnnkey': U = U                                         # (the representative key is the masked one: wrong configs)
    t1 = time.perf_counter()
    sU = _call_sign_keys(sign_fn, U, N, chunk)
    t2 = time.perf_counter()
    s = sU[jnp.asarray(emap)] if isinstance(sU, jnp.ndarray) else sU[emap]
    s = jnp.asarray(s)
    sx = s[:, 0]; sy = s[:, 1:1 + dA]; sz = s[:, 1 + dA:].reshape(M, dA, -1) if at_y else None
    _stat('cnn_calls_dedup', len(U)); _stat('cnn_calls_total_valid', M * (1 + dA) + (int(np.asarray(d['bm']).sum()) if at_y else 0))
    _stat('cnn_calls_old_padded', M * allk.shape[1]); _stat('sec_cnn_bookkeeping', (t1 - t0) + (time.perf_counter() - t2)); _stat('sec_cnn', t2 - t1)
    return sx, sy, sz


def PP2_NB(N): return lattice(int(round(np.sqrt(N))), 'none').NB


def make_sign_fn(size='L', seed=0, net=None, L=8, bias=0.0):
    """sign8.SignNet as a jax_native sign_fn for _call_sign / _call_sign_keys (f32 network, +-1 out; sign of logit - bias).  size 'L' = 64 ch x 6 layers.
    (A random-weight net has a nearly constant sign; the tests pass bias = the median logit so that the signs are balanced.)"""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sign8
    net = net or sign8.SignNet(L=L, size=size, seed=seed)
    f = jax.jit(lambda x: jnp.where(net.jax_logit(net.params, x.astype(jnp.float32)) - jnp.float32(bias) >= 0, 1, -1))
    def sign_fn(x): return f(x)
    sign_fn.jax_native = True; sign_fn.net = net; sign_fn.bias = bias
    return sign_fn


# ------------------------------------------------------------------------------------------ lemma-path 1-PP phase sign
class PhaseSign:
    """The sign angle's stored-sign class (s2b_train_m.py 'pp' arm; theta = [Re f, Im f, theta0]) evaluated through the determinant-ratio lemma.
    S(x) = M(x) sgn l(x),  l = Re(e^{-i theta0} P)/|P|, P = B of the PP2 (f complex, rbm 'none', no P_2, full projection x flip).
    endpoints(x, d, at_y): (sx, sy, sz) for the amplitude evaluation d (two_swap_ratios / _one_as_two output of the SAME x and dmax), margins on request.
    __call__(cfg): plain sign_fn on full configurations (direct slogdet of every image, via logB), for non-lemma callers.
    guard (default 1e-8, 0 = off): GUARD against the lemma path's conditioning error (max |dm| 7.6e-12 at 6x6, 4.4e-11 at 8x8 vs the direct code): every valid endpoint with
            |lemma margin| < guard is recomputed with the direct phase code (logB of the endpoint configuration); n_checked / n_fallback / fallback_rate count them
            (also STATS['phase_guard_checked' / 'phase_guard_fallbacks']).  Safe iff the lemma error stays below the threshold.  _inject: test hook, see _guard.
    broken: test hooks (a deliberately wrong version must fail the checks): 'parity' (slot parity P + Q of the lemma sign dropped), 'flip' (spin-flip images dropped),
            'phase' (e^{+i theta0}), 'marshall' (M = +1), 'marshall_mask' (A-sublattice mask of the wrong sublattice parity: sites x even)."""
    lemma_phase = True

    def __init__(self, f, theta0, L=None, proj='full', marshall=True, broken=None, guard=1e-8, _inject=0.0):
        f = np.asarray(f); N = f.shape[0]; L = L or int(round(np.sqrt(N))); assert L * L == N
        assert broken in (None, 'parity', 'flip', 'phase', 'marshall', 'marshall_mask'), broken
        self.broken = broken; self.marshall = marshall and broken != 'marshall'; self.theta0 = float(theta0); self.L = L; self.N = N
        self.params = PP2(f.astype(np.complex128), L=L, proj='full_noflip' if broken == 'flip' else proj, rbm='none')
        xs = np.arange(N) % L; ys = np.arange(N) // L
        a = ((xs % 2) == 0) if broken == 'marshall_mask' else (((xs + ys) % 2) == 0)
        self.amask = np.uint64(sum(1 << int(i) for i in np.nonzero(a)[0]))
        self._sgn_theta = -1.0 if broken != 'phase' else 1.0
        self.guard = float(guard); self._inject = float(_inject); self.reset_guard()

    @staticmethod
    def from_theta(theta, L=6, **kw):
        th = np.asarray(theta, np.float64); N = L * L; assert th.size == 2 * N * N + 1, th.size
        return PhaseSign((th[:N * N] + 1j * th[N * N:2 * N * N]).reshape(N, N), th[-1], L=L, **kw)

    def _marshall_x(self, X):
        A = jnp.asarray(np.array([(i % self.L + i // self.L) % 2 == 0 for i in range(self.N)]))
        n = jnp.sum(jnp.where(A[None, :], jnp.asarray(X) > 0, False), 1)
        return jnp.where(n % 2 == 0, 1.0, -1.0)

    def _margins(self, X, lP, r1, r2, at_y, am, bm):
        """lemma margins l at x (B,), the one-swap endpoints (B, dA) and the two-swap endpoints (B, dA, dB) (invalid entries: l(x))."""
        u0 = jnp.exp(1j * (self._sgn_theta * self.theta0 + jnp.imag(lP)))          # e^{-i theta0} P/|P|  (unit modulus)
        def marg(r, valid):                                                        # r = P(endpoint)/P(x); margin Re(e^{-i th0} P_e)/|P_e|
            ar = jnp.abs(r); m = jnp.real(u0.reshape((-1,) + (1,) * (r.ndim - 1)) * r) / jnp.maximum(ar, 1e-300)
            return jnp.where(valid, m, jnp.real(u0).reshape((-1,) + (1,) * (r.ndim - 1)))
        return jnp.real(u0), marg(r1, am), (marg(r2, bm) if at_y else None)

    def _guard(self, X, masks, mx, my, mz, am, bm):
        """GUARD: every valid endpoint (and x) with |lemma margin| < self.guard is recomputed with the direct phase code (logB: slogdet of every image of the endpoint
        configuration, margin cos(arg P - theta0)); returns the margins with those entries replaced.  Counts checked / fallback entries (self.n_checked, self.n_fallback)."""
        X = np.asarray(X); M_, N = X.shape; dA = my.shape[1]
        mxn = np.array(mx); myn = np.array(my); mzn = None if mz is None else np.array(mz)
        mk = np.asarray(masks); amn = np.asarray(am); bmn = None if bm is None else np.asarray(bm)
        thr = self.guard; inj = self._inject
        cols = [mxn[:, None], myn] + ([] if mzn is None else [mzn.reshape(M_, -1)])
        allm = np.concatenate(cols, 1); valid = np.concatenate([np.ones((M_, 1), bool), amn] + ([] if bmn is None else [bmn.reshape(M_, -1)]), 1)
        mkx = np.concatenate([np.zeros((M_, 1), np.uint64), mk], 1)                    # flip mask of every entry (x itself: 0)
        self.n_checked += int(valid.sum()); _stat('phase_guard_checked', int(valid.sum()))
        if inj: allm = np.where(valid & (np.abs(allm) < max(thr, 1e-8)), allm - inj * np.sign(np.where(allm == 0, 1.0, allm)), allm)   # TEST HOOK: perturbation toward / across zero on near-zero entries
        if thr > 0:
            w, e = np.nonzero(valid & (np.abs(allm) < thr))
            if len(w):
                sh = np.arange(N, dtype=np.uint64)[None]
                flip = ((mkx[w, e][:, None] >> sh) & np.uint64(1)).astype(bool)
                cfg = np.where(flip, -X[w], X[w])
                lPd = np.asarray(logB(self.params, cfg, 'f64'))
                allm[w, e] = np.cos(np.imag(lPd) + self._sgn_theta * self.theta0)
                self.n_fallback += len(w); _stat('phase_guard_fallbacks', len(w))
        mx2 = allm[:, 0]; my2 = allm[:, 1:1 + dA]; mz2 = None if mzn is None else allm[:, 1 + dA:].reshape(mzn.shape)
        return jnp.asarray(mx2), jnp.asarray(my2), (None if mz2 is None else jnp.asarray(mz2))

    @property
    def fallback_rate(self):
        """fraction of checked endpoints (valid + x) that fell back to the direct phase code since the last reset_guard()."""
        return self.n_fallback / max(self.n_checked, 1)

    def reset_guard(self):
        self.n_checked = 0; self.n_fallback = 0

    def _signs(self, X, lP, r1, r2, masks, at_y, margins, am, bm):
        """lP (B,) complex log P(x); r1 (B, dA) = P(y_a)/P(x) (0 where invalid); r2 (B, dA, dB) or None; masks (B, E) flip masks (E = dA [+ dA dB])."""
        mx, my, mz = self._margins(X, lP, r1, r2, at_y, am, bm)
        if self.guard > 0 or self._inject: mx, my, mz = self._guard(X, masks, mx, my, mz, am, bm)
        Mx = self._marshall_x(X) if self.marshall else jnp.ones(X.shape[0])
        if self.marshall:
            Mm = 1.0 - 2.0 * (jax.lax.population_count(masks & jnp.uint64(self.amask)) % 2).astype(jnp.float64)
        else: Mm = jnp.ones(masks.shape)
        dA = r1.shape[1]
        sgn = lambda m: jnp.where(m >= 0, 1.0, -1.0)
        sx = sgn(mx) * Mx; sy = sgn(my) * Mx[:, None] * Mm[:, :dA]
        out = [sx, sy]; mrg = [mx, my]
        if at_y:
            sz = sgn(mz) * Mx[:, None, None] * Mm[:, dA:].reshape(r2.shape); out.append(sz); mrg.append(mz)
        else: out.append(None); mrg.append(None)
        return (out, mrg) if margins else out

    def endpoints(self, x, d, at_y=True, dtype='f64', eval_path=None, walker_chunk=256, margins=False):
        t0 = time.perf_counter(); p = self.params; path = eval_path or DEFAULT_PATH; x = np.asarray(x); brk = (self.broken == 'parity')
        if at_y:
            dm = d['am'].shape[1]
            ds = two_swap_ratios(p, x, dtype, dm, walker_chunk=walker_chunk, eval_path=path, _broken=brk)
            lP = ds['logB']; r1 = ds['r1']; r2 = ds['r2']
            masks = _flip_masks(d['upa'], d['dna'], d['am'], d['ru'], d['rd'], d['k'], d['bm'])
        else:
            lP, r1 = one_swap_ratios(p, x, dtype, walker_chunk=walker_chunk, _broken=brk, return_logB=True); r2 = None
            masks = _flip_masks0(d['upa'], d['dna'], d['am'])
        res = self._signs(x, lP, r1, r2, masks, at_y, margins, d['am'], d['bm'] if at_y else None)
        jax.block_until_ready(res)
        _stat('sec_phase_sign', time.perf_counter() - t0); _stat('phase_sign_calls', 1)
        return res

    def __call__(self, cfg):
        cfg = np.asarray(cfg); lP = logB(self.params, cfg, 'f64')
        m = jnp.cos(jnp.imag(lP) + self._sgn_theta * self.theta0)
        M = self._marshall_x(cfg) if self.marshall else 1.0
        return np.asarray(jnp.where(m >= 0, 1.0, -1.0) * M).astype(np.int8)


def phase_sign_from_theta(theta, L=6, **kw):
    """PhaseSign from the sign angle's theta.npy ([Re f (L^2 x L^2), Im f, theta0])."""
    return PhaseSign.from_theta(theta, L, **kw)


# ------------------------------------------------------------------------------------------ f_J and the augmented guide
def _hxx(lat, X):
    return 0.25 * jnp.sum(jnp.asarray(lat.JB) * X[..., lat.BI] * X[..., lat.BJ], -1)


def _call_sign(sign_fn, cfg, chunk):
    """sign_fn on cfg (M, N) in fixed-size chunks (last one padded: one compile).  A sign_fn with attribute jax_native = True gets
    device arrays and its output stays on the device; any other sign_fn gets numpy arrays."""
    nat = getattr(sign_fn, 'jax_native', False); M = cfg.shape[0]; outs = []
    cfg = cfg if nat else np.asarray(cfg)
    for i in range(0, M, chunk):
        blk = cfg[i:i + chunk]; m = blk.shape[0]
        if m < chunk and M > chunk:
            blk = (jnp if nat else np).concatenate([blk, (jnp if nat else np).repeat(blk[:1], chunk - m, 0)])
        o = sign_fn(blk)
        outs.append((jnp.asarray(o) if nat else np.asarray(o))[:m])
    if nat: return jnp.concatenate(outs).astype(jnp.float64)
    out = np.concatenate(outs).astype(np.float64); assert np.all(np.abs(out) == 1), 'sign_fn must return +-1'
    return jnp.asarray(out)


@functools.partial(jax.jit, static_argnames=('with_z',))
def _configs(X, d, with_z=True):
    """y_a configs (B, dA, N) and z_ab configs (B, dA, dB, N) (None unless with_z), int8."""
    Bn, N = X.shape; Xi = X.astype(jnp.int8)
    def apply(x, ru, rd_, k):
        msk = jnp.stack([k >= 1, k >= 2], -1)
        def one(ru1, rd1, m1):
            z = x
            z = z.at[jnp.where(m1, ru1, N)].set(-1, mode='drop'); z = z.at[jnp.where(m1, rd1, N)].set(1, mode='drop')
            return z
        return jax.vmap(one)(ru.reshape(-1, 2), rd_.reshape(-1, 2), msk.reshape(-1, 2))
    Y = jax.vmap(lambda x, u, dn, am: apply(x, jnp.stack([u, u], -1), jnp.stack([dn, dn], -1), am.astype(jnp.int32)))(
        Xi, d['upa'], d['dna'], d['am'])
    if not with_z: return Y.reshape(Bn, -1, N), None
    Z = jax.vmap(apply)(Xi, d['ru'], d['rd'], d['k'])
    return Y.reshape(Bn, -1, N), Z.reshape(Bn, d['k'].shape[1], d['k'].shape[2], N)


def _one_as_two(params, x, dtype, walker_chunk):
    """the a-part of two_swap_ratios' output from one_swap_ratios alone (all NB bonds, am = anti-aligned): for f_J at x only."""
    lat = params.lat; lB, r = one_swap_ratios(params, x, dtype, walker_chunk=walker_chunk, return_logB=True)
    X = jnp.asarray(x); BI = jnp.asarray(lat.BI); BJ = jnp.asarray(lat.BJ); am = X[:, BI] != X[:, BJ]
    upa = jnp.where(X[:, BI] > 0, BI, BJ); dna = jnp.where(X[:, BI] > 0, BJ, BI)
    a = jnp.where(am, jnp.broadcast_to(jnp.arange(lat.NB), am.shape), -1)
    return dict(logB=lB, a=a, am=am, r1=r, upa=upa, dna=dna)


def _fj_parts(lat, X, d, sx, sy, sz, EP, at_y, broken=False):
    HWB = jnp.asarray(lat.HW)
    a = d['a']; am = d['am']; hwa = jnp.where(am, HWB[jnp.maximum(a, 0)], 0.0)
    ra = jnp.abs(d['r1'])
    kept = (sx[:, None] * sy < 0) if not broken else (sx[:, None] * sy > 0)
    SK = jnp.sum(jnp.where(am & kept, hwa * ra, 0.0), 1); VA = jnp.sum(jnp.where(am & ~kept, hwa * ra, 0.0), 1)
    fx = jnp.log(SK) - jnp.log(_hxx(lat, X) + VA - EP)
    if not at_y: return fx, None
    b = d['b']; bm = d['bm']; hwb = jnp.where(bm, HWB[jnp.maximum(b, 0)], 0.0)
    rb = jnp.abs(d['r2']) / jnp.where(am, ra, 1.0)[:, :, None]
    keptb = sy[:, :, None] * sz < 0
    SKy = jnp.sum(jnp.where(bm & keptb, hwb * rb, 0.0), 2); VAy = jnp.sum(jnp.where(bm & ~keptb, hwb * rb, 0.0), 2)
    return fx, (SKy, VAy)


def _apply_mode(f, mode, c, f0, node):
    fin = jnp.isfinite(f)
    if mode == 'old': return c * jnp.where(fin, f, 0.0)
    if mode == 'a1': return jnp.where(fin, c * f, node)
    if mode == 'fix':
        assert f0 is not None, "mode 'fix' needs f0"
        return jnp.where(fin, c * jnp.sign(f) * jnp.minimum(jnp.abs(f), f0), node)
    raise ValueError(mode)


def fJ_fixed(params, x, sign_fn, EP, c, f0=None, mode='old', at_y=True, dtype='f64', dmax=None, img_chunk=1,
             sign_chunk=16384, node=NODE_ADD, _broken=False, _two=None, walker_chunk=256, eval_path=None, cnn_dedup=None, _broken_save=None):
    """f_J (r5_bare.fJ_of arithmetic, per configuration instead of per orbit row) at x and at every neighbour y_a, and the fixed add."""
    lat = params.lat; path = eval_path or DEFAULT_PATH
    if cnn_dedup is None: cnn_dedup = (path == 'dedup')
    x = np.asarray(x); X = jnp.asarray(x, jnp.float64 if dtype == 'f64' else jnp.float32)
    if _two is not None: d = _two
    elif at_y: d = two_swap_ratios(params, x, dtype, dmax, img_chunk, walker_chunk, eval_path=path, _broken=_broken_save or False)
    else: d = _one_as_two(params, x, dtype, walker_chunk)
    lemma_sign = getattr(sign_fn, 'lemma_phase', False)
    Y, Z = _configs(X, {k: d[k] for k in ('upa', 'dna', 'am', 'ru', 'rd', 'k') if k in d}, with_z=at_y and not cnn_dedup and not lemma_sign)
    if lemma_sign: sx, sy, sz = sign_fn.endpoints(x, d, at_y, dtype, path, walker_chunk)
    elif cnn_dedup: sx, sy, sz = _signs_dedup(sign_fn, np.asarray(x), d, at_y, sign_chunk, _broken_save)
    else:
        sx = _call_sign(sign_fn, X.astype(jnp.int8), sign_chunk)
        sy = _call_sign(sign_fn, Y.reshape(-1, lat.N), sign_chunk).reshape(Y.shape[:2])
        sz = _call_sign(sign_fn, Z.reshape(-1, lat.N), sign_chunk).reshape(Z.shape[:3]) if at_y else None
    rdt = jnp.real(d['r1']).dtype
    fx, py = _fj_parts(lat, X.astype(rdt), d, sx.astype(rdt), sy.astype(rdt), None if sz is None else sz.astype(rdt), EP, at_y, _broken)
    out = dict(d, fx=fx, add_x=_apply_mode(fx, mode, c, f0, node), sx=sx, sy=sy, sz=sz, Y=Y)
    if at_y:
        SKy, VAy = py
        Hy = _hxx(lat, Y.astype(rdt))
        fy = jnp.log(SKy) - jnp.log(Hy + VAy - EP)
        fy = jnp.where(d['am'], fy, jnp.nan)
        out.update(fy=fy, add_y=_apply_mode(fy, mode, c, f0, node))
    return out


def aug_ratios(params, x, sign_fn, EP, c, f0=None, mode='fix', dtype='f64', dmax=None, img_chunk=1, sign_chunk=16384, node=NODE_ADD,
               walker_chunk=256, eval_path=None, cnn_dedup=None, _broken_save=None):
    """G(y_b)/G(x) for every bond b (B, NB), G = s_G |B| e^{add}; 0 for aligned bonds."""
    lat = params.lat
    o = fJ_fixed(params, x, sign_fn, EP, c, f0, mode, True, dtype, dmax, img_chunk, sign_chunk, node, walker_chunk=walker_chunk,
                 eval_path=eval_path, cnn_dedup=cnn_dedup, _broken_save=_broken_save)
    ga = o['sx'][:, None] * o['sy'] * jnp.abs(o['r1']) * jnp.exp(o['add_y'] - o['add_x'][:, None])
    ga = jnp.where(o['am'], ga, 0.0)
    Bn = ga.shape[0]
    full = jnp.zeros((Bn, lat.NB + 1), ga.dtype).at[jnp.arange(Bn)[:, None], jnp.where(o['am'], o['a'], lat.NB)].set(ga)[:, :lat.NB]
    o['ratio'] = full; o['logG_x'] = jnp.real(o['logB']) + o['add_x']
    return o
