"""Exact symmetric-sector machinery for the 6x6 J1-J2 model (J2/J1 = 0.5) on one GPU (JAX, float64).

Sector: k = 0, A1 of C4v, spin-flip even; D = 15,804,956 orbit representatives (sorted uint64 = psi0 table order).
A symmetric state is stored as a per-configuration log-amplitude la_r and a sign s_r on the reps;
its sector component is v_r = s_r exp(la_r) sqrt(n_r)  (n_r = orbit size).  Everything below is exact
(no sampling): energies, Krylov sign step with the energy-optimal threshold, lattice FN ground state.

Conventions identical to krylov_sign_structure/experiments/closed_fn_krylov_sym6x6.py (CPU reference):
  FN guide (a, s): w_r = max(a_r, 1e-15) sqrt(n_r) with a normalised (sum n a^2 = 1);
     D_FN = (1/w) * 1/2 [H w + s H (s w)]      (same-sign elements incl. the diagonal, weighted by w_r'/w_r)
     H_FN x = D_FN x - 1/2 [H x - s H (s x)]   (opposite-sign elements kept)
  Krylov: r = (H s v)/(s v); s' = s sgn(T - r), T energy-optimal over all cuts of sorted r (ties grouped).
"""
import json, os, time
from functools import partial
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import scipy.linalg

L = 6; N = 36; J2 = 0.5


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


# ------------------------------------------------------------------------------------------ lattice / group
def _site(x, y):
    return (x % L) + L * (y % L)


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


def byte_tables(perms):
    nb = (N + 7) // 8
    T = np.zeros((perms.shape[0], nb, 256), np.uint64)
    vals = np.arange(256, dtype=np.uint64)
    for b in range(nb):
        for k in range(8):
            i = 8 * b + k
            if i >= N: break
            bit = (vals >> np.uint64(k)) & np.uint64(1)
            T[:, b, :] |= bit[None, :] << perms[:, i].astype(np.uint64)[:, None]
    return T


def bonds():
    nn_, nnn = [], []
    for y in range(L):
        for x in range(L):
            i = _site(x, y)
            nn_ += [(i, _site(x + 1, y)), (i, _site(x, y + 1))]
            nnn += [(i, _site(x + 1, y + 1)), (i, _site(x + 1, y - 1))]
    return nn_, nnn


NN, NNN = bonds()
BI = np.array([b[0] for b in NN + NNN], np.uint64); BJ = np.array([b[1] for b in NN + NNN], np.uint64)
JB = np.array([1.0] * len(NN) + [J2] * len(NNN))
MASKS = (np.uint64(1) << BI) | (np.uint64(1) << BJ)
NBOND = len(JB)
FULL = np.uint64((1 << N) - 1)


# ------------------------------------------------------------------------------------------ sector
# Compact upper-triangular coded Hamiltonian (st6_build_upper.py):
#   H = diag(sqrt n) C diag(1/sqrt n), C_rr' = codes/8, only col >= row stored, row = cumsum(inc).
#   (H x)_r = sqrt(n_r) sum_{r'>=r} C_rr' x_r'/sqrt(n_r') + (1/sqrt n_r) sum_{r''<r} C_r''r sqrt(n_r'') x_r''
CH = 1 << int(os.environ.get("ST6_CHLOG", "23"))


@partial(jax.jit, donate_argnums=(0, 1))
def _up_chunk(acc1, acc2, rbase, inc, idx, cod, X, sq):
    row = rbase + jnp.cumsum(inc.astype(jnp.int32))
    c = cod.astype(jnp.float64) * 0.125
    acc1 = acc1.at[row].add((c / sq[idx])[:, None] * X[idx], indices_are_sorted=True)
    c2 = jnp.where(idx != row, c * sq[row], 0.0)
    acc2 = acc2.at[idx].add(c2[:, None] * X[row])
    return acc1, acc2


@partial(jax.jit, donate_argnums=(0,))
def _kry_chunk(delta, rbase, inc, idx, cod, gid, u, sq):
    row = rbase + jnp.cumsum(inc.astype(jnp.int32))
    A = cod.astype(jnp.float64) * 0.125 * sq[row] / sq[idx] * u[row] * u[idx]
    gi = gid[row]; gj = gid[idx]
    A = jnp.where((idx != row) & (gi != gj), A, 0.0)
    delta = delta.at[jnp.minimum(gi, gj)].add(-4.0 * A)
    return delta.at[jnp.maximum(gi, gj)].add(4.0 * A)


class Sector:
    def __init__(self, csr_dir, table):
        t0 = time.time()
        meta = json.load(open(os.path.join(csr_dir, 'meta.json')))
        z = np.load(table)
        n = np.load(os.path.join(csr_dir, 'norb.npy')).astype(np.float64)
        assert np.array_equal(n, z['orbit'].astype(np.float64))
        ui = np.load(os.path.join(csr_dir, 'up_indices.npy'), mmap_mode='r')
        uc = np.load(os.path.join(csr_dir, 'up_codes.npy'), mmap_mode='r')
        inc = np.load(os.path.join(csr_dir, 'up_inc.npy'), mmap_mode='r')
        self._setup(n, ui, uc, inc, z['reps'], z['amp'], float(z['E0']), t0)

    def _setup(self, n, ui, uc, inc, reps, amp, E0, t0=None, ch=None):
        t0 = time.time() if t0 is None else t0
        CHs = CH if ch is None else ch
        self.D = D = n.shape[0]; self.nnz_u = nnz = ui.shape[0]; self.E0 = E0
        self.chunks = []; self.chunks_host = []
        rprev = 0
        for p0 in range(0, nnz, CHs):
            p1 = min(nnz, p0 + CHs)
            i_ = np.zeros(CHs, np.int32); c_ = np.zeros(CHs, uc.dtype); n_ = np.zeros(CHs, np.uint8)
            i_[:p1 - p0] = ui[p0:p1]; c_[:p1 - p0] = uc[p0:p1]; n_[:p1 - p0] = inc[p0:p1]
            if p1 - p0 < CHs: i_[p1 - p0:] = i_[p1 - p0 - 1]   # padding: code 0, same row
            self.chunks_host.append((np.int32(rprev), n_, i_, c_))
            self.chunks.append(tuple(jnp.asarray(q) for q in self.chunks_host[-1]))
            rprev += int(n_[:p1 - p0].astype(np.int64).sum())
        assert rprev <= D - 1
        log(f'[sector] D={D} nnz_upper={nnz} chunks={len(self.chunks)} last row {rprev} load {time.time()-t0:.0f}s')
        self.n = jnp.asarray(n); self.sqrt_n = jnp.sqrt(self.n)
        self.reps_np = reps
        self.reps = jnp.asarray(reps)
        self.v0 = jnp.asarray(amp * np.sqrt(n))
        self.v0 = self.v0 / jnp.linalg.norm(self.v0)
        self.s0 = jnp.where(self.v0 >= 0, 1.0, -1.0).astype(jnp.float32)   # signs stored as float32 (memory)
        self.la0 = jnp.log(jnp.maximum(jnp.abs(jnp.asarray(amp)), 1e-300))
        self.p0 = self.v0 ** 2
        self.T = jnp.asarray(byte_tables(space_group()))        # (288, 5, 256)
        self.nmv = 0
        log(f'[sector] on device {time.time()-t0:.0f}s')

    def offload(self):
        """free the device copy of H (4.2 GB) while a network is trained; reload() restores it."""
        self.chunks = None

    def reload(self):
        if self.chunks is None:
            self.chunks = [tuple(jnp.asarray(q) for q in c) for c in self.chunks_host]

    def Hm(self, X):
        """H applied to the columns of X (D,) or (D, k)."""
        self.reload()
        one = X.ndim == 1
        if one: X = X[:, None]
        self.nmv += X.shape[1]
        a1 = jnp.zeros_like(X); a2 = jnp.zeros_like(X)
        for k, (rb, inc, idx, cod) in enumerate(self.chunks):
            a1, a2 = _up_chunk(a1, a2, rb, inc, idx, cod, X, self.sqrt_n)
            a1.block_until_ready()      # one chunk in flight (bounds device memory on an 11 GB card)
        Y = self.sqrt_n[:, None] * a1 + a2 / self.sqrt_n[:, None]
        return Y[:, 0] if one else Y

    # ----------------------------------------------------------------- basic ops
    def H(self, x):
        return self.Hm(x)

    def H2(self, x, y):
        Y = self.Hm(jnp.stack([x, y], 1))
        return Y[:, 0], Y[:, 1]

    def vec(self, la, s):
        """normalised sector vector from per-config log-amplitude la and sign s."""
        la = la - jnp.max(la)
        v = s * jnp.exp(la) * self.sqrt_n
        return v / jnp.linalg.norm(v)

    def energy(self, v):
        return float(v @ self.H(v) / (v @ v))

    def score(self, la, s, tag=''):
        v = self.vec(la, s)
        E = self.energy(v)
        ov = float(jnp.sum(self.p0 * s * self.s0))
        g = 1.0 if ov >= 0 else -1.0
        ws = max(0.0, (1 - abs(ov)) / 2)
        d = la - self.la0
        mu = jnp.sum(self.p0 * d)
        sd = float(jnp.sqrt(jnp.sum(self.p0 * (d - mu) ** 2)))
        fid = float((jnp.abs(v) @ jnp.abs(self.v0)) ** 2)
        return dict(E=E, E_site=E / N, dE_site=(E - self.E0) / N, w_s=ws, std_dlog=sd, amp_infid=1 - fid)

    # ----------------------------------------------------------------- FN
    def fn_op(self, la_g, s):
        """x -> H_FN[a_g, s] x  (closure holding the FN diagonal)."""
        la_g = la_g - jnp.max(la_g)
        a = jnp.exp(la_g); a = a / jnp.sqrt(jnp.sum(self.n * a * a)); a = jnp.maximum(a, 1e-15)
        w = a * self.sqrt_n; del a
        h1, h2 = self.H2(w, s * w)
        Dg = 0.5 * (h1 + s * h2) / w
        del h1, h2, w

        def A(x):
            h1, h2 = self.H2(x, s * x)
            return Dg * x - 0.5 * (h1 - s * h2)
        return A

    def fn_rayleigh(self, la_g, s, la_b):
        """exact frozen-FN energy <b|H_FN[a_g, s]|b>/<b|b> of a trial amplitude b (per-config log la_b)."""
        la_g = la_g - jnp.max(la_g)
        a = jnp.exp(la_g); a = a / jnp.sqrt(jnp.sum(self.n * a * a)); a = jnp.maximum(a, 1e-15)
        w = a * self.sqrt_n; del a
        h1, h2 = self.H2(w, s * w)
        Dg = 0.5 * (h1 + s * h2) / w
        del h1, h2, w
        b = self.vec(la_b, jnp.ones_like(la_b))
        h1, h2 = self.H2(b, s * b)
        return float(b @ (Dg * b - 0.5 * (h1 - s * h2)))

    def fn_solve(self, la, s, tol=1e-10, maxit=400, x0=None, verbose=False):
        """Perron ground state of H_FN[a, s].  Returns (E_FN, v_FN >= 0 normalised, info)."""
        t0 = time.time(); nm0 = self.nmv
        la = la - jnp.max(la)
        a = jnp.exp(la); a = a / jnp.sqrt(jnp.sum(self.n * a * a))
        a = jnp.maximum(a, 1e-15)
        w = a * self.sqrt_n
        del a, la
        h1, h2 = self.H2(w, s * w)
        Dg = 0.5 * (h1 + s * h2) / w
        del h1, h2

        def A(x):
            h1, h2 = self.H2(x, s * x)
            return Dg * x - 0.5 * (h1 - s * h2)

        x = (w if x0 is None else x0); x = x / jnp.linalg.norm(x)
        del w
        Ax = A(x); th = float(x @ Ax)
        den = jnp.maximum(Dg - th, 1.0)
        P = None; AP = None
        it = 0; rn = np.inf
        for it in range(1, maxit + 1):
            r = Ax - th * x
            rn = float(jnp.linalg.norm(r))
            if verbose and it % 10 == 1: log(f'  lobpcg it {it} theta {th:.12f} res {rn:.3e}')
            if rn <= tol * abs(th): break
            z = r / den
            z = z - (x @ z) * x
            if P is not None: z = z - (P @ z) * P / (P @ P)
            z = z / jnp.linalg.norm(z)
            Az = A(z)
            Sb = [x, z] + ([P] if P is not None else [])
            ASb = [Ax, Az] + ([AP] if P is not None else [])
            m = len(Sb)
            del r
            GA = np.array([[float(Sb[i] @ ASb[j]) for j in range(m)] for i in range(m)])
            GB = np.array([[float(Sb[i] @ Sb[j]) for j in range(m)] for i in range(m)])
            GA = 0.5 * (GA + GA.T)
            try:
                ev, ec = scipy.linalg.eigh(GA, GB)
            except Exception:
                P = None; AP = None; continue
            c = ec[:, 0]
            xn = sum(c[i] * Sb[i] for i in range(m)); Axn = sum(c[i] * ASb[i] for i in range(m))
            del x, Ax, P, AP
            P = sum(c[i] * Sb[i] for i in range(1, m)); AP = sum(c[i] * ASb[i] for i in range(1, m))
            del Sb, ASb, z, Az
            nr = float(jnp.linalg.norm(xn)); x = xn / nr; Ax = Axn / nr
            pn = float(jnp.linalg.norm(P))
            if pn > 0: P = P / pn; AP = AP / pn
            th = float(x @ Ax)
        sgn = 1.0 if float(jnp.sum(x)) >= 0 else -1.0
        x = x * sgn
        negw = float(jnp.sum(jnp.where(x < 0, x * x, 0.0)))
        u = jnp.abs(x); u = u / jnp.linalg.norm(u)
        E = float(u @ A(u))
        info = dict(E_FN=E, E_FN_site=E / N, dE_FN_site=(E - self.E0) / N, lobpcg_it=it, res=rn / abs(th),
                    neg_weight=negw, matvecs=self.nmv - nm0, sec=time.time() - t0)
        return E, u, info

    # ----------------------------------------------------------------- Krylov sign step
    def krylov(self, v, s, atol=1e-11, rtol=1e-11):
        """v: amplitude part (>=0 sector vector), s: current sign.  Returns (s', E(s'), info)."""
        t0 = time.time()
        v = jnp.abs(v) / jnp.linalg.norm(v)
        psi = v * s
        Hpsi = self.H(psi)
        r = Hpsi / jnp.where(jnp.abs(psi) > 1e-300, psi, 1.0)
        order = jnp.argsort(r)
        rs = r[order]
        scale = jnp.maximum(jnp.maximum(jnp.abs(rs[1:]), jnp.abs(rs[:-1])), 1.0)
        br = jnp.concatenate([jnp.ones(1, bool), jnp.abs(rs[1:] - rs[:-1]) > (atol + rtol * scale)])
        gs = jnp.cumsum(br.astype(jnp.int32)) - 1
        gid = jnp.zeros(self.D, jnp.int32).at[order].set(gs)
        G = int(gs[-1]) + 1
        u = -s * v
        delta = jnp.zeros(self.D, jnp.float64)
        e0 = float(u @ self.H(u))
        for k, (rb, inc, idx, cod) in enumerate(self.chunks):
            delta = _kry_chunk(delta, rb, inc, idx, cod, gid, u, self.sqrt_n)
            delta.block_until_ready()
        cand = np.concatenate([[e0], e0 + np.cumsum(np.asarray(delta[:G]))])
        kbest = int(np.argmin(cand)) - 1
        sn = -s
        if kbest >= 0:
            sn = jnp.where(gid <= kbest, s, -s)
        if float(sn[0]) < 0: sn = -sn
        Eb = float(np.min(cand))
        # threshold value for the record
        T = float(rs[jnp.searchsorted(gs, kbest, side='right') - 1]) if kbest >= 0 else -np.inf
        return sn, Eb, dict(E_kry=Eb, E_kry_site=Eb / N, groups=G, kbest=kbest, T=T,
                            n_flip_orbits=int(jnp.sum(sn != s)), sec=time.time() - t0)



# ------------------------------------------------------------------------------------------ full-basis lookup (jit-able)
def canon(T, reps, x):
    """x: (B,) uint64 full-basis configs -> index of their orbit representative in reps."""
    img = jnp.zeros((x.shape[0], T.shape[0]), jnp.uint64)
    for b in range(5):
        byte = ((x >> jnp.uint64(8 * b)) & jnp.uint64(255)).astype(jnp.int32)
        img = img | T[:, b, :][:, byte].T
    m = jnp.minimum(jnp.min(img, axis=1), jnp.min(img ^ jnp.uint64(FULL), axis=1))
    return jnp.clip(jnp.searchsorted(reps, m), 0, reps.shape[0] - 1)


def image(T, x, g, flip):
    """apply space-group element g (B,) and spin flip (B, bool) to configs x (B,)."""
    out = jnp.zeros_like(x)
    for b in range(5):
        byte = ((x >> jnp.uint64(8 * b)) & jnp.uint64(255)).astype(jnp.int32)
        out = out | T[g, b, byte]
    return jnp.where(flip, out ^ jnp.uint64(FULL), out)
