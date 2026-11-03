"""B1 library: all one-swap ratios of one pair-product determinant F = f[U, D] (identity image only), four arms.
Bond list, sign/parity convention, and the fast-path table are COPIES of experiments/gfmc_fn_lattice/fastratio/rbmpp_lib.py
(Lattice bond construction, log_one parity, img_quant_k fast branch); the Q arm follows quantax/model/fermion_mf.py:940-976
(SingletPair.ref_forward) and calls lrux.det_lru as quantax does.  Not production code; nothing here is imported by production.
"""
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import lrux
from quantax.nn.fermion import permute_sign                     # quantax/nn/fermion.py:68 (pure function)
from quantax.model.fermion_mf import _occupied_positions         # quantax/model/fermion_mf.py:111 (pure function)
jax.config.update("jax_enable_x64", True)
PI = jnp.pi


def bonds(L, J2=0.5):
    """copy of rbmpp_lib.Lattice bond construction (J1 bonds then J2 bonds)."""
    site = lambda x, y: (x % L) + L * (y % L)
    nn_, nnn = [], []
    for y in range(L):
        for x in range(L):
            i = site(x, y)
            nn_ += [(i, site(x + 1, y)), (i, site(x, y + 1))]
            nnn += [(i, site(x + 1, y + 1)), (i, site(x + 1, y - 1))]
    bl = nn_ + nnn
    return np.array([b[0] for b in bl], np.int32), np.array([b[1] for b in bl], np.int32)


def random_configs(N, B, seed):
    """copy of rbmpp_lib.random_configs (n up, n down)."""
    rng = np.random.default_rng(seed)
    S = np.ones((B, N), np.int8); S[:, N // 2:] = -1
    for r in range(B): rng.shuffle(S[r])
    return S


class Arms:
    """closures for lattice size L and a matrix dtype (complex128 or float64)."""

    def __init__(self, L, dtype=jnp.complex128, Kc=None):
        self.L = L; self.N = N = L * L; self.n = n = N // 2
        bi, bj = bonds(L)
        self.BI = BI = jnp.asarray(bi); self.BJ = BJ = jnp.asarray(bj); self.NB = NB = len(bi)
        self.dtype = dtype; self.Kc = Kc
        cdt = jnp.complex128
        self.cdt = cdt
        arange_n = jnp.arange(n, dtype=jnp.int32)

        # ---------------------------------------------------------------- common pieces
        def lists(z):
            order = jnp.argsort(-z, stable=True)
            return order[:n], order[n:]

        def inv_count(z):                                    # copy of rbmpp_lib parity: #(down j < up i) pairs
            up = (z > 0).astype(jnp.int32); dn = 1 - up
            return jnp.sum(up * (jnp.cumsum(dn) - dn))

        def bond_geom(z):                                    # p up, q down for each bond; valid = antiparallel
            zi = z[BI]; zj = z[BJ]
            return jnp.where(zi > 0, BI, BJ), jnp.where(zi > 0, BJ, BI), zi != zj

        def pos_of(U, D):
            return jnp.zeros(N, jnp.int32).at[U].set(arange_n).at[D].set(arange_n)

        def pick(z, key):                                    # copy of make_sampler's uniform antiparallel-bond choice
            p, q, valid = bond_geom(z)
            sc = jnp.where(valid, jax.random.uniform(key, valid.shape), -1.0)
            k = jnp.argmax(sc)
            return p[k], q[k]

        def swap(z, p, q):
            return z.at[p].set(-1).at[q].set(1)

        self.lists, self.inv_count, self.bond_geom, self.pos_of, self.pick, self.swap = lists, inv_count, bond_geom, pos_of, pick, swap

        # ---------------------------------------------------------------- brute force (rbmpp_lib.log_one, no RBM)
        def log_one(f, z):
            U, D = lists(z)
            sg, ld = jnp.linalg.slogdet(f[U][:, D])
            return ld.astype(cdt) + jnp.log(sg.astype(cdt)) + 1j * PI * inv_count(z)

        def brute(f, z):
            lx = log_one(f, z)

            def nb(k):
                y = z.at[BI[k]].set(z[BJ[k]]).at[BJ[k]].set(z[BI[k]])
                return jnp.exp(log_one(f, y) - lx)
            r = jax.lax.map(nb, jnp.arange(NB))
            return jnp.where(z[BI] != z[BJ], r, 0.0)
        self.brute = brute; self.log_one = log_one

        # ---------------------------------------------------------------- R: copy of img_quant_k fast path (RBM off)
        def table_ratios(f, G, U, D, p_, q_, a_, bc):
            fD = f[D]
            Am = fD[:, D]; Cm = fD[:, U]; Bm = f[U][:, U]
            Pm = Am @ G; Qm = G @ Bm; Sm = Pm @ Bm
            ii = (bc, a_)
            rdet = Pm[ii] * Qm[ii] + G[ii] * (Cm[ii] - Sm[ii])
            return -rdet

        def R_ratios(f, z):
            U, D = lists(z)
            F = f[U][:, D]
            sg, ld = jnp.linalg.slogdet(F)
            G = jnp.linalg.inv(F)
            lg = ld.astype(cdt) + jnp.log(sg.astype(cdt)) + 1j * PI * inv_count(z)
            up = (z > 0).astype(jnp.int32); dn = 1 - up
            pos = jnp.where(up > 0, jnp.cumsum(up) - 1, jnp.cumsum(dn) - 1)
            p_, q_, valid = bond_geom(z)
            ratio = table_ratios(f, G, U, D, p_, q_, pos[p_], pos[q_]).astype(cdt)
            return lg, jnp.where(valid, ratio, 0.0)

        def H_ratios(f, G, U, D, z):                         # R's table from a stored G (no inverse)
            p_, q_, valid = bond_geom(z)
            pos = pos_of(U, D)
            ratio = table_ratios(f, G, U, D, p_, q_, pos[p_], pos[q_]).astype(cdt)
            return jnp.where(valid, ratio, 0.0)
        self.R_ratios, self.H_ratios, self.table_ratios = R_ratios, H_ratios, table_ratios

        # ---------------------------------------------------------------- Q: quantax SingletPair.ref_forward + lrux.det_lru
        def uv_one(f, U, D, p, q, swap_idx=False):
            pu = p.reshape(1); qu = q.reshape(1)
            sign_up = permute_sign(U, pu, qu)                # up hop p -> q   (ann = p, cre = q)
            sign_dn = permute_sign(D, qu, pu)                # down hop q -> p (ann = q, cre = p)
            rui = _occupied_positions(U, pu)                 # row_update_idx
            cui = _occupied_positions(D, qu)                 # col_update_idx
            row_update = f[qu][:, D] - f[pu][:, D]           # F[cre_up][:, idx_dn] - F[ann_up][:, idx_dn]
            Un = U.at[rui].set(qu)
            col_update = f[Un][:, pu] - f[Un][:, qu]         # F[idx_up][:, cre_dn] - F[idx_up][:, ann_dn]
            if swap_idx:                                     # broken control: row/column indices exchanged
                u = (row_update.T, rui); v = (col_update, cui)
            else:
                u = (row_update.T, cui); v = (col_update, rui)
            return u, v, sign_up * sign_dn, rui, cui

        def q_one(f, G, U, D, p, q, swap_idx=False, raw=False):
            u, v, sg, _, _ = uv_one(f, U, D, p, q, swap_idx)
            r = lrux.det_lru(G, u, v, False) * sg
            if not raw:                                      # single-site gauge (-1)^(p-q): quantax = site-ordered up-first convention
                r = r * jnp.where(((p - q) & 1) == 1, -1, 1)
            return r

        def Q_ratios(f, G, U, D, z, swap_idx=False, raw=False, K=None):
            p_, q_, valid = bond_geom(z)
            pu = jnp.where(valid, p_, U[0]); qu = jnp.where(valid, q_, D[0])    # safe dummies on aligned bonds (masked below)
            if K is None:
                r = jax.vmap(lambda p, q: q_one(f, G, U, D, p, q, swap_idx, raw))(pu, qu)
                return jnp.where(valid, r.astype(cdt), 0.0)
            idx = jnp.argsort(~valid, stable=True)[:K]                       # compacted antiparallel list (static size K); gather, not scatter (scatter is 25x slower on GPU)
            r = jax.vmap(lambda p, q: q_one(f, G, U, D, p, q, swap_idx, raw))(pu[idx], qu[idx])
            rank = jnp.clip(jnp.cumsum(valid) - 1, 0, K - 1)
            return jnp.where(valid, r.astype(cdt)[rank], 0.0)

        def Q_fresh(f, z, **kw):
            U, D = lists(z)
            F = f[U][:, D]
            sg, ld = jnp.linalg.slogdet(F)                   # quantax init_internal computes both the inverse and the slogdet
            G = jnp.linalg.inv(F)
            lg = ld.astype(cdt) + jnp.log(sg.astype(cdt)) + 1j * PI * inv_count(z)
            return lg, Q_ratios(f, G, U, D, z, **kw)
        self.uv_one, self.q_one, self.Q_ratios, self.Q_fresh = uv_one, q_one, Q_ratios, Q_fresh

        # ---------------------------------------------------------------- persistent state: z, G, U, D (in-place order as quantax)
        def init_state(f, z):
            U, D = lists(z)
            return (z, jnp.linalg.inv(f[U][:, D]), U, D)

        def move(f, st, p, q):
            z, G, U, D = st
            u, v, sg, rui, cui = uv_one(f, U, D, p, q)
            ratio, Gn = lrux.det_lru(G, u, v, True)
            return (swap(z, p, q), Gn, U.at[rui].set(q.reshape(1)), D.at[cui].set(p.reshape(1))), ratio
        self.init_state, self.move = init_state, move

        # ---------------------------------------------------------------- one walker step per arm: (state, ratios checksum, new state)
        def stepR(f, z, key):
            p, q = pick(z, key); z = swap(z, p, q)
            lg, r = R_ratios(f, z)
            return z, jnp.sum(r) + lg

        def stepQ(f, z, key, K=None):
            p, q = pick(z, key); z = swap(z, p, q)
            lg, r = Q_fresh(f, z, K=K)
            return z, jnp.sum(r) + lg

        def stepL(f, st, key, K=None):
            p, q = pick(st[0], key)
            st, _ = move(f, st, p, q)
            z, G, U, D = st
            return st, jnp.sum(Q_ratios(f, G, U, D, z, K=K))

        def stepH(f, st, key):
            p, q = pick(st[0], key)
            st, _ = move(f, st, p, q)
            z, G, U, D = st
            return st, jnp.sum(H_ratios(f, G, U, D, z))
        self.stepR, self.stepQ, self.stepL, self.stepH = stepR, stepQ, stepL, stepH

        # ---------------------------------------------------------------- delayed L (lrux.det_lru_delayed), static delay index t
        def init_carrier(f, z, T):
            U, D = lists(z)
            return (z, lrux.init_det_carrier(f[U][:, D], T, 2), U, D)

        def stepLd(f, st, key, t, T, K=None):
            z, car, U, D = st
            p, q = pick(z, key)
            u, v, sg, rui, cui = uv_one(f, U, D, p, q)
            ratio, car = lrux.det_lru_delayed(car, u, v, True, t)
            z = swap(z, p, q); U = U.at[rui].set(q.reshape(1)); D = D.at[cui].set(p.reshape(1))
            cd = t + 1
            if cd == T:
                car = lrux.merge_det_delays(car); cd = 0
            p_, q_, valid = bond_geom(z)
            pu = jnp.where(valid, p_, U[0]); qu = jnp.where(valid, q_, D[0])

            def one(pp, qq):
                u, v, sg, _, _ = uv_one(f, U, D, pp, qq)
                if cd == 0:
                    r = lrux.det_lru(car.Ainv, u, v, False)
                else:
                    r = lrux.det_lru_delayed(car, u, v, False, cd)
                return r * sg * jnp.where(((pp - qq) & 1) == 1, -1, 1)
            if K is None:
                r = jax.vmap(one)(pu, qu); r = jnp.where(valid, r.astype(cdt), 0.0)
            else:
                idx = jnp.argsort(~valid, stable=True)[:K]
                r = jax.vmap(one)(pu[idx], qu[idx])
                rank = jnp.clip(jnp.cumsum(valid) - 1, 0, K - 1)
                r = jnp.where(valid, r.astype(cdt)[rank], 0.0)
            return (z, car, U, D), r
        self.init_carrier, self.stepLd = init_carrier, stepLd


def load_f(path, dtype):
    d = np.load(path)
    return jnp.asarray(d['f']).astype(dtype)


def relerr(r, r_ref, valid):
    """per configuration: elementwise max |r - r_ref|/|r_ref| over valid bonds, and normwise max|r - r_ref| / max|r_ref|."""
    d = jnp.abs(r - r_ref)
    el = jnp.max(jnp.where(valid, d / jnp.abs(r_ref), 0.0), axis=-1)
    nw = jnp.max(jnp.where(valid, d, 0.0), axis=-1) / jnp.max(jnp.where(valid, jnp.abs(r_ref), 0.0), axis=-1)
    return el, nw


def sign_mismatch(r, r_ref, valid):
    return jnp.sum(valid & (jnp.real(r * jnp.conj(r_ref)) < 0), axis=-1)
