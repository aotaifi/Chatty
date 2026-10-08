"""Shared exact setup for experiments/writeback_tail (6x6, J2/J1 = 0.5, sector k=0, A1, flip+).

Guide (a, s) = (|psi_P|, s_P) (symmetrised ViT); phi = Perron vector of H_FN[a, s]; delta = log phi - log a.
FN Laplacian (identity I2, amp_design/DESIGN_MEMO.md): with K = allowed-edge (s_x s_y = -1) part of |H_off| on sector
vectors and u = phi (sector vector, sum u^2 = 1),
  L e = u e (K u) - u K(u e),   Q(e) = e.L e / N = 1/2 sum_{xy allowed} |H_xy| phi_x phi_y (e_x - e_y)^2 / N,
  node weight c_e(x) = 1/2 sum_{y allowed} |H_xy| u_x u_y (e_x - e_y)^2 >= 0,  sum_x c_e(x) = N Q(e).
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p_ in (HERE, os.path.join(HERE, '..', 'stall_6x6')):
    if p_ not in sys.path: sys.path.insert(0, p_)
import st6_sector as SS
import jax
import jax.numpy as jnp

ROOT = '/project/theorie/a/A.Otaifi/chatty_stall6'
CSR = os.environ.get('ST6_CSR', ROOT + '/csr')
TABLE = os.environ.get('ST6_TABLE', ROOT + '/data/psi0_6x6_table.npz')
SYM = os.environ.get('ST6_SYM', ROOT + '/data/sym_tables.npz')
N = 36
log = SS.log
f64 = jnp.float64
BULK_CUT = 1e-10                                    # per-configuration phi^2: bulk >= cut, tail < cut


def dump(path, obj):
    json.dump(obj, open(path, 'w'), indent=1, default=float)


class Setup:
    def __init__(self, n_iter=1):
        """n_iter = 1: target phi = FN Perron vector of the guide (|psi_P|, s_P) (one FN iteration).
        n_iter > 1: target = phi_n of the ideal exact loop from psi_P (FN solve -> Krylov sign step -> FN solve ...);
        the frozen FN Hamiltonian is then H_FN[a_{n-1}, s_{n-1}] (its Perron vector is phi_n), sT = Krylov sign of
        phi_n (the sign the loop would use next).  The trial state is always b = |psi_P| exp(f)."""
        t0 = time.time()
        self.sec = sec = SS.Sector(CSR, TABLE)
        sec.v0 = sec.la0 = sec.s0 = sec.p0 = None
        z = np.load(SYM)
        self.lP = jnp.asarray(z['lP']); self.sP = jnp.asarray(z['sP'].astype(np.float32)); del z
        self.n_iter = n_iter; self.loop = []
        lg, sg = self.lP, self.sP
        for k in range(n_iter):
            Efn, u, info = sec.fn_solve(lg, sg)
            self.loop.append(dict(it=k + 1, dE_FN_site=info['dE_FN_site']))
            log(f'[setup] loop it {k + 1}: E_FN {info["dE_FN_site"]:.6e}')
            if k < n_iter - 1:
                sn, _, _ = sec.krylov(u, sg)
                lg = jnp.log(jnp.maximum(u / sec.sqrt_n, 1e-300)); sg = sn; del u
        self.lg, self.sg = lg, sg                                    # guide of the frozen FN Hamiltonian
        self.Efn, self.u, self.fn_info = Efn, u, info
        self.lphi = jnp.log(jnp.maximum(u / sec.sqrt_n, 1e-300))
        if n_iter > 1:
            self.sT, _, _ = sec.krylov(u, sg)
            vT = sec.vec(self.lphi, self.sT)
            self.loop.append(dict(target_H_site=(sec.energy(vT) - sec.E0) / N)); del vT
            log('[setup] target', self.loop[-1])
        else:
            self.sT = self.sP
        self.p = u * u                                               # rep probability under phi^2
        self.lpc = jnp.log(jnp.maximum(self.p, 1e-300)) - jnp.log(sec.n)   # per-configuration log phi^2
        self.E_f0 = sec.fn_rayleigh(self.lg, self.sg, self.lP)
        self.G0 = (self.E_f0 - self.Efn) / N
        self.delta = self.lphi - self.lP
        self.mu = float(jnp.sum(self.p * self.delta))
        self.Ku = self.Kop(u)
        self.c_delta = self.node_c(self.delta)
        self.Qdelta = float(jnp.sum(self.c_delta)) / N
        self.dec = jnp.floor(jnp.log10(jnp.maximum(jnp.exp(self.lpc), 1e-300))).astype(jnp.int32)
        self.bulk = self.lpc >= np.log(BULK_CUT)
        log(f'[setup] G0 {self.G0:.6e}  Q(delta) {self.Qdelta:.6e}  E_FN {(self.Efn - sec.E0) / N:.6e}  '
            f'{time.time() - t0:.0f}s')

    # ---------------------------------------------------------------- FN Laplacian
    def Kop(self, X):
        """allowed-edge |H_off| part on sector vectors, X (D,) or (D, k)."""
        one = X.ndim == 1
        if one: X = X[:, None]
        k = X.shape[1]
        s = self.sg[:, None].astype(f64)
        Y = self.sec.Hm(jnp.concatenate([X, s * X], 1))
        out = 0.5 * (Y[:, :k] - s * Y[:, k:])
        return out[:, 0] if one else out

    def Lop(self, e):
        return self.u * e * self.Ku - self.u * self.Kop(self.u * e)

    def Q(self, e):
        return float(e @ self.Lop(e)) / N

    def node_c(self, e):
        u = self.u
        K1 = self.Kop(u * e)                                   # two single-column calls: lower peak memory
        out = e * e * u * self.Ku - 2 * e * u * K1; del K1
        return 0.5 * (out + u * self.Kop(u * e * e))

    def decades(self, vals, tot=None):
        tot = float(jnp.sum(vals)) if tot is None else tot
        out = []
        for k in range(-20, -2):
            m = (self.dec == k) if k > -20 else (self.dec <= k)
            out.append(dict(decade=k, mass=float(jnp.sum(jnp.where(m, self.p, 0))),
                            share=float(jnp.sum(jnp.where(m, vals, 0))) / tot))
        return out

    def bulk_tail(self, vals, tot=None):
        tot = float(jnp.sum(vals)) if tot is None else tot
        b = float(jnp.sum(jnp.where(self.bulk, vals, 0))); t = float(jnp.sum(jnp.where(self.bulk, 0, vals)))
        return dict(bulk=b / tot, tail=t / tot)

    # ---------------------------------------------------------------- guide-local features
    def guide_features(self):
        """V(x) = sum_{y violating} |H_xy| a_y/a_x (sign-violating weight of the guide = FN diagonal shift),
        W(x) = sum_{y allowed} |H_xy| a_y/a_x (kept-edge weight), per configuration (one hop of the guide)."""
        sec = self.sec
        la = self.lP - jnp.max(self.lP)
        w = jnp.maximum(jnp.exp(la), 1e-300) * sec.sqrt_n
        h1, h2 = sec.H2(w, self.sP * w)
        same = 0.5 * (h1 + self.sP * h2) / w
        W = 0.5 * (h1 - self.sP * h2) / w
        del h1, h2, w
        d0 = self.diag()
        return same - d0, W

    def diag(self, batch=1 << 20):
        BI = jnp.asarray(SS.BI); BJ = jnp.asarray(SS.BJ); JB = jnp.asarray(SS.JB)

        @jax.jit
        def f(x):
            valid = (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(f64)
            return 0.25 * JB.sum() - 0.5 * (valid * JB[None, :]).sum(1)
        out = []
        for i in range(0, self.sec.D, batch):
            out.append(f(self.sec.reps[i:i + batch]))
        return jnp.concatenate(out)
