"""Exact sector helpers for experiments/itfit_6x6 (6x6 J1-J2, J2/J1 = 0.5, sector k=0, A1, flip+; float64).

Conventions of ../stall_6x6/st6_sector.py (imported read-only).  Sector vectors carry sqrt(n_r): an amplitude a (per
configuration) is the vector w_r = a_r sqrt(n_r).  Frozen FN Hamiltonian of a guide (a_g, s):
  F = D - K,  D_r = sum_{r': s_r s_r' = +1} H_rr' w_g,r'/w_g,r   (diagonal incl. the sign-violating "wall"),
              K x = 1/2 [H x - s H (s x)]                         (kept hops, entries >= 0, zero diagonal).
I2 node weight of a log-amplitude error e around a positive reference u (Perron vector of F):
  c_e(r) = 1/2 sum_{r' kept} K_rr' u_r u_r' (e_r - e_r')^2 ,  sum_r c_e(r) = N Q(e)   (formula of writeback_tail).
"""
import json, os, sys, time
from functools import partial
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p_ in (HERE, os.path.join(HERE, '..', 'stall_6x6')):
    if p_ not in sys.path: sys.path.insert(0, p_)
import st6_sector as SS
import jax
import jax.numpy as jnp
import scipy.linalg

ROOT = '/project/theorie/a/A.Otaifi/chatty_stall6'
CSR = os.environ.get('ST6_CSR', ROOT + '/csr')
TABLE = os.environ.get('ST6_TABLE', ROOT + '/data/psi0_6x6_table.npz')
SYM = os.environ.get('ST6_SYM', ROOT + '/data/sym_tables.npz')
N = 36
E0_SITE = -0.50380965389088
log = SS.log
f64 = jnp.float64


def dump(path, obj):
    tmp = path + '.tmp'
    json.dump(obj, open(tmp, 'w'), indent=1, default=float)
    os.replace(tmp, path)


@partial(jax.jit, donate_argnums=(0,))
def _diag_chunk(dg, rbase, inc, idx, cod):
    row = rbase + jnp.cumsum(inc.astype(jnp.int32))
    return dg.at[row].add(jnp.where(idx == row, cod.astype(f64) * 0.125, 0.0))


class Exact:
    def __init__(self):
        self.sec = sec = SS.Sector(CSR, TABLE)
        sec.v0 = sec.la0 = sec.s0 = sec.p0 = None
        z = np.load(SYM)
        self.lP = jnp.asarray(z['lP']); self.sP = jnp.asarray(z['sP'].astype(np.float32)); del z
        self.sqn = sec.sqrt_n
        dg = jnp.zeros(sec.D, f64)
        for (rb, inc, idx, cod) in sec.chunks:
            dg = _diag_chunk(dg, rb, inc, idx, cod)
        self.Hdiag = dg
        self.nK = 0

    # ------------------------------------------------------------------ operators
    def Kop(self, X, s):
        """kept-hop part on sector vectors, X (D,) or (D, k); s = sign (D,)."""
        one = X.ndim == 1
        if one: X = X[:, None]
        k = X.shape[1]
        sc = s[:, None].astype(f64)
        Y = self.sec.Hm(jnp.concatenate([X, sc * X], 1))
        self.nK += k
        out = 0.5 * (Y[:, :k] - sc * Y[:, k:])
        return out[:, 0] if one else out

    def guide_vec(self, la):
        """normalised positive sector vector of a per-config log amplitude, with the fn_solve clamp."""
        la = la - jnp.max(la)
        a = jnp.exp(la); a = a / jnp.sqrt(jnp.sum(self.sec.n * a * a)); a = jnp.maximum(a, 1e-15)
        return a * self.sqn

    def pos_vec(self, la):
        la = la - jnp.max(la)
        v = jnp.exp(la) * self.sqn
        return v / jnp.linalg.norm(v)

    def fn_diag(self, la_g, s):
        """FN diagonal D of the guide (la_g, s) and the guide's kept-hop vector K w (both (D,))."""
        w = self.guide_vec(la_g)
        Y = self.sec.Hm(jnp.stack([w, s * w], 1)); self.nK += 1
        D = 0.5 * (Y[:, 0] + s * Y[:, 1]) / w
        Kw = 0.5 * (Y[:, 0] - s * Y[:, 1])
        return D, w, Kw

    def Ef(self, b, D, s, Kb=None):
        """frozen energy (total, not per site) of a sector vector b on F = D - K[s]; returns (E, K b)."""
        Kb = self.Kop(b, s) if Kb is None else Kb
        return float((b @ (D * b) - b @ Kb) / (b @ b)), Kb

    def energy_H(self, la, s):
        v = self.sec.vec(la, s)
        return float(v @ self.sec.H(v))

    def node_c(self, e, u, Ku, s):
        c = e * e * u * Ku
        c = c - 2 * e * u * self.Kop(u * e, s)
        return 0.5 * (c + u * self.Kop(u * e * e, s))

    # ------------------------------------------------------------------ spectra / solvers
    def lanczos_ext(self, D, s, m=100, seed=0):
        """extreme eigenvalues of F = D - K[s] by plain Lanczos (no reorthogonalisation; extremes converge)."""
        v = jax.random.normal(jax.random.PRNGKey(seed), (self.sec.D,), f64)
        v = v / jnp.linalg.norm(v); vp = jnp.zeros_like(v); bet = 0.0
        al, be, hist = [], [], []
        for j in range(m):
            w = D * v - self.Kop(v, s)
            a = float(v @ w); al.append(a)
            w = w - a * v - bet * vp
            bet = float(jnp.linalg.norm(w)); be.append(bet)
            vp, v = v, w / bet
            if (j + 1) % 10 == 0 or j == m - 1:
                ev = scipy.linalg.eigvalsh_tridiagonal(np.array(al), np.array(be[:-1]))
                hist.append(dict(m=j + 1, lmax=float(ev[-1]), lmin=float(ev[0])))
        return hist

    def pcg(self, diagA, tau, s, rhs, tol=1e-8, maxit=200):
        """solve [diagA - tau K] x = rhs, Jacobi preconditioner diagA (SPD case)."""
        x = rhs / diagA; r = rhs - (diagA * x - tau * self.Kop(x, s))
        z = r / diagA; p = z; rz = float(r @ z); nb = float(jnp.linalg.norm(rhs))
        it = 0
        for it in range(1, maxit + 1):
            Ap = diagA * p - tau * self.Kop(p, s)
            al = rz / float(p @ Ap)
            x = x + al * p; r = r - al * Ap
            if float(jnp.linalg.norm(r)) <= tol * nb: break
            z = r / diagA; rzn = float(r @ z)
            p = z + (rzn / rz) * p; rz = rzn
        return x, it, float(jnp.linalg.norm(r)) / nb


def decade_table(key_vals, weights, lo=-4, hi=7):
    """share of each weight vector (dict name -> (D,) >= 0 or signed) per decade of key_vals (> 0)."""
    lk = jnp.floor(jnp.log10(jnp.maximum(key_vals, 1e-300))).astype(jnp.int32)
    tots = {k: float(jnp.sum(v)) for k, v in weights.items()}
    out = []
    for d in range(lo, hi + 1):
        m = (lk <= d) if d == lo else ((lk >= d) if d == hi else (lk == d))
        row = dict(decade=d, edge=('<=' if d == lo else ('>=' if d == hi else '')))
        for k, v in weights.items():
            row[k] = float(jnp.sum(jnp.where(m, v, 0.0))) / tots[k] if tots[k] != 0 else float('nan')
        out.append(row)
    return out


def base_feature_table(ex, la, s, p, E, taus=(0.3, 1.0, 3.0), zclip=6.0):
    """(D, 10 + len(taus)) float32 table of standardised (weights p), clipped one-hop features of the guide (la, s):
    log a, H_xx, log W, log V, nK, nV, log W2, log V2, LK, LV, and T_tau = log(1 + tau W) - log(1 + tau (H_xx + V - E)).
    (W, V: kept / sign-violating sum_y |H_xy| a(y)/a(x); W2, V2 with squared ratios; nK, nV bond weights; LK, LV sums
    of |H_xy| log ratios.)  Exact symmetric functions of the orbit, so a lookup by representative is exact."""
    sec = ex.sec; sqn = ex.sqn; Hd = ex.Hdiag
    la = la - jnp.max(la)
    a = jnp.maximum(jnp.exp(la), 1e-300)
    w = a * sqn
    X = jnp.stack([w, a * a * sqn, la * sqn, sqn], 1)
    sm, kp = [], []
    for j in range(4):
        Y = sec.Hm(jnp.stack([X[:, j], s * X[:, j]], 1))
        sm.append(0.5 * (Y[:, 0] + s * Y[:, 1])); kp.append(0.5 * (Y[:, 0] - s * Y[:, 1])); del Y
    same = jnp.stack(sm, 1); kept = jnp.stack(kp, 1); del X, sm, kp
    W = kept[:, 0] / w; V = same[:, 0] / w - Hd
    W2 = kept[:, 1] / (a * a * sqn); V2 = same[:, 1] / (a * a * sqn) - Hd
    nK = kept[:, 3] / sqn; nV = same[:, 3] / sqn - Hd
    LK = kept[:, 2] / sqn - la * nK; LV = (same[:, 2] / sqn - Hd * la) - la * nV
    lg = lambda x: jnp.log(jnp.maximum(x, 0.0) + 1e-6)
    cols = [la, Hd, lg(W), lg(V), nK, nV, lg(W2), lg(V2), LK, LV]
    cols += [jnp.log1p(t * jnp.maximum(W, 0)) - jnp.log1p(t * jnp.maximum(Hd + V - E, 1e-12)) for t in taus]
    out = []
    for c in cols:
        m = float(jnp.sum(p * c)); sd = float(jnp.sqrt(jnp.sum(p * (c - m) ** 2))) + 1e-30
        out.append(jnp.clip((c - m) / sd, -zclip, zclip).astype(jnp.float32))
    return jnp.stack(out, 1)
