"""Bond-class hop operators V_J1, V_J2 in the exact 6x6 symmetric sector (k=0, A1, spin-flip even).

Basis: normalised orbit states |r> = n_r^{-1/2} sum_{x in orbit r} |x>; a vector is v_r = amp_r sqrt(n_r)
(as sec.vec / sec.v0).  For the hop V_c = (J_c/2) sum_{b in c} (S+_i S-_j + h.c.) (flips an antiparallel bond b):
  <r'|V_c|r> = (n_r n_r')^{-1/2} sum_{x in r} sum_{x' in r'} <x'|V_c|x> = sqrt(n_r/n_r') #(bonds b in c of x_r with
  flip_b(x_r) in orbit r') J_c/2,   a symmetric matrix, so
  (V_c u)_r = sum_{b in c, b antiparallel at x_r} (J_c/2) sqrt(n_r / n_{r'(b)}) u_{r'(b)}.
Since all sector characters are +1 there are no phases; flips that return into orbit r are kept (r' = r), so
D0 + V_J1 + V_J2 = sec.H with D0 the per-rep S^z S^z diagonal.

Neighbour table nb_c[r, k] (int32) = orbit index of flip_{b_k}(x_r) for the k-th bond of class c, -1 if parallel.
NN bonds are the first len(SS.NN) entries of BI/BJ/MASKS, NNN the rest.
"""
import os, time, json, hashlib
from functools import partial
import numpy as np
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import st6_sector as SS

CLASSES = {'J1': (0, len(SS.NN), 1.0), 'J2': (len(SS.NN), len(SS.NN) + len(SS.NNN), SS.J2)}


def canon_rep(T, reps, x):
    """as SS.canon, also returning the canonical representative m (to verify reps[idx] == m)."""
    img = jnp.zeros((x.shape[0], T.shape[0]), jnp.uint64)
    for b in range(5):
        byte = ((x >> jnp.uint64(8 * b)) & jnp.uint64(255)).astype(jnp.int32)
        img = img | T[:, b, :][:, byte].T
    m = jnp.minimum(jnp.min(img, axis=1), jnp.min(img ^ jnp.uint64(SS.FULL), axis=1))
    return jnp.clip(jnp.searchsorted(reps, m), 0, reps.shape[0] - 1), m


@partial(jax.jit, static_argnums=(3, 4))
def _nb_chunk(T, reps, x, b0, b1):
    bi = jnp.asarray(SS.BI[b0:b1]); bj = jnp.asarray(SS.BJ[b0:b1]); mk = jnp.asarray(SS.MASKS[b0:b1])
    anti = (((x[:, None] >> bi[None, :]) ^ (x[:, None] >> bj[None, :])) & jnp.uint64(1)) == 1
    y = (x[:, None] ^ mk[None, :]).reshape(-1)
    idx, m = canon_rep(T, reps, y)
    miss = jnp.sum((reps[idx] != m).reshape(anti.shape) & anti)
    nb = jnp.where(anti, idx.reshape(anti.shape).astype(jnp.int32), -1)
    return nb, miss


def build_table(sec, cls, rows=8192, limit=None, log=print):
    """(D or limit, nbond) int32 table; returns (table, info).  limit: build only the first `limit` reps (timing)."""
    b0, b1, _ = CLASSES[cls]
    D = sec.D if limit is None else min(limit, sec.D)
    out = np.empty((D, b1 - b0), np.int32); miss = 0; t0 = time.time()
    for i in range(0, D, rows):
        x = sec.reps[i:min(D, i + rows)]
        if x.shape[0] < rows: x = jnp.concatenate([x, jnp.repeat(x[-1:], rows - x.shape[0])])
        nb, ms = _nb_chunk(sec.T, sec.reps, x, b0, b1)
        k = min(D, i + rows) - i
        out[i:i + k] = np.asarray(nb[:k]); miss += int(ms)
        if (i // rows) % 200 == 0: log(f'[ops] {cls} table rows {i + k}/{D} {time.time() - t0:.0f}s')
    info = dict(cls=cls, rows=D, nbond=b1 - b0, missing_orbits=miss, sec=time.time() - t0,
                n_anti=int((out >= 0).sum()))
    return out, info


def reps_hash(sec):
    return hashlib.sha256(np.asarray(sec.reps_np[:: 997]).tobytes()).hexdigest()[:16]


def load_or_build(sec, cls, tabdir, log=print):
    """table from tabdir/nb_<cls>.npy if its meta matches this sector, else build and save."""
    os.makedirs(tabdir, exist_ok=True)
    f = os.path.join(tabdir, f'nb_{cls}.npy'); fm = f + '.json'; h = reps_hash(sec)
    if os.path.exists(f) and os.path.exists(fm):
        meta = json.load(open(fm))
        if meta.get('D') == sec.D and meta.get('reps_hash') == h:
            t = np.load(f); log(f'[ops] loaded {f}'); return t, dict(meta, loaded=True)
    t, info = build_table(sec, cls, log=log)
    np.save(f + '.tmp.npy', t); os.replace(f + '.tmp.npy', f)
    info.update(D=sec.D, reps_hash=h, file=f, sha256=hashlib.sha256(open(f, 'rb').read()).hexdigest())
    json.dump(info, open(fm, 'w'), indent=1); log(f'[ops] saved {f} {info}')
    return t, info


class BondOp:
    """x -> V_c x in the normalised orbit basis; scale multiplies J_c (for the deliberate mis-scaling check)."""

    def __init__(self, sec, table, cls, scale=1.0, ch=1 << 20):
        self.cls = cls; self.J = CLASSES[cls][2] * scale; self.D = D = sec.D; self.ch = ch
        nc = -(-D // ch); Dp = nc * ch
        tp = np.full((Dp, table.shape[1]), -1, np.int32); tp[:D] = table
        self.tab = jnp.asarray(tp.reshape(nc, ch, table.shape[1])); del tp
        sq = np.ones(Dp); sq[:D] = np.asarray(sec.sqrt_n)
        self.sq = jnp.asarray(sq.reshape(nc, ch)); self.isq = jnp.asarray(1.0 / np.asarray(sec.sqrt_n))
        self.nc = nc; self.Dp = Dp

    def with_scale(self, s):
        o = object.__new__(BondOp); o.__dict__.update(self.__dict__); o.J = self.J * s; return o

    def __call__(self, u):
        return _bond_mv(self.tab, self.sq, self.isq, u, self.J)[:self.D]


@jax.jit
def _bond_mv(tab, sq, isq, u, J):
    w = u * isq                                         # u_r' / sqrt(n_r')

    def one(args):
        t, s = args
        g = jnp.where(t >= 0, w[jnp.maximum(t, 0)], 0.0)
        return 0.5 * J * s * g.sum(1)
    return jax.lax.map(one, (tab, sq)).reshape(-1)


def diag0(sec, b0=0, b1=None):
    """per-rep S^z S^z diagonal over bonds b0:b1 (default all bonds = D0 of c2_t0_6x6.py)."""
    b1 = len(SS.JB) if b1 is None else b1
    BI = jnp.asarray(SS.BI[b0:b1]); BJ = jnp.asarray(SS.BJ[b0:b1]); JB = jnp.asarray(SS.JB[b0:b1])

    @jax.jit
    def diag_chunk(x):
        valid = (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(jnp.float64)
        return 0.25 * JB.sum() - 0.5 * (valid * JB[None, :]).sum(1)
    return jnp.concatenate([diag_chunk(sec.reps[i:i + (1 << 20)]) for i in range(0, sec.D, 1 << 20)])


def checks(sec, D0, V1, V2, nvec=3, seed=7):
    """(a) D0+V1+V2 == H; (b) same with J2 x 1.01 must fail; (c) symmetry of V1, V2."""
    ks = jax.random.split(jax.random.PRNGKey(seed), nvec + 2)
    a, b = [], []
    V2b = V2.with_scale(1.01)
    for k in ks[:nvec]:
        u = jax.random.normal(k, (sec.D,)); Hu = sec.H(u); n = float(jnp.linalg.norm(Hu))
        r1 = V1(u)
        a.append(float(jnp.linalg.norm(D0 * u + r1 + V2(u) - Hu)) / n)
        b.append(float(jnp.linalg.norm(D0 * u + r1 + V2b(u) - Hu)) / n)
    u = jax.random.normal(ks[nvec], (sec.D,)); w = jax.random.normal(ks[nvec + 1], (sec.D,))
    c = {}
    for V in (V1, V2):
        x = float(u @ V(w)); y = float(w @ V(u)); c[V.cls] = abs(x - y) / abs(x)
    return dict(a_rel=a, b_rel_misscaled=b, c_sym=c,
                a_pass=max(a) < 1e-12, b_fails=min(b) > 1e-12, c_pass=max(c.values()) < 1e-12)
