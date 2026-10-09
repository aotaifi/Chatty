"""Core of the 8x8 route-B test (experiments/loop_8x8/README.md): periodic 8x8 J1-J2, J2/J1 = 0.5, everything sampled.

Conventions (as ll6_core / lanczos1_8x8 / fn_guides_compare): state = uint64, bit i = spin up on site i = x + 8 y,
network input 2*bit-1 in site order; H = sum_bonds J S_i.S_j, off-diagonal element between x and y = x with (i,j)
exchanged is +J/2.  Guide (a, s): a = |psi| of the 8x8 ViT (translation-invariant; optionally projected onto D4 x spin
flip: psi_P(x) = sum_{g in D4, flip} psi(g x), as stall_6x6 st6_run.py), s = binarised own sign
s(x) = sign cos(arg psi(x) - phi) with one global phase phi (lanczos1_8x8.binary_phi).
Kept edge: s(x) s(y) = -1; violating: s(x) s(y) = +1.  One-hop guide features
  V(x) = sum_viol J/2 a(y)/a(x),  W(x) = sum_kept J/2 a(y)/a(x),  D(x) = H_xx,  E_L(x) = D + V - W.
The ViT is the dtype-configurable copy experiments/learned_loop_6x6/vit_dt.py (same parameter tree as nqsmagic), fp32
with full-precision matmuls.
"""
import math, os, sys, time, json
from functools import partial
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p_ in (HERE, os.path.join(HERE, '..', 'learned_loop_6x6')):
    if p_ not in sys.path: sys.path.append(p_)
import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')      # full fp32 (TF32 off)
import jax.numpy as jnp
import flax
import vit_dt

L = 8; N = 64; J2 = 0.5
ONE = np.uint64(1)


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


def dump(path, obj):
    json.dump(obj, open(path, 'w'), indent=1, default=float)


def _bonds():
    nn_, nnn = [], []
    q = lambda x, y: (x % L) + L * (y % L)
    for y in range(L):
        for x in range(L):
            i = q(x, y); nn_ += [(i, q(x + 1, y)), (i, q(x, y + 1))]
            nnn += [(i, q(x + 1, y + 1)), (i, q(x + 1, y - 1))]
    return nn_, nnn


NN, NNN = _bonds()
ALL = [(i, j, 1.) for i, j in NN] + [(i, j, J2) for i, j in NNN]
NB_ = len(ALL)                                                    # 256 bonds
BI = np.array([b[0] for b in ALL], np.uint64); BJ = np.array([b[1] for b in ALL], np.uint64)
JB = np.array([b[2] for b in ALL], float)
MASKS = (ONE << BI) | (ONE << BJ)
A_MASK = np.uint64(sum(1 << (x + L * y) for y in range(L) for x in range(L) if (x + y) % 2 == 0))


def d4_perms():
    """8 site permutations of the point group around the origin: X[:, P[g]] is the image of configuration X."""
    ops = [lambda x, y: (x, y), lambda x, y: (-y, x), lambda x, y: (-x, -y), lambda x, y: (y, -x),
           lambda x, y: (-x, y), lambda x, y: (x, -y), lambda x, y: (y, x), lambda x, y: (-y, -x)]
    P = np.zeros((8, N), np.int32)
    for g, op in enumerate(ops):
        for y in range(L):
            for x in range(L):
                u, v = op(x, y); P[g, (x % L) + L * (y % L)] = (u % L) + L * (v % L)
    return P


PERM = d4_perms()


def neighbors(S):
    S = np.asarray(S, np.uint64)[:, None]
    valid = (((S >> BI) ^ (S >> BJ)) & ONE).astype(bool)
    return S ^ MASKS, valid


def diag_vec(valid):
    return 0.25 * JB.sum() - 0.5 * (valid.astype(float) @ JB)


def bits_to_spins_np(S):
    a = np.asarray(S, np.uint64).reshape(-1, 1)
    return 2.0 * ((a >> np.arange(N, dtype=np.uint64)) & ONE).astype(np.float32) - 1.0


def spins_to_bits_np(X):
    X = np.asarray(X).reshape(-1, N)
    return np.sum((X > 0).astype(np.uint64) << np.arange(N, dtype=np.uint64)[None, :], axis=1, dtype=np.uint64)


def random_sz0(rg, n):
    X = -np.ones((n, N), np.float32)
    for k in range(n):
        X[k, rg.choice(N, N // 2, replace=False)] = 1
    return X


AR = jnp.arange(N, dtype=jnp.uint64)


def bits_j(S):
    return (((S[:, None] >> AR[None, :]) & jnp.uint64(1)).astype(jnp.float32) * 2 - 1)


# ============================================================================================ guide (ViT)
def make_model_8x8():
    return vit_dt.ViT(num_layers=8, d_model=60, heads=10, L_eff=16, b=2, transl_invariant=True, two_dimensional=True)


class Guide:
    """log|psi| and arg psi of the 8x8 ViT (sym=False: translation-invariant network as trained; sym=True: projected
    onto D4 x spin flip, 16x the cost).  eval(S uint64) -> (log a, phase) float64 numpy, fixed padded batches."""

    def __init__(self, ckpt, sym=False, batch=8192, phi=0.0):
        vit_dt.set_dtype('float32')
        model = make_model_8x8()
        tmpl = model.init(jax.random.PRNGKey(1234), jnp.zeros((1, N), jnp.float32))
        obj = flax.serialization.msgpack_restore(open(ckpt, 'rb').read())
        if 'variables' in obj: obj = obj['variables']
        var = flax.serialization.from_state_dict(tmpl, obj)
        self.params = jax.tree_util.tree_map(lambda p: jnp.asarray(p, jnp.float32), var['params'])
        self.npar = int(sum(np.size(p) for p in jax.tree_util.tree_leaves(self.params)))
        self.sym = sym; self.batch = batch; self.phi = float(phi); self.neval = 0
        perm = jnp.asarray(PERM); sg = jnp.asarray([1.0, -1.0], jnp.float32)

        def z1(params, X):
            return vit_dt.logpsi_transl_2d(model.apply, 2, {'params': params}, X)

        if sym:
            def lz(params, X):
                zs = jax.lax.map(lambda k: z1(params, X[:, perm[k // 2]] * sg[k % 2]), jnp.arange(16))
                m = jnp.max(jnp.real(zs), 0)
                Ssum = jnp.sum(jnp.exp(zs - m[None, :]), 0)
                return m + jnp.log(jnp.abs(Ssum)), jnp.angle(Ssum)
        else:
            def lz(params, X):
                z = z1(params, X)
                return jnp.real(z), jnp.imag(z)
        self.lz_spins = lz                                         # (params, X float32 spins) -> (log a, phase)
        self._bits = jax.jit(lambda params, S: lz(params, bits_j(S)))

    def eval(self, S):
        S = np.asarray(S, np.uint64); n = len(S); B = self.batch
        la = np.empty(n); ph = np.empty(n)
        if n == 0: return la, ph
        res = []
        for i in range(0, n, B):
            s = S[i:i + B]; m = len(s)
            if m < B: s = np.concatenate([s, np.repeat(s[:1], B - m)])
            res.append((i, m, self._bits(self.params, jnp.asarray(s))))
        for i, m, (a, p) in res:
            la[i:i + m] = np.asarray(a)[:m]; ph[i:i + m] = np.asarray(p)[:m]
        self.neval += n
        return la, ph

    def sign(self, ph):
        return np.where(np.cos(ph - self.phi) >= 0, 1.0, -1.0)


def onehop(guide, S, Ea=None, keep_edges=False):
    """Guide one-hop data at configurations S (any order).  Returns per-x arrays; with keep_edges also the per-bond
    arrays (n, 256): valid mask, neighbour log a, neighbour sign, neighbour states."""
    S = np.asarray(S, np.uint64); n = len(S)
    NBs, V = neighbors(S)
    st = np.concatenate([S, NBs[V]])
    u, inv = np.unique(st, return_inverse=True)
    lu, pu = guide.eval(u)
    la_all = lu[inv]; ph_all = pu[inv]
    lax_ = la_all[:n]; phx = ph_all[:n]
    lan = np.zeros((n, NB_)); phn = np.zeros((n, NB_))
    lan[V] = la_all[n:]; phn[V] = ph_all[n:]
    sx = guide.sign(phx); sn = np.where(V, guide.sign(phn), 0.0)
    r = np.where(V, 0.5 * JB[None, :] * np.exp(np.clip(lan - lax_[:, None], -60, 60)), 0.0)
    kept = V & (sx[:, None] * sn < 0); viol = V & ~kept
    Vv = (r * viol).sum(1); W = (r * kept).sum(1); D = diag_vec(V)
    # complex local energy of the network itself (own phase), for reference energies
    elc = D + np.sum(np.where(V, 0.5 * JB[None, :] * np.exp(np.clip(lan - lax_[:, None], -60, 60)
                                                               + 1j * (phn - phx[:, None])), 0.0), 1)
    out = dict(la=lax_, ph=phx, s=sx, V=Vv, W=W, D=D, ELa=D + Vv - W, ELc=elc, nval=V.sum(1), nunique=len(u))
    if Ea is not None:
        out['T'] = np.log1p(np.maximum(W, 0.0)) - np.log1p(np.maximum(D + Vv - Ea, 1e-12))
    if keep_edges:
        out.update(valid=V, lan=lan, sn=sn, nb=NBs, r=r)
    return out


# ============================================================================================ sampler
BI_J = jnp.asarray(BI.astype(np.int32)); BJ_J = jnp.asarray(BJ.astype(np.int32))


def make_sampler(guide, beta):
    """Bond-exchange Metropolis for a^(2 beta) (same move set as ll6_core / fn_guides_compare), nsteps per call."""
    lz = guide.lz_spins

    @partial(jax.jit, static_argnums=(4,))
    def run(params, X, la, key, nsteps):
        nc = X.shape[0]; ar = jnp.arange(nc)

        def body(carry, _):
            X, la, key, acc = carry
            key, k1, k2 = jax.random.split(key, 3)
            b = jax.random.randint(k1, (nc,), 0, NB_)
            i = BI_J[b]; j = BJ_J[b]
            xi = X[ar, i]; xj = X[ar, j]
            ok = xi != xj
            cand = X.at[ar, i].set(xj).at[ar, j].set(xi)
            lb, _ = lz(params, cand)
            u = jax.random.uniform(k2, (nc,), jnp.float32)
            accept = ok & (jnp.log(u) < 2.0 * beta * (lb - la))
            X = jnp.where(accept[:, None], cand, X); la = jnp.where(accept, lb, la)
            return (X, la, key, acc + accept.sum()), None

        (X, la, key, acc), _ = jax.lax.scan(body, (X, la, key, jnp.asarray(0)), None, length=nsteps)
        return X, la, key, acc
    return run


def sample_chains(guide, beta, nchains, burn_sweeps, nsamp, thin_sweeps, seed):
    """returns states (nsamp, nchains) uint64 (row = time), acceptance."""
    rg = np.random.default_rng(seed)
    run = make_sampler(guide, beta)
    X = jnp.asarray(random_sz0(rg, nchains))
    la, _ = jax.jit(guide.lz_spins)(guide.params, X)
    key = jax.random.PRNGKey(seed)
    X, la, key, acc = run(guide.params, X, la, key, int(burn_sweeps * N))
    out = []; acc_tot = 0
    for t in range(nsamp):
        X, la, key, acc = run(guide.params, X, la, key, int(thin_sweeps * N))
        acc_tot += int(acc)
        out.append(spins_to_bits_np(np.asarray(X)))
    guide.neval += nchains * N * (burn_sweeps + nsamp * thin_sweeps)
    return np.stack(out), acc_tot / (nchains * N * thin_sweeps * nsamp)


def chain_se(vals, chains):
    """SE of the mean from per-chain means (vals, chains = chain id per sample)."""
    ids = np.unique(chains)
    m = np.array([vals[chains == c].mean() for c in ids])
    return float(m.std(ddof=1) / np.sqrt(len(ids)))
