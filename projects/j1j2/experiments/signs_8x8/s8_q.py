"""Numpy-only helpers (no jax): lattice, Marshall sign, one-hop Krylov quantities from two-hop chunks, energies.
Conventions as experiments/loop_8x8/l8_core.py: state uint64, bit i = up spin on site i = x + 8 y; bonds NN (J=1) then
NNN (J2 = 0.5); off-diagonal element of H between x and its exchange neighbour = +J/2.
"""
import json
import numpy as np

L = 8; N = 64; ONE = np.uint64(1)


def _bonds():
    nn_, nnn = [], []
    q = lambda x, y: (x % L) + L * (y % L)
    for y in range(L):
        for x in range(L):
            i = q(x, y); nn_ += [(i, q(x + 1, y)), (i, q(x, y + 1))]
            nnn += [(i, q(x + 1, y + 1)), (i, q(x + 1, y - 1))]
    return [(i, j, 1.) for i, j in nn_] + [(i, j, 0.5) for i, j in nnn]


ALL = _bonds()
BI = np.array([b[0] for b in ALL], np.uint64); BJ = np.array([b[1] for b in ALL], np.uint64)
JB = np.array([b[2] for b in ALL], float)
MASKS = (ONE << BI) | (ONE << BJ)
A_MASK = np.uint64(sum(1 << (x + L * y) for y in range(L) for x in range(L) if (x + y) % 2 == 0))


def popcount(S):
    S = np.asarray(S, np.uint64)
    if hasattr(np, 'bitwise_count'): return np.bitwise_count(S).astype(np.int64)
    c = np.zeros(S.shape, np.int64); s = S.copy()
    for _ in range(64):
        c += (s & ONE).astype(np.int64); s >>= ONE
    return c


def marshall(S):
    return np.where(popcount(np.asarray(S, np.uint64) & A_MASK) % 2 == 0, 1, -1).astype(np.int8)


def neighbors(S):
    S = np.asarray(S, np.uint64)[:, None]
    valid = (((S >> BI) ^ (S >> BJ)) & ONE).astype(bool)
    return S ^ MASKS, valid


def diag_vec(valid):
    return 0.25 * JB.sum() - 0.5 * (valid.astype(float) @ JB)


def onehop_from_chunk(S, U2, la, signs):
    """S: samples x of the chunk; U2 sorted two-hop set with log a; signs: dict name -> int8 array on U2.
    Returns L1 (sorted {x} u N1(x)), per-L1 r_b and la, and for the samples: per-x D, la, own-sign products and per-edge
    arrays (own, Jhalf, dla, b_x b_y, r_b(y))."""
    S = np.asarray(S, np.uint64)
    NBs, V = neighbors(S)
    L1 = np.unique(np.concatenate([S, NBs[V]]))
    NB1, V1 = neighbors(L1)
    idx1 = np.searchsorted(U2, NB1); idx1[~V1] = 0
    self1 = np.searchsorted(U2, L1)
    assert np.all(U2[self1] == L1) and np.all(U2[idx1[V1]] == NB1[V1])
    la1 = la[self1]
    w = np.where(V1, 0.5 * JB[None, :] * np.exp(np.clip(la[idx1] - la1[:, None], -80, 80)), 0.0)
    D1 = diag_vec(V1)
    r1 = {b: D1 + np.sum(w * (sg[idx1] * sg[self1][:, None]).astype(float), axis=1) for b, sg in signs.items()}
    s1 = {b: sg[self1] for b, sg in signs.items()}
    px = np.searchsorted(L1, S)
    own, col = np.nonzero(V); py = np.searchsorted(L1, NBs[own, col])
    x = dict(D=D1[px], la=la1[px], **{f'r_{b}': r1[b][px] for b in signs}, **{f's_{b}': s1[b][px] for b in signs})
    e = dict(own=own, Jh=0.5 * JB[col], dla=la1[py] - la1[px][own],
             **{f'bb_{b}': (s1[b][py] * s1[b][px][own]).astype(np.int8) for b in signs},
             **{f'ry_{b}': r1[b][py] for b in signs})
    return L1, r1, la1, s1, x, e


def eloc(q, b, T=None):
    """Local energy at the samples of (a, b) [T None] or of (a, b sgn(T - r_b)) [one exact Krylov hop]."""
    c = q['Jh'] * q[f'bb_{b}'] * np.exp(q['dla'])
    if T is not None:
        qx = np.where(q[f'r_{b}'] <= T, 1., -1.); qy = np.where(q[f'ry_{b}'] <= T, 1., -1.)
        c = c * qx[q['own']] * qy
    e = q['D'].copy(); np.add.at(e, q['own'], c)
    return e


def curve(q, b, mask=None):
    """E(T) per site of (a, b sgn(T - r_b)) on the samples in mask; returns base (T=+inf), candidates, curve."""
    n_all = len(q['D']); m = np.ones(n_all, bool) if mask is None else mask
    ke = m[q['own']]; n = m.sum()
    c = (q['Jh'] * q[f'bb_{b}'] * np.exp(q['dla']))[ke] / n
    rx = q[f'r_{b}'][q['own']][ke]; ry = q[f'ry_{b}'][ke]
    base = (q['D'][m].sum() / n) + c.sum()
    rv = np.unique(np.concatenate([rx, ry]))
    cand = np.r_[rv[0] - 1.0, 0.5 * (rv[:-1] + rv[1:]), rv[-1] + 1.0]
    lo = np.minimum(rx, ry); hi = np.maximum(rx, ry)
    a = np.searchsorted(cand, lo, 'left'); bb = np.searchsorted(cand, hi, 'left')
    diff = np.zeros(len(cand) + 1)
    np.add.at(diff, a, -2 * c); np.add.at(diff, bb, +2 * c)
    cu = base + np.cumsum(diff)[:len(cand)]
    return base / N, cand, cu / N


def opt_T(q, b, mask=None):
    base, cand, cu = curve(q, b, mask); j = int(np.argmin(cu))
    return float(cand[j]), float(cu[j] - base)


def chain_mean_se(v, ch):
    u, inv = np.unique(ch, return_inverse=True)
    m = np.bincount(inv, weights=v) / np.bincount(inv)
    return float(v.mean()), float(m.std(ddof=1) / np.sqrt(len(u)))


def concat_q(qs):
    out = {}; off = 0
    for q in qs:
        for k, v in q.items():
            out.setdefault(k, []).append(v + off if k == 'own' else v)
        off += len(q['D'])
    return {k: np.concatenate(v) for k, v in out.items()}


def dump(path, obj):
    json.dump(obj, open(path, 'w'), indent=1, default=float)
