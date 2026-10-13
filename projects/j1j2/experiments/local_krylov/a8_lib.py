"""Check (a8) helpers: bond energies, window sums, connected components of marked windows, per-sample energy changes of
one sign-only step (global G or local C_w) on the signs_8x8 two-hop referee chunks.  Numpy + numba, CPU only.
Conventions as experiments/signs_8x8/s8_q.py (state uint64, bit i = up spin on site i = x + 8 y; bonds NN then NNN).
"""
import os, sys
import numpy as np
import numba as nb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'signs_8x8'))
import s8_q as Q  # noqa: E402

L = 8; N = 64; NB = len(Q.ALL)
BI = np.array([b[0] for b in Q.ALL]); BJ = np.array([b[1] for b in Q.ALL]); JB = Q.JB
IS_NN = JB == 1.0


def window_bonds(w):
    """(64 windows x 256 bonds) 0/1: bond b in window W(x0, y0) iff both ends in the w x w block at (x0, y0)."""
    M = np.zeros((N, NB))
    for y0 in range(L):
        for x0 in range(L):
            sites = {((x0 + i) % L) + L * ((y0 + j) % L) for i in range(w) for j in range(w)}
            for b in range(NB):
                if BI[b] in sites and BJ[b] in sites: M[x0 + L * y0, b] = 1.0
    return M


def window_adjacency(WB):
    """bitmask per window of the windows that share at least one bond with it."""
    share = (WB @ WB.T) > 0
    adj = np.zeros(N, np.uint64)
    for i in range(N):
        for j in range(N):
            if i != j and share[i, j]: adj[i] |= np.uint64(1) << np.uint64(j)
    return adj


WB = {w: window_bonds(w) for w in (2, 3, 4)}
ADJ = {w: window_adjacency(WB[w]) for w in WB}
NBOND_W = {w: (int(WB[w][0, IS_NN].sum()), int(WB[w][0, ~IS_NN].sum())) for w in WB}


@nb.njit(cache=True)
def _find(par, i):
    while par[i] != i:
        par[i] = par[par[i]]; i = par[i]
    return i


@nb.njit(cache=True)
def comp_curves(Sw, adj):
    """For each state: windows sorted by decreasing window sum, and the number of connected components after adding the
    first k windows (k = 0..64).  Returns (vals_desc (n, 64), ncomp (n, 65))."""
    n = Sw.shape[0]; vals = np.empty((n, 64)); nc = np.zeros((n, 65), np.int16)
    par = np.empty(64, np.int64); added = np.zeros(64, np.bool_)
    for s in range(n):
        order = np.argsort(-Sw[s])
        for i in range(64):
            par[i] = i; added[i] = False
        c = 0
        for k in range(64):
            i = order[k]; vals[s, k] = Sw[s, i]; added[i] = True; c += 1
            m = adj[i]
            for j in range(64):
                if (m >> np.uint64(j)) & np.uint64(1):
                    if added[j]:
                        ri = _find(par, i); rj = _find(par, j)
                        if ri != rj:
                            par[ri] = rj; c -= 1
            nc[s, k + 1] = c
    return vals, nc


@nb.njit(cache=True)
def q_from_curves(vals, nc, grid):
    """q[s, t] = (-1)^{#components of windows with value > grid[t]}; grid ascending (last may be +inf)."""
    n = vals.shape[0]; G = len(grid); q = np.empty((n, G), np.int8); k_out = np.empty((n, G), np.int16)
    for s in range(n):
        for t in range(G):
            k = 0
            while k < 64 and vals[s, k] > grid[t]:
                k += 1
            c = nc[s, k]; q[s, t] = 1 if c % 2 == 0 else -1; k_out[s, t] = c
    return q, k_out


def chunk_states(path):
    """Load one s8_twohop chunk; return samples, chain, and for L1 = {x} u N1(x): index arrays into U2 + edges."""
    d = np.load(path); S = d['X'].astype(np.uint64); U2 = d['U2']; la = d['la'].astype(float); sV = d['s'].astype(np.int8)
    NBs, V = Q.neighbors(S)
    L1 = np.unique(np.concatenate([S, NBs[V]]))
    NB1, V1 = Q.neighbors(L1)
    idx1 = np.searchsorted(U2, NB1); idx1[~V1] = 0
    self1 = np.searchsorted(U2, L1)
    assert np.all(U2[self1] == L1) and np.all(U2[idx1[V1]] == NB1[V1])
    px = np.searchsorted(L1, S); own, col = np.nonzero(V); py = np.searchsorted(L1, NBs[own, col])
    return dict(S=S, chain=d['chain'], U2=U2, la=la, signs=dict(V=sV, M=Q.marshall(U2)), L1=L1, V1=V1, idx1=idx1,
                self1=self1, px=px, own=own, col=col, py=py)


def bond_energies(C, base):
    """e_b(z) for z in L1 (n1, 256); sum_b e_b = r_base(z)."""
    sg = C['signs'][base]; la = C['la']; idx1 = C['idx1']; self1 = C['self1']; V1 = C['V1']
    diag = np.where(V1, -0.25, 0.25) * JB[None, :]
    off = np.where(V1, 0.5 * JB[None, :] * np.exp(np.clip(la[idx1] - la[self1][:, None], -80, 80)) *
                   (sg[idx1] * sg[self1][:, None]).astype(float), 0.0)
    return diag + off


def edge_coeffs(C, base):
    """c_xy = (J_b/2) s(x)s(y) a(y)/a(x) for the sample edges (x = own sample, y = neighbour)."""
    sg = C['signs'][base][C['self1']]; la = C['la'][C['self1']]
    px, own, py = C['px'], C['own'], C['py']
    return 0.5 * JB[C['col']] * (sg[py] * sg[px][own]).astype(float) * np.exp(la[py] - la[px][own])


def delta_matrix(C, c, q):
    """per-sample E1(x) - E_L(x) for every column of q (q on L1 states)."""
    px, own, py = C['px'], C['own'], C['py']
    qx = q[px][own].astype(float); qy = q[py].astype(float)
    contrib = c[:, None] * (qx * qy - 1.0)
    starts = np.searchsorted(own, np.arange(len(px)))
    assert np.all(np.diff(own) >= 0) and np.all(np.diff(starts) > 0)
    return np.add.reduceat(contrib, starts, axis=0)
