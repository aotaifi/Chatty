"""Shared exact 4x4 periodic J1-J2 (J2/J1=0.5) helpers, S^z=0 sector (D=12870).
Copied from ../learning_ladder/rung2_sampled_sr_loop_4x4.py (same conventions)."""
import hashlib
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

L = 4; N = 16; B = 2


def red(x, y):
    return (x % L) + L * (y % L)


NN = []; NNN = []
for yy in range(L):
    for xx in range(L):
        i = red(xx, yy)
        NN += [(i, red(xx + 1, yy)), (i, red(xx, yy + 1))]
        NNN += [(i, red(xx + 1, yy + 1)), (i, red(xx + 1, yy - 1))]
MASK = 0
for yy in range(L):
    for xx in range(L):
        if (xx + yy) % 2 == 0:
            MASK |= 1 << red(xx, yy)

BASIS = np.array([s for s in range(1 << N) if s.bit_count() == N // 2], dtype=np.uint32)
D = len(BASIS)
S2I = np.full(1 << N, -1, dtype=np.int32); S2I[BASIS] = np.arange(D, dtype=np.int32)


def build_H(J2):
    diag = np.zeros(D); rr = []; cc = []; vv = []; rows = np.arange(D, dtype=np.int32)
    for bonds, J in ((NN, 1.0), (NNN, J2)):
        for u, v in bonds:
            bu = (BASIS >> u) & 1; bv = (BASIS >> v) & 1; same = bu == bv
            diag += J * np.where(same, .25, -.25)
            sel = ~same; r = rows[sel]
            t = BASIS[sel] ^ (np.uint32(1 << u) | np.uint32(1 << v))
            rr.append(r); cc.append(S2I[t]); vv.append(np.full(r.size, .5 * J))
    rr = np.concatenate(rr); cc = np.concatenate(cc); vv = np.concatenate(vv)
    H = sp.coo_matrix((np.r_[vv, diag], (np.r_[rr, rows], np.r_[cc, rows])), shape=(D, D)).tocsr()
    up = sp.triu(H - sp.diags(diag), k=1).tocoo()
    return H, diag, up.row.astype(np.int32), up.col.astype(np.int32), up.data.astype(float)


def marshall():
    return np.array([1 if (int(s) & MASK).bit_count() % 2 == 0 else -1 for s in BASIS], dtype=np.int8)


def ground(A, v0=None, tol=2e-11):
    ew, ev = sla.eigsh(A, k=1, which='SA', v0=v0, tol=tol, maxiter=300000)
    v = np.asarray(ev[:, 0], float); v /= np.linalg.norm(v)
    return float(ew[0]), v


def canonical(s):
    s = np.asarray(s, dtype=np.int8).copy()
    if s[0] < 0: s = -s
    return s


def sign_hash(s):
    return hashlib.sha1(np.packbits((s > 0).astype(np.uint8)).tobytes()).hexdigest()[:12]


def bits2x(ss):
    a = np.asarray(ss, np.uint32).reshape(-1, 1)
    return 2.0 * (((a >> np.arange(N, dtype=np.uint32)) & 1).astype(np.float64)) - 1.0


def guide_from_psi(psi):
    """complex psi on the basis -> (amp=|psi|, sign=sgn Re(psi e^{-i th0}), imag_frac).
    th0 = arg(sum psi^2)/2 removes the global phase (oracle-free)."""
    th0 = 0.5 * np.angle(np.sum(psi * psi))
    pr = psi * np.exp(-1j * th0)
    amp = np.abs(psi); amp = amp / np.linalg.norm(amp)
    sg = canonical(np.where(pr.real >= 0, 1, -1).astype(np.int8))
    imf = float(np.sum(pr.imag ** 2) / np.sum(np.abs(pr) ** 2))
    return amp, sg, imf
