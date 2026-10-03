"""Exact ground state psi0(x) of the spin-1/2 J1-J2 model (J2/J1 = 0.5) on the
periodic 6x6 square lattice, S^z_tot = 0, evaluated on arbitrary full-basis
configurations.

Usage:
    from psi0_6x6 import Psi0
    p = Psi0("psi0_6x6_table.npz")          # sorted canonical reps + amplitudes
    a = p(x)          # x: uint64 bitstrings (bit i = spin up on site i = x+6y),
                      #    or (B,36) arrays of +-1 / 0-1 spins in site order i
    p.E0              # exact energy (total)

psi0 is real, normalised over the full C(36,18)-dim S^z=0 basis, and fully
symmetric (k=0, A1 of C4v, spin-flip even), so psi0(x) = amp[canonical rep of x].
The canonical rep is the minimum uint64 over the 288 space-group images and
their spin-flipped copies.  Configurations with S^z != 0 return 0.
Global sign convention: sum_x psi0(x) * Marshall(x) > 0.
Depends only on numpy.
"""
import numpy as np


def _site(x, y, L):
    return (x % L) + L * (y % L)


def _space_group(L):
    def pm(f):
        p = np.empty(L * L, dtype=np.int64)
        for y in range(L):
            for x in range(L):
                xx, yy = f(x, y)
                p[_site(x, y, L)] = _site(xx, yy, L)
        return p
    c4 = pm(lambda x, y: (-y, x))
    sx = pm(lambda x, y: (-x, y))
    r = np.arange(L * L)
    pts = []
    for k in range(4):
        for refl in (False, True):
            q = r.copy()
            for _ in range(k):
                q = c4[q]
            if refl:
                q = sx[q]
            pts.append(q)
    out = []
    for ty in range(L):
        for tx in range(L):
            t = pm(lambda x, y: (x + tx, y + ty))
            for q in pts:
                out.append(t[q])
    return np.array(out)


def _byte_tables(perms, n):
    nb = (n + 7) // 8
    T = np.zeros((perms.shape[0], nb, 256), dtype=np.uint64)
    vals = np.arange(256, dtype=np.uint64)
    for b in range(nb):
        for k in range(8):
            i = 8 * b + k
            if i >= n:
                break
            bit = (vals >> np.uint64(k)) & np.uint64(1)
            T[:, b, :] |= bit[None, :] << perms[:, i].astype(np.uint64)[:, None]
    return T


class Psi0:
    def __init__(self, path):
        d = np.load(path)
        self.L = int(d["L"])
        self.n = self.L * self.L
        self.reps = d["reps"]
        self.amp = d["amp"]
        self.E0 = float(d["E0"])
        self._T = _byte_tables(_space_group(self.L), self.n)
        self._full = np.uint64((1 << self.n) - 1)

    def to_bits(self, x):
        x = np.asarray(x)
        if x.ndim >= 1 and x.shape[-1] == self.n and x.dtype != np.uint64:
            up = (x > 0).astype(np.uint64)
            w = np.uint64(1) << np.arange(self.n, dtype=np.uint64)
            return (up * w).sum(axis=-1, dtype=np.uint64)
        return x.astype(np.uint64)

    def canonical(self, bits):
        bits = np.atleast_1d(np.asarray(bits, dtype=np.uint64))
        img = np.zeros((bits.shape[0], self._T.shape[0]), dtype=np.uint64)
        for b in range(self._T.shape[1]):
            byte = ((bits >> np.uint64(8 * b)) & np.uint64(255)).astype(np.intp)
            img |= self._T[:, b, :][:, byte].T
        return np.minimum(img.min(axis=1), (img ^ self._full).min(axis=1))

    def __call__(self, x, chunk=8192):
        bits = np.atleast_1d(self.to_bits(x))
        out = np.zeros(bits.shape[0], dtype=np.float64)
        for s in range(0, bits.shape[0], chunk):
            b = bits[s:s + chunk]
            r = self.canonical(b)
            idx = np.searchsorted(self.reps, r)
            idx = np.minimum(idx, self.reps.shape[0] - 1)
            hit = self.reps[idx] == r
            out[s:s + chunk] = np.where(hit, self.amp[idx], 0.0)
        return out

    def marshall(self, x):
        bits = np.atleast_1d(self.to_bits(x))
        m = 0
        for y in range(self.L):
            for xx in range(self.L):
                if (xx + y) % 2 == 0:
                    m |= 1 << _site(xx, y, self.L)
        a = bits & np.uint64(m)
        cnt = np.zeros(bits.shape[0], dtype=np.int64)
        for i in range(self.n):
            cnt += ((a >> np.uint64(i)) & np.uint64(1)).astype(np.int64)
        return np.where(cnt % 2 == 0, 1, -1)
