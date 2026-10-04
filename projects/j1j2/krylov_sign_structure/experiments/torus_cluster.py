"""Tilted-torus clusters for the symmetric-sector FN/Krylov loop.

Cluster = Z^2 / <T1, T2> (T1,T2 integer vectors, N = |det|, bipartite if every lattice vector has even x+y).
Sites are enumerated in the fundamental parallelogram (row-major by (y,x) of the canonical point).
Bonds follow the convention of closed_fn_krylov_anderson.make_lattice: NN = (x+1,y),(x,y+1);
NNN = (x+1,y+1),(x+1,y-1) from every site.
Group = all translations x the point operations (x,y)->(+-x,+-y),(+-y,+-x) that map the lattice to itself.
Only trivial characters are used here (k=0, all point irreps trivial, spin flip +).
"""
import itertools
import numpy as np

POINT_OPS = [lambda x, y: (x, y), lambda x, y: (-y, x), lambda x, y: (-x, -y), lambda x, y: (y, -x),
             lambda x, y: (-x, y), lambda x, y: (x, -y), lambda x, y: (y, x), lambda x, y: (-y, -x)]


class Cluster:
    def __init__(self, t1, t2):
        self.t1, self.t2 = tuple(t1), tuple(t2)
        det = t1[0] * t2[1] - t1[1] * t2[0]
        self.det = abs(det)
        self.N = self.det
        self._sgn = 1 if det > 0 else -1
        pts = []
        R = abs(t1[0]) + abs(t2[0]) + abs(t1[1]) + abs(t2[1]) + 1
        for y in range(-R, R + 1):
            for x in range(-R, R + 1):
                a, b = self._ab(x, y)
                if 0 <= a < self.det and 0 <= b < self.det:
                    pts.append((y, x))
        pts.sort()
        self.coords = [(x, y) for (y, x) in pts]
        assert len(self.coords) == self.N, (len(self.coords), self.N)
        self.index = {c: i for i, c in enumerate(self.coords)}
        t1, t2 = self.t1, self.t2
        # bipartite check
        assert (t1[0] + t1[1]) % 2 == 0 and (t2[0] + t2[1]) % 2 == 0, "cluster not bipartite"
        self.nn, self.nnn = [], []
        for (x, y) in self.coords:
            i = self.red(x, y)
            self.nn += [(i, self.red(x + 1, y)), (i, self.red(x, y + 1))]
            self.nnn += [(i, self.red(x + 1, y + 1)), (i, self.red(x + 1, y - 1))]
        self.mask = sum(1 << i for i, (x, y) in enumerate(self.coords) if (x + y) % 2 == 0)
        # generators: unit translations + point ops preserving the lattice
        self.gens = {}
        self.gens["Tx"] = self.perm(lambda x, y: (x + 1, y))
        self.gens["Ty"] = self.perm(lambda x, y: (x, y + 1))
        self.point_gens = []
        for k, f in enumerate(POINT_OPS[1:], 1):
            if self._preserves(f):
                self.point_gens.append((k, self.perm(f)))
        self.group = self._closure([self.gens["Tx"], self.gens["Ty"]] + [p for _, p in self.point_gens])
        self.n_trans = self.N
        self.n_point = len(self.group) // self.N

    def _ab(self, x, y):
        t1, t2 = self.t1, self.t2
        # (x,y) = a/det*T1 + b/det*T2 with integers a,b (scaled by det), sign-normalised
        a = (x * t2[1] - y * t2[0]) * self._sgn
        b = (t1[0] * y - t1[1] * x) * self._sgn
        return a, b

    def red(self, x, y):
        a, b = self._ab(x, y)
        qa, qb = a // self.det, b // self.det
        xr = x - qa * self.t1[0] - qb * self.t2[0]
        yr = y - qa * self.t1[1] - qb * self.t2[1]
        return self.index[(xr, yr)]

    def perm(self, f):
        return np.array([self.red(*f(x, y)) for (x, y) in self.coords], dtype=np.int64)

    def _in_lattice(self, x, y):
        a, b = self._ab(x, y)
        return a % self.det == 0 and b % self.det == 0

    def _preserves(self, f):
        return self._in_lattice(*f(*self.t1)) and self._in_lattice(*f(*self.t2))

    def _closure(self, gens):
        ident = tuple(range(self.N))
        seen = {ident}
        frontier = [np.arange(self.N)]
        out = [np.arange(self.N)]
        while frontier:
            nxt = []
            for p in frontier:
                for g in gens:
                    q = g[p]
                    t = tuple(q)
                    if t not in seen:
                        seen.add(t); nxt.append(q); out.append(q)
            frontier = nxt
        return np.array(out)

    def order(self, p):
        q = p.copy(); k = 1
        while not np.array_equal(q, np.arange(self.N)):
            q = p[q]; k += 1
        return k

    def make_basis(self, k=(0, 0), point_sectors=None, flip=1, hamming=None):
        """k in units of the translation-generator order; point_sectors: dict k->sector for point gens (default 0)."""
        import lattice_symmetries as ls
        syms = [ls.Symmetry(list(self.gens["Tx"]), sector=int(k[0])),
                ls.Symmetry(list(self.gens["Ty"]), sector=int(k[1]))]
        for kk, p in self.point_gens:
            syms.append(ls.Symmetry(list(p), sector=int((point_sectors or {}).get(kk, 0))))
        group = ls.Group(syms)
        basis = ls.SpinBasis(group, number_spins=self.N, hamming_weight=hamming or self.N // 2,
                             spin_inversion=(None if flip == 0 else int(flip)))
        return basis, len(group)
