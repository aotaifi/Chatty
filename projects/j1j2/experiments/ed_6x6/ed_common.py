"""Common helpers for the symmetric exact diagonalisation of the spin-1/2
J1-J2 Heisenberg model on the periodic L x L square lattice.

Conventions (shared by the ED driver, the post-processing and psi0_6x6.py):
  * site index i = x + L*y  (x, y in 0..L-1).  The lattice and the A1 ground
    state are invariant under x<->y, so row- vs column-major does not matter.
  * a configuration is a uint64 bitstring; bit i = 1 <=> spin up on site i.
  * H = J1 sum_<ij> S_i.S_j + J2 sum_<<ij>> S_i.S_j, S = sigma/2, in the plain
    sigma^z basis (no Marshall rotation).
  * Marshall sign M(x) = (-1)^{# up spins on sublattice A}, A = {(x+y) even}.

lattice_symmetries (v0.8.4, Westerhout) is imported lazily so that the pure
numpy parts (lattice, permutations, lookup tables) are usable without it.
"""
import numpy as np


def site(x, y, L):
    return (x % L) + L * (y % L)


def bonds(L):
    nn, nnn = [], []
    for y in range(L):
        for x in range(L):
            i = site(x, y, L)
            nn.append((i, site(x + 1, y, L)))
            nn.append((i, site(x, y + 1, L)))
            nnn.append((i, site(x + 1, y + 1, L)))
            nnn.append((i, site(x + 1, y - 1, L)))
    return nn, nnn


def perm_from_map(f, L):
    """p[i] = image site of i under the lattice map (x,y)->f(x,y)."""
    p = np.empty(L * L, dtype=np.int64)
    for y in range(L):
        for x in range(L):
            xx, yy = f(x, y)
            p[site(x, y, L)] = site(xx, yy, L)
    return p


def generators(L):
    return {
        "Tx": perm_from_map(lambda x, y: (x + 1, y), L),
        "Ty": perm_from_map(lambda x, y: (x, y + 1), L),
        "C4": perm_from_map(lambda x, y: (-y, x), L),      # 90 deg about site 0
        "C2": perm_from_map(lambda x, y: (-x, -y), L),
        "Sx": perm_from_map(lambda x, y: (-x, y), L),      # axis reflection
        "Sd": perm_from_map(lambda x, y: (y, x), L),       # diagonal reflection
    }


def space_group(L):
    """All L*L*8 permutations of the p4m space group of the L x L torus."""
    g = generators(L)
    pts = []
    r = np.arange(L * L)
    for k in range(4):
        for refl in (False, True):
            q = r.copy()
            for _ in range(k):
                q = g["C4"][q]
            if refl:
                q = g["Sx"][q]
            pts.append(q)
    out = []
    for ty in range(L):
        for tx in range(L):
            t = r.copy()
            for _ in range(tx):
                t = g["Tx"][t]
            for _ in range(ty):
                t = g["Ty"][t]
            for q in pts:
                out.append(t[q])  # first point op, then translation
    out = np.array(out)
    assert len({tuple(p) for p in out}) == L * L * 8
    return out


def sublattice_mask(L):
    m = 0
    for y in range(L):
        for x in range(L):
            if (x + y) % 2 == 0:
                m |= 1 << site(x, y, L)
    return m


def byte_tables(perms, n):
    """Lookup tables T[g, b, v] = image bits of byte value v at byte position b
    under permutation g (bit i -> bit perms[g, i]).  image(x) = OR_b T[g,b,byte_b(x)]."""
    nb = (n + 7) // 8
    G = perms.shape[0]
    T = np.zeros((G, nb, 256), dtype=np.uint64)
    vals = np.arange(256, dtype=np.uint64)
    for b in range(nb):
        for k in range(8):
            i = 8 * b + k
            if i >= n:
                break
            bit = (vals >> np.uint64(k)) & np.uint64(1)
            T[:, b, :] |= bit[None, :] << perms[:, i].astype(np.uint64)[:, None]
    return T


def all_images(x, T, n, with_flip=True):
    """x: (B,) uint64 -> (B, G[*2]) uint64 images under the space group [x spin flip]."""
    x = np.asarray(x, dtype=np.uint64)
    nb = T.shape[1]
    img = np.zeros((x.shape[0], T.shape[0]), dtype=np.uint64)
    for b in range(nb):
        byte = ((x >> np.uint64(8 * b)) & np.uint64(255)).astype(np.intp)
        img |= T[:, b, :][:, byte].T
    if with_flip:
        full = np.uint64((1 << n) - 1)
        img = np.concatenate([img, img ^ full], axis=1)
    return img


# ----------------------------------------------------------------------------
# lattice_symmetries wrappers
# ----------------------------------------------------------------------------
HEIS = np.array([[0.25, 0, 0, 0],
                 [0, -0.25, 0.5, 0],
                 [0, 0.5, -0.25, 0],
                 [0, 0, 0, 0.25]])

# sectors: (kx, ky) in units of 2pi/L * k ; characters are all real (+-1).
# point: 'A1','A2','B1','B2' use C4 + Sx ; 'E' uses C2=-1, Sx=+1 (k=0 only);
# 'none' uses translations only.
POINT = {"A1": (0, 0), "A2": (0, 1), "B1": (2, 0), "B2": (2, 1)}


def make_basis(L, k=(0, 0), point="A1", flip=1, hamming=None):
    import lattice_symmetries as ls
    n = L * L
    g = generators(L)
    if hamming is None:
        hamming = n // 2
    syms = [ls.Symmetry(list(g["Tx"]), sector=int(k[0])),
            ls.Symmetry(list(g["Ty"]), sector=int(k[1]))]
    if point in POINT:
        c4, sx = POINT[point]
        syms.append(ls.Symmetry(list(g["C4"]), sector=c4))
        syms.append(ls.Symmetry(list(g["Sx"]), sector=sx))
    elif point == "E":
        syms.append(ls.Symmetry(list(g["C2"]), sector=1))
        syms.append(ls.Symmetry(list(g["Sx"]), sector=0))
    elif point == "none":
        pass
    else:
        raise ValueError(point)
    group = ls.Group(syms)
    basis = ls.SpinBasis(group, number_spins=n, hamming_weight=hamming,
                         spin_inversion=(None if flip == 0 else int(flip)))
    return basis, len(group)


def make_operator(basis, L, J1=1.0, J2=0.5):
    import lattice_symmetries as ls
    nn, nnn = bonds(L)
    terms = [ls.Interaction(J1 * HEIS, [list(b) for b in nn]),
             ls.Interaction(J2 * HEIS, [list(b) for b in nnn])]
    return ls.Operator(basis, terms)


# ----------------------------------------------------------------------------
# independent full-basis H application (no symmetry), for local-energy checks
# ----------------------------------------------------------------------------
def local_connections(x, L, J1=1.0, J2=0.5):
    """For each config x (B,) return diag (B,), neighbours (B, nb), coeff (B, nb)
    with coeff = 0 for parallel bonds (neighbour then = x, harmless)."""
    x = np.asarray(x, dtype=np.uint64)
    nn, nnn = bonds(L)
    bl = [(i, j, J1) for i, j in nn] + [(i, j, J2) for i, j in nnn]
    I = np.array([b[0] for b in bl], dtype=np.uint64)
    Jb = np.array([b[1] for b in bl], dtype=np.uint64)
    Jv = np.array([b[2] for b in bl])
    si = (x[:, None] >> I[None, :]) & np.uint64(1)
    sj = (x[:, None] >> Jb[None, :]) & np.uint64(1)
    anti = si != sj
    diag = np.where(anti, -0.25, 0.25) @ Jv
    mask = (np.uint64(1) << I) | (np.uint64(1) << Jb)
    nbr = np.where(anti, x[:, None] ^ mask[None, :], x[:, None])
    coeff = np.where(anti, 0.5 * Jv[None, :], 0.0)
    return diag, nbr, coeff


def rep_and_orbit(x, T, n, chunk=20000):
    """Canonical representative (min over space group x spin flip) and orbit size."""
    x = np.asarray(x, dtype=np.uint64)
    reps = np.empty(x.shape[0], dtype=np.uint64)
    orb = np.empty(x.shape[0], dtype=np.int64)
    for s in range(0, x.shape[0], chunk):
        img = np.sort(all_images(x[s:s + chunk], T, n), axis=1)
        reps[s:s + chunk] = img[:, 0]
        orb[s:s + chunk] = 1 + np.count_nonzero(img[:, 1:] != img[:, :-1], axis=1)
    return reps, orb


def amplitude_table(ls_states, v, L, chunk=20000):
    """Fully symmetric (k=0, A1, flip=+1) sector only: psi(x) = v_r / sqrt(|orbit(r)|)
    for every x in the orbit of representative r.  Returns (sorted canonical reps, amp, orbit)."""
    n = L * L
    T = byte_tables(space_group(L), n)
    reps, orb = rep_and_orbit(ls_states, T, n, chunk=chunk)
    amp = np.asarray(v, dtype=np.float64) / np.sqrt(orb)
    o = np.argsort(reps)
    return reps[o], amp[o], orb[o]
