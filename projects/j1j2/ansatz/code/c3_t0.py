"""C3-T0 (PLAN_ansatz.md): do vison (Z2 flux) fluctuations of a projected Z2 spin liquid carry the missing physics?
  python3 c3_t0.py OUTDIR
Exact 4x4 PBC J1-J2, J2 = 0.5, Sz = 0. Gutzwiller-projected BCS state (Hu-Becca-Parola-Sorella Z2Azz13 type):
NN hopping t = 1, NN d_{x2-y2} pairing D1 (+ on x bonds, - on y bonds), NNN d_xy pairing D2 (+ on (1,1), - on (1,-1)),
chemical potential mu, spinon BCs (P/A in x, y) as sign flips on boundary-crossing bonds.
Nambu psi = (c_up, c_dn^+), H_MF = psi^+ [[T, Delta], [Delta, -T]] psi; GS annihilated by positive-energy
gamma_k = sum_i A_ik c_i,up + B_ik c^+_i,dn  ->  GS = exp(sum f_ij c^+_i,up c^+_j,dn)|0>,  f = -(A^{-1})^T B^T.
Projected amplitude on spin config (basis = prod_{site ascending} c^+_{i,s_i}|0>):
  psi(x) = sgn(perm sorting (u1,d1,u2,d2,...) to site order) * det f[u_a, d_b].
Spans (0), (S), (V), (R) as in the PLAN; Ritz by generalised eigenproblem with S-eigenvalue cut 1e-10 * max.
"""
import sys, os, json, time, itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import minimize
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a):
    s = f'[{time.time() - T0:7.1f}s] ' + ' '.join(str(x) for x in a)
    print(s, flush=True); open(os.path.join(OUT, 'c3_t0.log'), 'a').write(s + '\n')
res = {}

# ====================================================================== 0. brute-force checks of f and the sign rule
def perm_sign_rows(seq):                       # seq (M, K) distinct labels per row -> parity of sorting permutation
    inv = np.zeros(seq.shape[0], np.int64)
    K = seq.shape[1]
    for a in range(K):
        inv += (seq[:, a:a + 1] > seq[:, a + 1:]).sum(1)
    return 1 - 2 * (inv & 1)

def f_from_TD(T, Dl):
    n = T.shape[0]
    M = np.block([[T, Dl], [Dl, -T]])
    w, W = np.linalg.eigh(M)
    pos = W[:, n:]                             # n positive-energy modes (spectrum is +-symmetric)
    A, B = pos[:n], pos[n:]
    gap = float(min(abs(w[n - 1]), abs(w[n])))
    cA = float(np.linalg.cond(A))
    f = -(np.linalg.pinv(A)).T @ B.T             # only meaningful when cond(A) is finite (<0|GS> != 0)
    return f, gap, cA

def neg_modes(T, Dl):
    n = T.shape[0]
    w, W = np.linalg.eigh(np.block([[T, Dl], [Dl, -T]]))
    return W[:, :n], float(min(abs(w[n - 1]), abs(w[n])))

def popcount(a):
    c = np.zeros(a.shape, np.int64)
    for k in range(40): c += (a >> k) & 1
    return c

def slater_sign(Usets, nsite):
    """sign of <x| c+_{u1,up}..c+_{un,up} c_{u1,dn}..c_{un,dn} prod_{i asc} c+_{i,dn}|0>, modes 2i+s, JW ascending."""
    M = Usets.shape[0]; st = np.full(M, sum(1 << (2 * i + 1) for i in range(nsite)), np.int64); sg = np.ones(M, np.int64)
    ops = [(2 * Usets[:, a] + 1, False) for a in range(Usets.shape[1] - 1, -1, -1)] + \
          [(2 * Usets[:, a], True) for a in range(Usets.shape[1] - 1, -1, -1)]
    for m, dag in ops:
        occ = (st >> m) & 1
        assert np.all(occ == (0 if dag else 1))
        sg *= 1 - 2 * (popcount(st & ((np.int64(1) << m) - 1)) & 1)
        st ^= (np.int64(1) << m)
    return st, sg

def slater_amp(T, Dl, Usets, sg):
    n = T.shape[0]; Wn, gap = neg_modes(T, Dl)
    rows = np.concatenate([Usets, Usets + n], axis=1)       # [U[u,:]; V[u,:]]
    return sg * np.linalg.det(Wn[rows]), gap

def brute_check(nsite=6, seed=3):
    rng = np.random.default_rng(seed)
    T = rng.normal(size=(nsite, nsite)); T = (T + T.T) / 2
    Dl = rng.normal(size=(nsite, nsite)); Dl = (Dl + Dl.T) / 2
    nm = 2 * nsite; dim = 1 << nm
    states = np.arange(dim, dtype=np.int64)
    def cdag(m):                               # JW, mode order (0up, 0dn, 1up, 1dn, ...), mode m = 2i + s
        occ = (states >> m) & 1
        low = np.zeros(dim, np.int64)
        for k in range(m): low += (states >> k) & 1
        r = np.nonzero(occ == 0)[0]
        return sp.csr_matrix(((1 - 2 * (low[r] & 1)).astype(float), (states[r] | (1 << m), states[r])), shape=(dim, dim))
    cd = [cdag(m) for m in range(nm)]; c = [x.T.tocsr() for x in cd]
    H = sp.csr_matrix((dim, dim))
    for i in range(nsite):
        for j in range(nsite):
            H = H + T[i, j] * (cd[2 * i] @ c[2 * j] + cd[2 * i + 1] @ c[2 * j + 1])
            pr = cd[2 * i] @ cd[2 * j + 1]
            H = H + Dl[i, j] * (pr + pr.T)
    w, v = sla.eigsh(H.tocsc(), k=2, which='SA', tol=1e-12)
    o = np.argsort(w); w, v = w[o], v[:, o]; g = v[:, 0]
    f, gap, cA = f_from_TD(T, Dl)
    out = dict(fock_gap=float(w[1] - w[0]), bdg_gap=gap, condA=cA)
    # (a) full fixed-N sector N = nsite (n_up = n_dn = nsite/2, double occupancy allowed)
    npair = nsite // 2
    pc_up = np.zeros(dim, int); pc_dn = np.zeros(dim, int)
    for i in range(nsite):
        pc_up += (states >> (2 * i)) & 1; pc_dn += (states >> (2 * i + 1)) & 1
    for label, sel in (('fixedN', (pc_up == npair) & (pc_dn == npair)),
                       ('projected', (pc_up == npair) & (pc_dn == npair) & (pc_up + pc_dn == nsite))):
        if label == 'projected':
            single = np.ones(dim, bool)
            for i in range(nsite): single &= (((states >> (2 * i)) & 1) + ((states >> (2 * i + 1)) & 1)) == 1
            sel = sel & single
        idx = np.nonzero(sel)[0]
        U = np.array([[i for i in range(nsite) if (s >> (2 * i)) & 1] for s in states[idx]])
        Dn = np.array([[i for i in range(nsite) if (s >> (2 * i + 1)) & 1] for s in states[idx]])
        seq = np.empty((idx.size, 2 * npair), np.int64); seq[:, 0::2] = 2 * U; seq[:, 1::2] = 2 * Dn + 1
        det = np.linalg.det(f[U[:, :, None], Dn[:, None, :]])
        cands = [('right_sign', perm_sign_rows(seq) * det), ('broken_no_permsign', det)]
        if label == 'projected':
            st, sg = slater_sign(U, nsite); assert np.all(st == states[idx])
            cands += [('slater', slater_amp(T, Dl, U, sg)[0]), ('slater_broken_nosign', slater_amp(T, Dl, U, np.ones_like(sg))[0])]
        for vname, amp in cands:
            a = g[idx]
            fid = float(abs(a @ amp) / (np.linalg.norm(a) * np.linalg.norm(amp)))
            out[f'{label}_{vname}_fidelity'] = fid
        out[f'{label}_weight_of_GS'] = float(np.sum(g[idx] ** 2))
    return out

bc = brute_check()
res['brute_force_6site'] = bc
log('brute-force 6-site check', bc)

# ====================================================================== 1. 4x4 Heisenberg (code from small_pretests.py)
LX = LY = 4; N = LX * LY; J2 = 0.5
def site(x, y): return (x % LX) + LX * (y % LY)
NN = [(site(x, y), site(x + 1, y)) for y in range(LY) for x in range(LX)] + \
     [(site(x, y), site(x, y + 1)) for y in range(LY) for x in range(LX)]
NNN = [(site(x, y), site(x + 1, y + 1)) for y in range(LY) for x in range(LX)] + \
      [(site(x, y), site(x + 1, y - 1)) for y in range(LY) for x in range(LX)]
allx = np.arange(1 << N, dtype=np.int64)
pc = np.zeros(1 << N, np.int8)
for i in range(N): pc += ((allx >> i) & 1).astype(np.int8)
basis = allx[pc == N // 2]; del allx, pc
Dim = basis.size
def ham(bonds, J):
    rows, cols, vals = [], [], []; dg = np.zeros(Dim)
    for (i, j) in bonds:
        bi = (basis >> i) & 1; bj = (basis >> j) & 1; anti = bi != bj
        dg += J * np.where(anti, -0.25, 0.25)
        r = np.nonzero(anti)[0]; y = basis[r] ^ ((1 << i) | (1 << j))
        rows.append(r); cols.append(np.searchsorted(basis, y)); vals.append(np.full(r.size, 0.5 * J))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(Dim, Dim)) + sp.diags(dg)
H = (ham(NN, 1.0) + ham(NNN, J2)).tocsr()
E, V = sla.eigsh(H, k=2, which='SA', tol=1e-12); o = np.argsort(E); E, V = E[o], V[:, o]
psi0 = V[:, 0]; E0 = float(E[0])
log('D', Dim, 'E0/N', E0 / N, 'gap', E[1] - E[0])
assert abs(E0 / N + 0.5286202095) < 1e-8
res['E0_site'] = E0 / N
MA = sum(1 << site(x, y) for y in range(LY) for x in range(LX) if (x + y) % 2 == 0)
def par(v, m):
    c = np.zeros(v.size, np.int64); w = v & m
    for i in range(N): c += (w >> i) & 1
    return 1 - 2 * (c & 1)
sN = par(basis, MA)
def WM(v, sgn=sN):
    v = v / np.linalg.norm(v); s = np.sign(np.sum(v * np.abs(v) * sgn)) or 1.0
    return float(np.sum(v ** 2 * ((np.sign(v) * s) != sgn) * (v != 0)))
res['W_M_psi0'] = WM(psi0)

# configuration tables (bit 1 = up)
bits = ((basis[:, None] >> np.arange(N)) & 1).astype(bool)
Uidx = np.array([np.nonzero(b)[0] for b in bits]); Didx = np.array([np.nonzero(~b)[0] for b in bits])
seq = np.empty((Dim, N), np.int64); seq[:, 0::2] = Uidx; seq[:, 1::2] = Didx
PSIGN = perm_sign_rows(seq)
_st, SSIGN = slater_sign(Uidx, N)
_tgt = np.zeros(Dim, np.int64)
for i in range(N): _tgt |= np.where(bits[:, i], np.int64(1) << (2 * i), np.int64(1) << (2 * i + 1))
assert np.all(_st == _tgt)

# ====================================================================== 2. lattice, bonds with BC signs, mean field
# bond list: (i, j, kind, x, y, dx, dy); kind 0 = NN x, 1 = NN y, 2 = NNN (1,1), 3 = NNN (1,-1)
BONDS = []
for y in range(LY):
    for x in range(LX):
        for kind, (dx, dy) in enumerate(((1, 0), (0, 1), (1, 1), (1, -1))):
            BONDS.append((site(x, y), site(x + dx, y + dy), kind, x, y, dx, dy))
NB = len(BONDS)
def bc_sign(b, bcx, bcy):                      # bcx, bcy: 0 periodic, 1 antiperiodic
    _, _, _, x, y, dx, dy = b
    s = 1
    if not (0 <= x + dx < LX) and bcx: s = -s
    if not (0 <= y + dy < LY) and bcy: s = -s
    return s
def mf_mats(D1, D2, mu, bcx, bcy, fac=None):
    T = -mu * np.eye(N); Dl = np.zeros((N, N))
    for k, b in enumerate(BONDS):
        i, j, kind = b[:3]; s = bc_sign(b, bcx, bcy) * (1.0 if fac is None else fac[k])
        if kind < 2:
            T[i, j] += s; T[j, i] += s
            d = D1 if kind == 0 else -D1
        else:
            d = D2 if kind == 2 else -D2
        Dl[i, j] += s * d; Dl[j, i] += s * d
    return T, Dl
def proj_state(T, Dl, permsign=True, form='slater'):
    if form == 'f':
        f, gap, cA = f_from_TD(T, Dl)
        if not np.isfinite(cA) or cA > 1e10: return None, gap, cA
        amp = np.linalg.det(f[Uidx[:, :, None], Didx[:, None, :]])
        if permsign: amp = amp * PSIGN
    else:
        amp, gap = slater_amp(T, Dl, Uidx, SSIGN if permsign else np.ones(Dim, np.int64)); cA = None
        if gap < 1e-9: return None, gap, cA                  # degenerate mean-field ground state: ill-defined
    nrm = np.linalg.norm(amp)
    if not np.isfinite(nrm) or nrm == 0: return None, gap, cA
    return amp / nrm, gap, cA
def state(p, bcx, bcy, fac=None, permsign=True, form='slater'):
    return proj_state(*mf_mats(p[0], p[1], p[2], bcx, bcy, fac), permsign=permsign, form=form)
def energy(v): return float(v @ (H @ v) / (v @ v))

xc = {}
for (bcx, bcy) in ((0, 0), (0, 1), (1, 1)):
    for p in ((0.6, 0.3, 0.2), (1.2, -0.4, -0.3)):
        va, ga, _ = state(p, bcx, bcy); vb, gb, cA = state(p, bcx, bcy, form='f')
        xc[f'bc{bcx}{bcy}_{p}'] = dict(condA=cA, fid=None if (va is None or vb is None) else float(abs(va @ vb)))
res['check_4x4_f_vs_slater'] = xc
log('4x4 f-form vs Slater-form fidelity', xc)
# ---------------------------------------------------------------------- check (i): Marshall for bipartite pairing
chk = {}
for (bcx, bcy) in ((0, 1), (1, 0), (1, 1), (0, 0)):
    v, gap, cA = state((0.6, 0.0, 0.0), bcx, bcy)
    vb, _, _ = state((0.6, 0.0, 0.0), bcx, bcy, permsign=False)
    if v is None: chk[f'bc{bcx}{bcy}'] = dict(gap=gap, condA=cA, state='ill-defined'); continue
    chk[f'bc{bcx}{bcy}'] = dict(gap=gap, condA=cA, W_M=WM(v), W_M_broken_permsign=WM(vb),
                                overlap2=float((psi0 @ v) ** 2), E_site=energy(v) / N)
res['check_i_marshall_D2_0_mu_0_D1_0.6'] = chk
log('check (i)', chk)

# ====================================================================== 3. (0) best single state per BC sector
def negov(p, bcx, bcy):
    v, gap, cA = state(p, bcx, bcy)
    return 0.0 if v is None else -float((psi0 @ v) ** 2)
rng = np.random.default_rng(11)
starts = [(0.6, 0.0, 0.0), (0.6, 0.2, 0.0), (1.0, 0.4, 0.1), (0.3, -0.2, 0.0), (1.5, 0.6, -0.1)] + \
         [tuple(rng.uniform([-0.1, -1, -0.5], [2, 1, 0.5])) for _ in range(3)]
best = {}
for (bcx, bcy) in ((0, 0), (0, 1), (1, 0), (1, 1)):
    runs = []
    for s0 in starts:
        r = minimize(negov, s0, args=(bcx, bcy), method='Nelder-Mead',
                     options=dict(xatol=1e-5, fatol=1e-9, maxiter=1500, maxfev=3000))
        runs.append((float(-r.fun), [float(q) for q in r.x], int(r.nfev)))
    runs.sort(key=lambda q: -q[0])
    v, gap, cA = state(runs[0][1], bcx, bcy)
    best[(bcx, bcy)] = dict(overlap2=runs[0][0], params=runs[0][1], E_site=energy(v) / N, gap=gap, condA=cA,
                            all_starts=[(q[0], q[1]) for q in runs], W_M=WM(v))
    log('BC', bcx, bcy, 'best |<psi0|psiP>|^2', runs[0][0], 'params (D1,D2,mu)', runs[0][1], 'E/N', energy(v) / N,
        'starts', [round(q[0], 6) for q in runs])
res['single_by_bc'] = {f'{k[0]}{k[1]}': v for k, v in best.items()}
kbest = max(best, key=lambda k: best[k]['overlap2'])
log('best sector', kbest)

# ====================================================================== 4. symmetry group (16 translations x D4)
def pg_maps():
    ops = []
    for (a, b, c, d) in ((1, 0, 0, 1), (0, -1, 1, 0), (-1, 0, 0, -1), (0, 1, -1, 0),
                         (-1, 0, 0, 1), (1, 0, 0, -1), (0, 1, 1, 0), (0, -1, -1, 0)):
        for tx in range(LX):
            for ty in range(LY):
                perm = np.zeros(N, int)
                for y in range(LY):
                    for x in range(LX):
                        perm[site(x, y)] = site(a * x + b * y + tx, c * x + d * y + ty)
                ops.append(perm)
    return ops
OPS = pg_maps()
IMG = []
for perm in OPS:
    img = np.zeros(Dim, np.int64)
    for i in range(N): img |= ((basis >> i) & 1) << perm[i]
    IMG.append(np.searchsorted(basis, img))
CHI = np.array([float(psi0 @ psi0[ix]) for ix in IMG])
res['psi0_characters_min_abs'] = float(np.min(np.abs(CHI))); res['psi0_characters_neg'] = int(np.sum(CHI < 0))
log('psi0 characters: min |chi|', np.min(np.abs(CHI)), 'n negative', int(np.sum(CHI < 0)))
CHI = np.sign(CHI)
def symmetrise(v):
    w = np.zeros(Dim)
    for ch, ix in zip(CHI, IMG): w += ch * v[ix]
    return w

# ====================================================================== 5. vison strings
def seg_cross(p1, p2, q1, q2):
    def orient(a, b, c): return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return orient(p1, p2, q1) * orient(p1, p2, q2) < 0 and orient(q1, q2, p1) * orient(q1, q2, p2) < 0
EPS = (0.1, 0.03)
def string_mask(pl1, pl2):
    """flip mask over BONDS for an L-shaped dual string from plaquette pl1 to pl1 + (dx, dy) (x first, then y)."""
    (px, py), (dx, dy) = pl1, pl2
    a = (px + 0.5 + EPS[0], py + 0.5 + EPS[1]); b = (a[0] + dx, a[1]); c = (b[0], b[1] + dy)
    path = [(a, b), (b, c)]
    mask = np.zeros(NB, int)
    for k, (i, j, kind, x, y, ddx, ddy) in enumerate(BONDS):
        n = 0
        for (p, q) in path:
            for mx in range(-2, 3):
                for my in range(-2, 3):
                    if seg_cross((x + mx * LX, y + my * LY), (x + ddx + mx * LX, y + ddy + my * LY), p, q): n += 1
        mask[k] = n & 1
    return mask
bond_index = {}
for k, b in enumerate(BONDS): bond_index[(b[3], b[4], b[5], b[6])] = k
def bsign(mask, x, y, dx, dy):
    if dx < 0 or (dx == 0 and dy < 0): x, y, dx, dy = x + dx, y + dy, -dx, -dy
    return 1 - 2 * mask[bond_index[(x % LX, y % LY, dx, dy)]]
def flux(mask):
    plaq, tri = {}, {}
    for y in range(LY):
        for x in range(LX):
            p = bsign(mask, x, y, 1, 0) * bsign(mask, x + 1, y, 0, 1) * bsign(mask, x, y + 1, 1, 0) * bsign(mask, x, y, 0, 1)
            if p < 0: plaq[(x, y)] = -1
            ts = [bsign(mask, x, y, 1, 0) * bsign(mask, x + 1, y, 0, 1) * bsign(mask, x, y, 1, 1),
                  bsign(mask, x, y, 0, 1) * bsign(mask, x, y + 1, 1, 0) * bsign(mask, x, y, 1, 1),
                  bsign(mask, x, y, 1, 0) * bsign(mask, x, y, 0, 1) * bsign(mask, x, y + 1, 1, -1),
                  bsign(mask, x + 1, y, 0, 1) * bsign(mask, x, y + 1, 1, 0) * bsign(mask, x, y + 1, 1, -1)]
            if min(ts) < 0: tri[(x, y)] = ts
    return plaq, tri
CLASSES = [(0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]   # plaquette-pair separations up to the space group of the 4x4 torus
masks = {}
for (dx, dy) in CLASSES:
    m = string_mask((0, 0), (dx, dy))
    plaq, tri = flux(m)
    ends = {(0, 0), (dx % LX, dy % LY)}
    ok = set(plaq) == ends and set(tri) <= ends
    log('string', (dx, dy), 'flipped bonds', int(m.sum()), 'pi-flux plaquettes', sorted(plaq), 'pi triangles in',
        sorted(tri), 'OK' if ok else 'FLUX CHECK FAILED')
    res.setdefault('flux_check', {})[f'{dx},{dy}'] = dict(nflip=int(m.sum()), pi_plaquettes=sorted(plaq),
                                                         tri_plaquettes=sorted(tri), ok=bool(ok))
    masks[(dx, dy)] = m
# control of the flux check: a closed contractible dual loop must give no flux
mloop = (string_mask((0, 0), (1, 0)) + string_mask((1, 0), (0, 1)) + string_mask((1, 1), (-1, 0)) +
         string_mask((0, 1), (0, -1))) % 2
res['flux_check_closed_loop'] = dict(nflip=int(mloop.sum()), pi_plaquettes=sorted(flux(mloop)[0]))
log('closed-loop control: flipped', int(mloop.sum()), 'pi plaquettes', sorted(flux(mloop)[0]))

# ====================================================================== 6. spans and Ritz
def ritz(vecs, cut=1e-10):
    Phi = np.array([v / np.linalg.norm(v) for v in vecs]).T
    S = Phi.T @ Phi; Hm = Phi.T @ (H @ Phi)
    s, U = np.linalg.eigh(S); keep = s > cut * s.max()
    X = U[:, keep] / np.sqrt(s[keep])
    e, c = np.linalg.eigh(X.T @ Hm @ X)
    vg = Phi @ (X @ c[:, 0]); vg /= np.linalg.norm(vg)
    b = X.T @ (Phi.T @ psi0); ovmax = float(b @ b)
    return dict(n=len(vecs), rank=int(keep.sum()), E_ritz_site=float(e[0] / N), dE_site=float((e[0] - E0) / N),
                infid_best_comb=float(1 - ovmax), infid_ritz_vec=float(1 - (psi0 @ vg) ** 2))
spans = {}
v0, _, _ = state(best[kbest]['params'], *kbest)
r1 = ritz([v0]); r1['E_expect_site'] = energy(v0) / N
res['check_ii_1state_ritz_minus_expect'] = r1['E_ritz_site'] - r1['E_expect_site']
spans['0'] = r1
S_states = [state(best[k]['params'], *k)[0] for k in sorted(best)]
spans['S'] = ritz(S_states)
spans['S+symS'] = ritz(S_states + [symmetrise(v) for v in S_states])
vis, ctl = [], {s: [] for s in range(3)}
condinfo = []
for k in sorted(best):
    p = best[k]['params']
    for cl in CLASSES:
        m = masks[cl]
        v, gap, cA = state(p, *k, fac=1.0 - 2.0 * m); condinfo.append((k, cl, gap, cA))
        if v is None: log('vison state ill-defined', k, cl, gap, cA); continue
        w = symmetrise(v)
        if np.linalg.norm(w) < 1e-8: log('vison state symmetrises to zero', k, cl); continue
        vis.append(w)
        for sd in ctl:
            rr = np.random.default_rng(1000 * sd + len(vis))
            fac = np.where(m == 1, rr.uniform(0.9, 1.1, NB), 1.0)
            vr, _, _ = state(p, *k, fac=fac)
            ctl[sd].append(symmetrise(vr))
K = len(vis)
res['vison_gap_condA'] = [(f'{k[0]}{k[1]}', list(cl), g, c) for (k, cl, g, c) in condinfo]
spans['V'] = ritz(S_states + vis); spans['V']['K'] = K
for sd in ctl:
    spans[f'R_seed{sd}'] = ritz(S_states + ctl[sd]); spans[f'R_seed{sd}']['K'] = len(ctl[sd])
# control: only the symmetrised vison states without (S) and the extra 'flux-free symmetrised base' are in S+symS
res['spans'] = spans
for nm, r in spans.items(): log('span', nm, r)

# ====================================================================== 7. verdict (PLAN rule)
dV = spans['S']['dE_site'] - spans['V']['dE_site']
dR = [spans['S']['dE_site'] - spans[f'R_seed{s}']['dE_site'] for s in ctl]
dRmax = max(dR)
if dV <= dRmax: verdict = 'KILL'
elif dV >= 2 * dRmax: verdict = 'PASS'
else: verdict = 'INCONCLUSIVE'
res['verdict'] = dict(dV_lowering_site=dV, dR_lowering_site=dR, rule='PASS if dV >= 2 max(dR); KILL if dV <= max(dR)',
                      verdict=verdict)
log('VERDICT C3-T0', verdict, 'lowering vs (S): V', dV, 'R', dR)
json.dump(res, open(os.path.join(OUT, 'c3_t0.json'), 'w'), indent=1, default=str)
