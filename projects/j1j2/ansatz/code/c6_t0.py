"""C6-T0 (PLAN_ansatz.md): is the frustrated ground state a local J2 dressing of an UNFRUSTRATED parent?
  python3 c6_t0.py LX LY OUT_JSON
Exact, Sz = 0 sector, J1-J2 at J2/J1 = 0.5, PBC.
Phi = (1 + a H_J2)^{-1} psi0 (a < 0, cond 5, 10, 30, 100 from the Lanczos extreme eigenvalues of H_J2; MINRES).
Operator basis (translation sums, each unordered pair once, classes merged under the lattice reflections psi0 is
even under): XY_d = sum (SxSx + SySy), ZZ_d = sum SzSz for displacement classes |d| <= 2 sqrt 2; RING = sum over J1
plaquettes of (P + P^-1). Covariance C = <O_k O_l> - <O_k><O_l> in normalised Phi. Parent: min c^T C c with
c[XY of the class containing (1,0)] = 1; (u) unconstrained, (s) stoquastic in the Marshall frame (exact active-set
enumeration of the bound-constrained QP, cross-checked by L-BFGS-B). For each parent: GS(H_p) (eigsh, 3 lowest),
fidelity with Phi, dressed (1 + a' H_J2) GS (a' by 2x2 Rayleigh-Ritz = closed-form min E_var), E_var, overlap with
psi0, W_M, E_FN (lattice FN, own sign and guide, LOBPCG + Jacobi preconditioner, guide ratio cap 1e12).
Baselines: Marshall-FN (guide |psi0|) and (1 + a H_J2) on its ground state.
On 4x4 every pair is in the basis, so S^2 and (Sz_tot)^2 are in the span with zero variance in any singlet: the
gauge is fixed by c = 0 on the farthest class (XY and ZZ), in both arms (reported).
"""
import sys, json, time, itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.linalg import eigh
from scipy.optimize import minimize
LX, LY, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
N = LX * LY; J2 = 0.5; T0 = time.time(); RCAP = 1e12
def log(*a): print(f'[{time.time() - T0:8.1f}s] PROGRESS', *a, flush=True)
def site(x, y): return (x % LX) + LX * (y % LY)
XY_OF = [(s % LX, s // LX) for s in range(N)]

# ------------------------------------------------------------------ basis (Sz = 0) and Sz = +1 (for S^2)
allx = np.arange(1 << N, dtype=np.int64); pc = np.zeros(1 << N, np.int8)
for i in range(N): pc += ((allx >> i) & 1).astype(np.int8)
basis = allx[pc == N // 2]; basis1 = allx[pc == N // 2 + 1]; del allx, pc
D = basis.size; log('D', D)
def idx_of(y): return np.searchsorted(basis, y)
MA = sum(1 << site(x, y) for y in range(LY) for x in range(LX) if (x + y) % 2 == 0)
cnt = np.zeros(D, np.int64)
for i in range(N):
    if (MA >> i) & 1: cnt += (basis >> i) & 1
sN = (1 - 2 * (cnt & 1)).astype(float); del cnt

def bond_xy(bonds):
    rows, cols = [], []
    for (i, j) in bonds:
        anti = ((basis >> i) & 1) != ((basis >> j) & 1); r = np.nonzero(anti)[0]
        rows.append(r.astype(np.int32)); cols.append(idx_of(basis[r] ^ ((1 << i) | (1 << j))).astype(np.int32))
    r = np.concatenate(rows); c = np.concatenate(cols); del rows, cols
    return sp.csr_matrix((np.full(r.size, 0.5), (r, c)), shape=(D, D))
def bond_zz(bonds):
    dg = np.zeros(D)
    for (i, j) in bonds: dg += np.where(((basis >> i) & 1) != ((basis >> j) & 1), -0.25, 0.25)
    return dg
NN = sorted({tuple(sorted((site(x, y), site(x + 1, y)))) for y in range(LY) for x in range(LX)} |
            {tuple(sorted((site(x, y), site(x, y + 1)))) for y in range(LY) for x in range(LX)})
NNN = sorted({tuple(sorted((site(x, y), site(x + 1, y + 1)))) for y in range(LY) for x in range(LX)} |
             {tuple(sorted((site(x, y), site(x + 1, y - 1)))) for y in range(LY) for x in range(LX)})
H1 = (bond_xy(NN) + sp.diags(bond_zz(NN))).tocsr(); HJ2 = (J2 * bond_xy(NNN) + sp.diags(J2 * bond_zz(NNN))).tocsr()
H = (H1 + HJ2).tocsr(); del H1
E, V = sla.eigsh(H, k=2, which='SA', tol=1e-13); o = np.argsort(E); E, V = E[o], V[:, o]
E0 = float(E[0]); psi0 = V[:, 0]; del V
if psi0 @ (np.abs(psi0) * sN) < 0: psi0 = -psi0
log('E0/N', E0 / N, 'gap', E[1] - E[0])
res = dict(LX=LX, LY=LY, D=int(D), E0_site=E0 / N, gap=float(E[1] - E[0]), J2=J2)

def WM(v):
    v = v / np.linalg.norm(v); g = np.sign(np.sum(v * np.abs(v) * sN)) or 1.0
    return float(np.sum(v ** 2 * ((np.sign(v) * g) != sN) * (v != 0)))
def S2(v):
    w = np.zeros(basis1.size)
    for i in range(N):
        r = np.nonzero(((basis >> i) & 1) == 0)[0]
        w[np.searchsorted(basis1, basis[r] | (1 << i))] += v[r]          # targets distinct for fixed i
    return float(w @ w / (v @ v))
res['W_M_psi0'] = WM(psi0); res['S2_psi0'] = S2(psi0)

# ------------------------------------------------------------------ lattice reflections: parity of psi0
def perm_index(perm):
    y = np.zeros(D, np.int64)
    for s, t in enumerate(perm): y |= ((basis >> s) & 1) << t
    return idx_of(y)
REFL = {'Rx': lambda x, y: (-x, y), 'Ry': lambda x, y: (x, -y)}
if LX == LY: REFL['Rd'] = lambda x, y: (y, x)
res['parity'] = {}
for nm, f in REFL.items():
    p = perm_index([site(*f(*XY_OF[s])) for s in range(N)])
    v = np.empty(D); v[p] = psi0; res['parity'][nm] = float(v @ psi0)
log('psi0 reflection parities', res['parity'], 'S2', res['S2_psi0'])
G = [f for nm, f in REFL.items() if res['parity'][nm] > 1 - 1e-8]
# symmetric sector of psi0: translations, reflections, global spin flip; projector = group average with psi0's characters
SYM = [('Tx', perm_index([site(XY_OF[s][0] + 1, XY_OF[s][1]) for s in range(N)]), LX),
       ('Ty', perm_index([site(XY_OF[s][0], XY_OF[s][1] + 1) for s in range(N)]), LY),
       ('F', idx_of(basis ^ ((1 << N) - 1)), 2)]
SYM += [(nm, perm_index([site(*f(*XY_OF[s])) for s in range(N)]), 2) for nm, f in REFL.items()]
def gapply(p, v): w = np.empty_like(v); w[p] = v; return w
res['characters'] = {}
for nm, p, n in SYM: res['characters'][nm] = float(gapply(p, psi0) @ psi0)
assert all(abs(abs(x) - 1) < 1e-8 for x in res['characters'].values()), res['characters']
def Pproj(v):
    for nm, p, n in SYM:
        chi = round(res['characters'][nm]); acc = v.copy(); w = v
        for q in range(1, n): w = gapply(p, w); acc = acc + (chi ** q) * w
        v = acc / n
    return v
res['check_P_psi0'] = float(np.linalg.norm(Pproj(psi0) - psi0)); vr0 = np.random.default_rng(7).standard_normal(D)
res['check_P_idempotent'] = float(np.linalg.norm(Pproj(Pproj(vr0)) - Pproj(vr0)) / np.linalg.norm(Pproj(vr0)))
log('characters', res['characters'], 'CHECK |P psi0 - psi0|', res['check_P_psi0'], 'P^2-P', res['check_P_idempotent'])

# ------------------------------------------------------------------ displacement classes
disp = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if 0 < dx * dx + dy * dy <= 8]
def pairs_of(d):
    return {tuple(sorted((site(x, y), site(x + d[0], y + d[1])))) for y in range(LY) for x in range(LX)} - \
           {(s, s) for s in range(N)}
def orbit(d):
    o = {d, (-d[0], -d[1])}; ch = True
    while ch:
        ch = False
        for f in G:
            for e in list(o):
                for e2 in (f(*e), tuple(-q for q in f(*e))):
                    if e2 not in o: o.add(e2); ch = True
    return frozenset(o)
orbs = list({orbit(d) for d in disp})
cls = [dict(disps=sorted(o), pairs=set().union(*[pairs_of(d) for d in o])) for o in orbs]
merged = True                                                   # torus aliasing: classes sharing a pair merge
while merged:
    merged = False
    for a, b in itertools.combinations(range(len(cls)), 2):
        if cls[a]['pairs'] & cls[b]['pairs']:
            cls[a] = dict(disps=sorted(set(cls[a]['disps']) | set(cls[b]['disps'])), pairs=cls[a]['pairs'] | cls[b]['pairs'])
            del cls[b]; merged = True; break
cls.sort(key=lambda c: (min(d[0] ** 2 + d[1] ** 2 for d in c['disps']), -max(abs(d[0]) for d in c['disps'])))
for c in cls: c['r2'] = min(d[0] ** 2 + d[1] ** 2 for d in c['disps']); c['pairs'] = sorted(c['pairs'])
all_pairs = N * (N - 1) // 2; covered = sum(len(c['pairs']) for c in cls)
gauge = covered == all_pairs
log('classes', [(c['disps'][0], len(c['pairs'])) for c in cls], 'pairs covered', covered, 'of', all_pairs, 'gauge fix', gauge)

# ------------------------------------------------------------------ operators
ops = []                                                        # dict(name, M (sparse) or dg (diag), kind)
for c in cls:
    tag = 'd=(%d,%d)' % tuple(np.abs(c['disps'][0]))
    ops.append(dict(name='XY ' + tag, kind='XY', cls=c, M=bond_xy(c['pairs'])))
    ops.append(dict(name='ZZ ' + tag, kind='ZZ', cls=c, dg=bond_zz(c['pairs'])))
    log('built', tag, 'bonds', len(c['pairs']))
rows, cols = [], []
for y in range(LY):
    for x in range(LX):
        a, b, c_, d = site(x, y), site(x + 1, y), site(x + 1, y + 1), site(x, y + 1)
        m = (1 << a) | (1 << b) | (1 << c_) | (1 << d)
        ba, bb, bc, bd = [(basis >> s) & 1 for s in (a, b, c_, d)]
        yv = (basis & ~m) | (ba << b) | (bb << c_) | (bc << d) | (bd << a)
        rows.append(idx_of(yv).astype(np.int32)); cols.append(np.arange(D, dtype=np.int32))
r = np.concatenate(rows); c_ = np.concatenate(cols); del rows, cols
P = sp.csr_matrix((np.ones(r.size), (r, c_)), shape=(D, D)); del r, c_
ops.append(dict(name='RING', kind='RING', M=(P + P.T).tocsr())); del P
K = len(ops)
def apply_op(o, v): return o['M'] @ v if 'M' in o else (o['dg'][:, None] * v if v.ndim == 2 else o['dg'] * v)
for o in ops:                                                  # Marshall-frame off-diagonal signs
    if 'M' in o:
        C = o['M'].tocoo(); off = C.row != C.col; sg = C.data[off] * sN[C.row[off]] * sN[C.col[off]]
        o['n_pos'], o['n_neg'] = int((sg > 0).sum()), int((sg < 0).sum()); del C, off, sg
    else: o['n_pos'] = o['n_neg'] = 0
    o['bound'] = ('free' if o['n_pos'] + o['n_neg'] == 0 else 'le0' if o['n_neg'] == 0 else 'ge0' if o['n_pos'] == 0
                  else 'mixed')
i_J1 = next(k for k, o in enumerate(ops) if o['kind'] == 'XY' and (1, 0) in o['cls']['disps'])
names = [o['name'] for o in ops]
gfix = []
if gauge:
    far = max(range(len(cls)), key=lambda q: cls[q]['r2'])
    gfix = [k for k, o in enumerate(ops) if o['kind'] in ('XY', 'ZZ') and o['cls'] is cls[far]]
res['operators'] = [dict(name=o['name'], disps=o['cls']['disps'] if 'cls' in o else None,
                         n_bonds=len(o['cls']['pairs']) if 'cls' in o else N, offdiag_pos=o['n_pos'],
                         offdiag_neg=o['n_neg'], stoquastic_bound=o['bound']) for o in ops]
res['i_fixed'] = i_J1; res['gauge_fixed_zero'] = [names[k] for k in gfix]
log('operators', [(o['name'], o['bound']) for o in ops], 'fixed', names[i_J1], 'gauge zero', res['gauge_fixed_zero'])

# exact H in this basis (check: rebuilt H equals H on random vectors)
cH = np.zeros(K)
for k, o in enumerate(ops):
    if o['kind'] in ('XY', 'ZZ'):
        if (1, 0) in o['cls']['disps'] or (0, 1) in o['cls']['disps']: cH[k] = 1.0
        if (1, 1) in o['cls']['disps']: cH[k] = J2
def Hp_apply(c, v): return sum(c[k] * apply_op(o, v) for k, o in enumerate(ops) if c[k] != 0)
rng = np.random.default_rng(1); vr = rng.standard_normal(D)
res['check_H_rebuild'] = float(np.linalg.norm(Hp_apply(cH, vr) - H @ vr) / np.linalg.norm(H @ vr))
cbad = cH.copy(); cbad[[k for k in range(K) if ops[k]['kind'] == 'XY' and (1, 1) in ops[k]['cls']['disps']][0]] *= 1.01
res['check_H_rebuild_broken'] = float(np.linalg.norm(Hp_apply(cbad, vr) - H @ vr) / np.linalg.norm(H @ vr))
log('CHECK H rebuild', res['check_H_rebuild'], 'broken', res['check_H_rebuild_broken'])

# ------------------------------------------------------------------ covariance and parent
def cov(phi):
    phi = phi / np.linalg.norm(phi)
    Y = np.stack([apply_op(o, phi) for o in ops], axis=1)
    m = Y.T @ phi; Gm = Y.T @ Y; return Gm - np.outer(m, m), Gm, m
def parent(Cm, Gm, arm):
    fixed0 = set(gfix) | ({k for k in range(K) if ops[k]['bound'] == 'mixed'} if arm == 's' else set())
    free = [k for k in range(K) if k != i_J1 and k not in fixed0]
    bnd = [k for k in free if arm == 's' and ops[k]['bound'] in ('le0', 'ge0')]
    def solve(act):
        fr = [k for k in free if k not in act]; c = np.zeros(K); c[i_J1] = 1.0
        if fr: c[fr] = -np.linalg.lstsq(Cm[np.ix_(fr, fr)], Cm[fr, i_J1], rcond=1e-14)[0]
        return c
    def feas(c, tol=1e-12):
        return all((c[k] <= tol if ops[k]['bound'] == 'le0' else c[k] >= -tol) for k in bnd)
    best = None
    for nact in range(len(bnd) + 1):
        for act in itertools.combinations(bnd, nact):
            c = solve(set(act))
            if not feas(c): continue
            f = float(c @ Cm @ c)
            if best is None or f < best[0] - 1e-18: best = (f, c)
    f, c = best
    out = dict(arm=arm, coef={names[k]: float(c[k]) for k in range(K)}, var=f, rel_var=f / float(c @ Gm @ c),
               fixed_zero=[names[k] for k in sorted(fixed0)])
    if arm == 's':                                              # cross-check: L-BFGS-B from the bound-projected (u) solution
        x0 = solve(set()); bounds = []
        for k in free:
            b = ops[k]['bound'] if k in bnd else 'free'; bounds.append((None, 0) if b == 'le0' else (0, None) if b == 'ge0' else (None, None))
        x0f = np.array([min(x0[k], 0) if b == (None, 0) else max(x0[k], 0) if b == (0, None) else x0[k] for k, b in zip(free, bounds)])
        def obj(z):
            cc = np.zeros(K); cc[i_J1] = 1; cc[free] = z; g = 2 * Cm @ cc; return float(cc @ Cm @ cc), g[free]
        r_ = minimize(obj, x0f, jac=True, method='L-BFGS-B', bounds=bounds, options=dict(ftol=1e-15, gtol=1e-13, maxiter=20000))
        out['lbfgsb_var'] = float(r_.fun); out['active_bounds'] = [names[k] for k in bnd if abs(c[k]) < 1e-14]
    return out, c
for o in ops: o['rowmax'] = float(abs(o['M']).sum(axis=1).max()) if 'M' in o else float(np.abs(o['dg']).max())
def gs_parent(c, phi):
    """H_p applied term by term (no summed matrix: saves ~2x the operator memory at 4x6)."""
    nz = [k for k in range(K) if c[k] != 0]; nmv = [0]
    def Hp(v):
        nmv[0] += 1
        if nmv[0] % 200 == 0: log('  H_p matvecs', nmv[0])
        return sum(c[k] * apply_op(ops[k], v) for k in nz)
    phi = phi / np.linalg.norm(phi)
    opU = sla.LinearOperator((D, D), matvec=lambda v: Hp(np.ravel(v)), dtype=np.float64)
    ev, U = sla.eigsh(opU, k=2, which='SA', tol=1e-10, v0=phi.copy()); o_ = np.argsort(ev); ev, U = ev[o_], U[:, o_]
    lows = [dict(E=float(ev[q]), fid_phi=float((U[:, q] @ phi) ** 2), S2=S2(U[:, q]),
                 sector_weight=float(np.linalg.norm(Pproj(U[:, q])) ** 2)) for q in range(2)]
    shift = sum(abs(c[k]) * ops[k]['rowmax'] for k in nz) + 1.0  # Gershgorin bound: P (H_p - shift) P <= 0 in the sector
    op = sla.LinearOperator((D, D), matvec=lambda v: (lambda pv: Hp(pv) - shift * pv)(Pproj(np.ravel(v))), dtype=np.float64)
    es, Us = sla.eigsh(op, k=2, which='SA', tol=1e-10, v0=Pproj(phi)); o_ = np.argsort(es); es, Us = es[o_] + shift, Us[:, o_]
    sec = [dict(E=float(es[q]), fid_phi=float((Us[:, q] @ phi) ** 2), S2=S2(Us[:, q])) for q in range(2)]
    g = Us[:, 0]; g = g * np.sign(g @ phi) if g @ phi != 0 else g
    return g, lows, sec
def en(v): return float(v @ (H @ v) / (v @ v))
def fn_energy(v, tol=1e-7, maxiter=4000):
    s = np.where(v >= 0, 1.0, -1.0); g = np.maximum(np.abs(v), 1e-300)
    viol = HO_v * s[HO_r] * s[HO_c] > 0
    keep = sp.csr_matrix((HO_v[~viol], (HO_r[~viol], HO_c[~viol])), shape=(D, D))
    rv, cv = HO_r[viol], HO_c[viol]
    dF = H_dg + np.bincount(rv, weights=HO_v[viol] * s[rv] * s[cv] * np.minimum(g[cv] / g[rv], RCAP), minlength=D)
    del viol, rv, cv
    HF = (keep + sp.diags(dF)).tocsr(); del keep
    x0 = g * s; x0 = x0 / np.linalg.norm(x0); e = float(x0 @ (HF @ x0)); pre = 1.0 / (np.abs(dF - e) + 1.0)
    nmv = [0]
    def mv(X):
        nmv[0] += 1
        if nmv[0] % 500 == 0: log('  FN lobpcg matvecs', nmv[0])
        return HF @ X
    A = sla.LinearOperator((D, D), matvec=lambda x: mv(np.ravel(x)), matmat=mv, dtype=np.float64)
    M = sla.LinearOperator((D, D), matvec=lambda r: pre * np.ravel(r), matmat=lambda R: pre[:, None] * R, dtype=np.float64)
    w, U, rh = sla.lobpcg(A, x0[:, None], M=M, largest=False, tol=tol, maxiter=maxiter, retResidualNormsHistory=True)
    u = U[:, 0]; rr = float(np.linalg.norm(HF @ u - w[0] * u) / np.linalg.norm(u))
    return float(w[0]), u, dict(iters=len(rh), resid=rr)
HC = H.tocoo(); offm = HC.row != HC.col
HO_r, HO_c, HO_v = HC.row[offm].astype(np.int32), HC.col[offm].astype(np.int32), HC.data[offm]; H_dg = H.diagonal(); del HC, offm
def dressed(g):
    g = g / np.linalg.norm(g); w = HJ2 @ g; Hg, Hw = H @ g, H @ w
    Hs = np.array([[g @ Hg, g @ Hw], [w @ Hg, w @ Hw]]); Ss = np.array([[g @ g, g @ w], [w @ g, w @ w]])
    Hs = (Hs + Hs.T) / 2; ev, U = eigh(Hs, Ss); al, be = U[:, 0]
    a = float(be / al); psi = g + a * w; psi /= np.linalg.norm(psi)
    return psi, a
def full_eval(psi, tag):
    psi = psi / np.linalg.norm(psi); efn, _, inf = fn_energy(psi)
    r = dict(E_var=(en(psi) - E0) / N, E_FN=(efn - E0) / N, fid_psi0=float((psi @ psi0) ** 2), W_M=WM(psi), fn_info=inf)
    log(tag, r); return r

# ------------------------------------------------------------------ checks
chk = {}
Cm, Gm, _ = cov(psi0); pu, cu = parent(Cm, Gm, 'u')
chk['psi0_u'] = dict(pu, max_dev_from_H=float(np.abs(cu - cH).max()),
                     J2_over_J1=float(cu[[k for k in range(K) if ops[k]['kind'] == 'XY' and (1, 1) in ops[k]['cls']['disps']][0]]),
                     C_eigs_rel=(np.linalg.eigvalsh(Cm)[:5] / np.linalg.eigvalsh(Cm)[-1]).tolist())
pert = psi0 + 1e-2 * rng.standard_normal(D) / np.sqrt(D); pert /= np.linalg.norm(pert)
Cm, Gm, _ = cov(pert); pp_, cp_ = parent(Cm, Gm, 'u')
chk['perturbed_u'] = dict(pp_, max_dev_from_H=float(np.abs(cp_ - cH).max()), fid_with_psi0=float((pert @ psi0) ** 2))
chk['psi0_recovery_pass'] = bool(chk['psi0_u']['max_dev_from_H'] < 1e-6 and chk['psi0_u']['rel_var'] < 1e-12)
chk['perturbed_fails_pass'] = bool(chk['perturbed_u']['max_dev_from_H'] > 1e-4 and chk['perturbed_u']['rel_var'] > 1e-8)
log('CHECK psi0 parent', chk['psi0_u']['max_dev_from_H'], chk['psi0_u']['rel_var'], 'perturbed', chk['perturbed_u']['max_dev_from_H'],
    chk['perturbed_u']['rel_var'])
res['checks'] = chk; json.dump(res, open(OUT, 'w'), indent=1)
efx, _, _ = fn_energy(psi0); chk['FN_exact_sign_minus_E0_site'] = (efx - E0) / N
log('CHECK FN exact sign', chk['FN_exact_sign_minus_E0_site'])

# ------------------------------------------------------------------ baselines
efm, um, inf = fn_energy(np.abs(psi0) * sN); um = um * (np.sign(um @ sN) or 1.0)
res['baseline'] = dict(marshall_FN=dict(E_FN=(efm - E0) / N, E_var=(en(um) - E0) / N, fid_psi0=float((um @ psi0) ** 2 / (um @ um)),
                                        W_M=WM(um), fn_info=inf))
log('baseline Marshall-FN', res['baseline']['marshall_FN'])
psd, ad = dressed(um); res['baseline']['dressed_marshall_FN'] = dict(full_eval(psd, 'baseline dressed Marshall-FN'), a=ad)
json.dump(res, open(OUT, 'w'), indent=1)

# ------------------------------------------------------------------ Phi and parents
lmin = float(sla.eigsh(HJ2, k=1, which='SA', tol=1e-10)[0][0]); lmax = float(sla.eigsh(HJ2, k=1, which='LA', tol=1e-10)[0][0])
res['HJ2_spectrum'] = [lmin, lmax]; res['Phi'] = {}
for cond in (5, 10, 30, 100, 1000, 10000):
    a = -(cond - 1) / (abs(lmin) + cond * lmax)                  # 1 + a lambda in [1 - |a| lmax, 1 + |a| |lmin|]
    A = (sp.identity(D, format='csr') + a * HJ2).tocsr()
    phi, info = sla.minres(A, psi0, tol=1e-12, maxiter=20000); rr = float(np.linalg.norm(A @ phi - psi0)); del A
    phi /= np.linalg.norm(phi)
    R = dict(a=a, cond=cond, minres_resid=rr, minres_info=int(info), W_M_Phi=WM(phi), S2_Phi=S2(phi),
             fid_Phi_psi0=float((phi @ psi0) ** 2))
    log('Phi cond', cond, R)
    for nmK, KK in (('H_J2', HJ2), ('H', H)):
        u = KK @ phi; R['relvar_' + nmK] = float((u @ u - (phi @ u) ** 2) / (u @ u))
    Cm, Gm, _ = cov(phi); R['C_eigs_rel'] = (np.linalg.eigvalsh(Cm)[:5] / np.linalg.eigvalsh(Cm)[-1]).tolist()
    for arm in ('u', 's'):
        pr, c = parent(Cm, Gm, arm); R[arm] = pr
        log('parent cond', cond, arm, 'rel_var', pr['rel_var'], {k: round(v, 5) for k, v in pr['coef'].items()})
        if cond >= 100: continue
        g, lows, sec = gs_parent(c, phi); pr['lowest2_Sz0'] = lows; pr['lowest2_sector'] = sec
        pr['GS_fid_Phi'] = sec[0]['fid_phi']; pr['sector_gap'] = sec[1]['E'] - sec[0]['E']
        pr['GS_W_M'] = WM(g); pr['GS_E_var'] = (en(g) - E0) / N; pr['GS_fid_psi0'] = float((g @ psi0) ** 2)
        log('GS(H_p) cond', cond, arm, 'Sz0', lows, 'sector', sec, 'W_M', pr['GS_W_M'])
        psd, ad = dressed(g); pr['dressed'] = dict(full_eval(psd, f'dressed cond {cond} arm {arm}'), a=ad)
        json.dump(res, open(OUT, 'w'), indent=1)
    res['Phi'][str(cond)] = R; json.dump(res, open(OUT, 'w'), indent=1)

# ------------------------------------------------------------------ sanity check and verdict
c100 = res['Phi']['100']['s']['coef']; j2xy = [n for n in names if n.startswith('XY') and n.endswith('(1,1)')][0]
chk['cond100_s_J2XY'] = c100[j2xy]
chk['cond100_s_dominated_by_negative_J2XY'] = bool(c100[j2xy] < 0 and abs(c100[j2xy]) >= max(abs(v) for v in c100.values()) - 1e-12)
best_s = min(res['Phi'][str(c)]['s']['dressed']['E_var'] for c in (5, 10, 30))
base = res['baseline']['dressed_marshall_FN']['E_var']
verd = 'PASS' if best_s <= 1e-3 else 'KILL' if best_s >= base else 'INCONCLUSIVE'
res['verdict'] = dict(best_stoquastic_dressed_E_var=best_s, baseline_dressed_marshallFN_E_var=base, verdict=verd,
                      rule='pass: best (s) dressed E_var-E0 <= 1e-3/site; kill: >= dressed Marshall-FN E_var; decisive at 4x6')
chk['diag_s_J2XY_by_cond'] = {c: res['Phi'][c]['s']['coef'][j2xy] for c in res['Phi']}
res['sec'] = time.time() - T0; json.dump(res, open(OUT, 'w'), indent=1)
print(f'VERDICT c6_t0 {LX}x{LY}: {verd} best_s_dressed_Evar={best_s:.3e} baseline={base:.3e} '
      f'checks psi0_recovery={chk["psi0_recovery_pass"]} perturbed_fails={chk["perturbed_fails_pass"]} '
      f'cond100_negJ2={chk["cond100_s_dominated_by_negative_J2XY"]} FN_exact={chk["FN_exact_sign_minus_E0_site"]:.1e}', flush=True)
