"""C6-T0b (PLAN_ansatz.md, "Amendments after the adversarial review"): optimise the stoquastic parent couplings
directly for the dressed state's E_var. Exact 4x4 only.
  python3 c6_t0b.py 4 4 OUT_JSON C6T0_JSON
Parameters: every XY_d and ZZ_d class coefficient (RING excluded: mixed Marshall-frame signs). XY_d carries its
stoquastic sign, c_k = sigma_k |x_k|; ZZ_d is free, c_k = x_k; then c <- c / |c| (no coefficient is fixed, so -H_J2 is
reachable). Objective: E_var of (1 + a' H_J2) GS(H_p), GS in psi0's symmetric sector (translations, reflections,
spin flip), a' re-optimised in closed form at each evaluation. Powell then Nelder-Mead from 5 starts: the C6-T0
stoquastic parents at cond 5/10/30, -H_J2 + H_J1, H with the J2 XY term set to 0.
Setup (basis, operators, symmetry projector, FN, dressing) is copied verbatim from c6_t0.py.
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

# ------------------------------------------------------------------ C6-T0b
from scipy.optimize import minimize as _min
T0J = json.load(open(sys.argv[4]))
iop = [k for k in range(K) if ops[k]['kind'] in ('XY', 'ZZ')]
sig = np.array([1.0 if ops[k]['bound'] == 'ge0' else -1.0 if ops[k]['bound'] == 'le0' else 0.0 for k in iop])
assert all(ops[k]['bound'] != 'mixed' for k in iop)
def c_of(x):
    c = np.zeros(K); v = np.where(sig != 0, sig * np.abs(x), x); c[iop] = v / np.linalg.norm(v); return c
def x_of(c):
    v = c[iop]; assert np.all(v * sig >= -1e-12), 'start not stoquastic'
    return np.where(sig != 0, np.abs(v), v)
Mops = [(o['M'] if 'M' in o else sp.diags(o['dg'])).tocsr() for o in ops]
state = dict(v0=None, n=0, best=(np.inf, None))
def sector_gs(c, k=1, v0=None):
    Hp = sum(c[q] * Mops[q] for q in range(K) if c[q] != 0).tocsr()
    shift = float(abs(Hp).sum(axis=1).max()) + 1.0
    op = sla.LinearOperator((D, D), matvec=lambda v: (lambda pv: Hp @ pv - shift * pv)(Pproj(np.ravel(v))), dtype=np.float64)
    st = Pproj(v0 if v0 is not None else psi0)
    es, U = sla.eigsh(op, k=k, which='SA', tol=1e-11, v0=st); o_ = np.argsort(es)
    return es[o_] + shift, U[:, o_], Hp
def objective(x):
    c = c_of(x)
    try:
        es, U, _ = sector_gs(c, v0=state['v0'])
    except Exception as e:
        log('eigsh failed', e); return 1.0
    g = U[:, 0]; state['v0'] = g; psi, a = dressed(g); ev = (en(psi) - E0) / N
    state['n'] += 1
    if ev < state['best'][0]: state['best'] = (ev, x.copy())
    if state['n'] % 200 == 0: log('  evals', state['n'], 'best', state['best'][0])
    return ev
# checks: the pipeline gives 0 for the exact H (non-stoquastic, so outside the search) and > 0 when J2 XY is flipped
chk = {}
cHn = cH / np.linalg.norm(cH[iop]); es, U, _ = sector_gs(cHn); psi_, a_ = dressed(U[:, 0])
chk['exact_H_dressed_Evar'] = (en(psi_) - E0) / N; chk['exact_H_a'] = a_
cb = cH.copy(); j2 = [k for k in iop if ops[k]['kind'] == 'XY' and (1, 1) in ops[k]['cls']['disps']][0]; cb[j2] *= -1
es, U, _ = sector_gs(cb / np.linalg.norm(cb)); chk['J2XY_flipped_dressed_Evar'] = (en(dressed(U[:, 0])[0]) - E0) / N
log('CHECK exact H ->', chk['exact_H_dressed_Evar'], 'J2 XY flipped ->', chk['J2XY_flipped_dressed_Evar'])
efm, um, _ = fn_energy(np.abs(psi0) * sN); um = um * (np.sign(um @ sN) or 1.0)
psd, ad = dressed(um); base = dict(E_var=(en(psd) - E0) / N, E_FN=(fn_energy(psd)[0] - E0) / N, a=ad, marshall_FN=(efm - E0) / N)
log('baseline dressed Marshall-FN', base)
# Phi(cond 5)
lmin = float(sla.eigsh(HJ2, k=1, which='SA', tol=1e-12)[0][0]); lmax = float(sla.eigsh(HJ2, k=1, which='LA', tol=1e-12)[0][0])
aP = -(5 - 1) / (abs(lmin) + 5 * lmax); phi5 = sla.minres((sp.identity(D, format='csr') + aP * HJ2).tocsr(), psi0, tol=1e-13, maxiter=20000)[0]
phi5 /= np.linalg.norm(phi5)
# starts
def named(d):
    c = np.zeros(K)
    for k in iop:
        nm = names[k]
        if nm in d: c[k] = d[nm]
    return c
starts = {}
for cnd in ('5', '10', '30'):
    starts[f'C6T0_s_cond{cnd}'] = named(T0J['Phi'][cnd]['s']['coef'])
cJ1 = np.zeros(K); cJ2 = np.zeros(K)
for k in iop:
    if (1, 0) in ops[k]['cls']['disps']: cJ1[k] = 1.0
    if (1, 1) in ops[k]['cls']['disps']: cJ2[k] = J2
starts['-H_J2+H_J1'] = cJ1 - cJ2
c_ = cH.copy(); c_[j2] = 0.0; starts['H_J2XY=0'] = c_
out = dict(LX=LX, LY=LY, E0_site=E0 / N, checks=chk, baseline=base, names=[names[k] for k in iop],
           signs=sig.tolist(), starts={}, phi5_a=aP)
for nm, c0 in starts.items():
    x0 = x_of(c0); state.update(v0=None, n=0, best=(np.inf, None))
    f0 = objective(x0)
    r1 = _min(objective, x0, method='Powell', options=dict(maxfev=int(__import__('os').environ.get('C6B_MAXFEV', 4000)), xtol=1e-6, ftol=1e-12))
    r2 = _min(objective, state['best'][1], method='Nelder-Mead',
              options=dict(maxfev=int(__import__('os').environ.get('C6B_MAXFEV', 3000)), xatol=1e-7, fatol=1e-12, adaptive=True))
    fb, xb = state['best']; c = c_of(xb)
    es, U, Hp = sector_gs(c, k=2, v0=None); g = U[:, 0]; g = g * (np.sign(g @ phi5) or 1.0)
    eg, Ug = sla.eigsh(Hp, k=2, which='SA', tol=1e-11)             # global Sz=0 check
    psi, a = dressed(g)
    R = dict(start_E_var=f0, best_E_var=(en(psi) - E0) / N, evals=state['n'], a=a,
             coef={names[k]: float(c[k]) for k in iop}, GS_fid_Phi5=float((g @ phi5) ** 2),
             GS_fid_psi0=float((g @ psi0) ** 2), GS_W_M=WM(g), sector_gap=float(es[1] - es[0]),
             GS_is_global_Sz0_GS=bool(abs(eg.min() - es[0]) < 1e-8), global_Sz0_lowest=sorted(eg.tolist()),
             S2_GS=S2(g), dressed=full_eval(psi, f'C6-T0b {nm}'))
    out['starts'][nm] = R
    log('START', nm, 'E_var', f0, '->', R['best_E_var'], 'fidPhi5', R['GS_fid_Phi5'],
        {k: round(v, 4) for k, v in R['coef'].items()})
    json.dump(out, open(OUT, 'w'), indent=1)
bn = min(out['starts'], key=lambda s: out['starts'][s]['best_E_var']); b = out['starts'][bn]['best_E_var']
verd = 'PASS' if b <= 0.5 * base['E_var'] else 'KILL' if b >= base['E_var'] else 'INCONCLUSIVE'
out['verdict'] = dict(best_start=bn, best_dressed_E_var=b, baseline=base['E_var'], verdict=verd,
                      rule='pass <= 0.5 x dressed Marshall-FN E_var; kill >= it')
out['sec'] = time.time() - T0; json.dump(out, open(OUT, 'w'), indent=1)
print(f'VERDICT c6_t0b {LX}x{LY}: {verd} best_dressed_Evar={b:.4e} (start {bn}) baseline={base["E_var"]:.4e} '
      f'E_FN={out["starts"][bn]["dressed"]["E_FN"]:.4e} fidPhi5={out["starts"][bn]["GS_fid_Phi5"]:.4f} '
      f'check_exactH={chk["exact_H_dressed_Evar"]:.1e} check_flip={chk["J2XY_flipped_dressed_Evar"]:.2e}', flush=True)
