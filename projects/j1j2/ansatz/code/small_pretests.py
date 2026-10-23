"""C2-T0 and C4-T0 (PLAN_ansatz.md): exact small-lattice representability pre-tests, J1-J2 at J2/J1 = 0.5, PBC.
  python3 small_pretests.py LX LY OUT_JSON
C2: Phi_a = (1 + a K)^{-1} psi0, K = H_J2 or V_J2 (J2 hop part); Marshall-violating weight W_M(Phi_a).
C4: psi0 in a cluster frame (2x2 crossed plaquettes; 2x1 dimers), Sz-conserving local rotations optimised by gradient
ascent of F = sum psi'|psi'| = 1 - 2 W_neg; translation-invariant (one R) and free (one R per cluster).
Control: Ising frame, greedy per-site sign gauge (product sign rules) must not beat Marshall by > 10%.
"""
import sys, json, time, itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.linalg import expm, eigh
LX, LY, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
N = LX * LY; J2 = 0.5; T0 = time.time()
def site(x, y): return (x % LX) + LX * (y % LY)
NN = [(site(x, y), site(x + 1, y)) for y in range(LY) for x in range(LX)] + \
     [(site(x, y), site(x, y + 1)) for y in range(LY) for x in range(LX)]
NNN = [(site(x, y), site(x + 1, y + 1)) for y in range(LY) for x in range(LX)] + \
      [(site(x, y), site(x + 1, y - 1)) for y in range(LY) for x in range(LX)]
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)

# ------------------------------------------------------------------ Sz = 0 basis and Hamiltonians
allx = np.arange(1 << N, dtype=np.int64)
pc = np.zeros(1 << N, np.int8)
for i in range(N): pc += ((allx >> i) & 1).astype(np.int8)
basis = allx[pc == N // 2]; del allx, pc
D = basis.size; log('D', D)
def ham(bonds, J, diag=True, off=True):
    rows, cols, vals = [], [], []
    dg = np.zeros(D)
    for (i, j) in bonds:
        bi = (basis >> i) & 1; bj = (basis >> j) & 1; anti = bi != bj
        dg += J * np.where(anti, -0.25, 0.25)
        if off:
            r = np.nonzero(anti)[0]; y = basis[r] ^ ((1 << i) | (1 << j))
            rows.append(r); cols.append(np.searchsorted(basis, y)); vals.append(np.full(r.size, 0.5 * J))
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(D, D)) if off \
        else sp.csr_matrix((D, D))
    return M + sp.diags(dg) if diag else M
NNN3 = [(site(x, y), site(x + 2, y)) for y in range(LY) for x in range(LX)] + \
       [(site(x, y), site(x, y + 2)) for y in range(LY) for x in range(LX)]
NNN3 = list({tuple(sorted(b)) for b in NNN3})                         # (2,0) bonds double up on a width-4 torus
H1 = ham(NN, 1.0); HJ2 = ham(NNN, J2); VJ2 = ham(NNN, J2, diag=False)
VJ1 = ham(NN, 1.0, diag=False); VJ3 = ham(NNN3, J2, diag=False); VH = (VJ1 + VJ2).tocsr()
H = (H1 + HJ2).tocsr()
E, V = sla.eigsh(H, k=2, which='SA', tol=1e-12); o = np.argsort(E); E, V = E[o], V[:, o]
psi = V[:, 0]; E0 = E[0]
log('E0/N', E0 / N, 'gap', E[1] - E[0])
res = dict(LX=LX, LY=LY, D=int(D), E0_site=E0 / N, gap=float(E[1] - E[0]))
MA = sum(1 << site(x, y) for y in range(LY) for x in range(LX) if (x + y) % 2 == 0)
def par(v, m):
    c = np.zeros(v.size, np.int64); w = v & m
    for i in range(N): c += (w >> i) & 1
    return 1 - 2 * (c & 1)
sN = par(basis, MA)
if np.sum(psi ** 2 * np.sign(psi) * sN) < 0: psi = -psi
def WM(v, sgn=sN):
    v = v / np.linalg.norm(v); s = np.sign(np.sum(v * np.abs(v) * sgn)) or 1.0
    return float(np.sum(v ** 2 * ((np.sign(v) * s) != sgn) * (v != 0)))
def by_dec(v, sgn):
    v = v / np.linalg.norm(v); p = v ** 2; d = np.floor(np.log10(np.maximum(p, 1e-300))).astype(int)
    s = np.sign(np.sum(v * np.abs(v) * sgn)) or 1.0; bad = ((np.sign(v) * s) != sgn) & (v != 0)
    return {str(k): [float(p[d == k].sum()), float((p * bad)[d == k].sum() / max(p[d == k].sum(), 1e-300))]
            for k in range(d[p > 0].max(), max(d[p > 0].min(), -20) - 1, -1) if (d == k).any()}
res['W_M_psi0'] = WM(psi); res['psi0_by_decade'] = by_dec(psi, sN)
log('W_M(psi0)', res['W_M_psi0'])

# ------------------------------------------------------------------ control: greedy product sign gauge (Ising frame)
s_g = sN.copy(); best = WM(psi, s_g); improved = True; it = 0
while improved and it < 5 * N:
    improved = False
    for i in range(N):
        t = s_g * (1 - 2 * ((basis >> i) & 1))                # flip the gauge sign of site i's up state
        w = WM(psi, t)
        if w < best - 1e-15: best, s_g, improved = w, t, True
    it += 1
res['control_greedy_product_gauge_WM'] = best
log('control greedy product gauge W', best, 'Marshall', res['W_M_psi0'])

# ------------------------------------------------------------------ C2: Phi_a = (1 + a K)^{-1} psi0
res['C2'] = {}
for nm, K in (('H_J2', HJ2), ('V_J2', VJ2), ('ctl_V_J1', VJ1), ('ctl_V_J3', VJ3), ('V_H', VH)):
    lmin = sla.eigsh(K, k=1, which='SA', tol=1e-8)[0][0]; lmax = sla.eigsh(K, k=1, which='LA', tol=1e-8)[0][0]
    out = []
    for a in np.concatenate([-np.logspace(-3, 1, 25)[::-1], np.logspace(-3, 1, 25)]):
        ev = 1 + a * np.array([lmin, lmax])
        if ev.min() <= 0 and ev.max() >= 0: continue          # spectrum of 1 + aK straddles 0: singular inside
        cond = np.abs(ev).max() / np.abs(ev).min()
        if cond > 1e3: continue
        A = sp.identity(D, format='csr') + a * K
        phi, info = sla.minres(A, psi, tol=1e-12, maxiter=5000)
        rr = float(np.linalg.norm(A @ phi - psi))
        out.append(dict(a=float(a), cond=float(cond), W_M=WM(phi), resid=rr, info=int(info)))
    bestc = min(out, key=lambda r: r['W_M'])
    phi = sla.minres(sp.identity(D, format='csr') + bestc['a'] * K, psi, tol=1e-12, maxiter=5000)[0]
    res['C2'][nm] = dict(lmin=float(lmin), lmax=float(lmax), scan=out, best=bestc, best_by_decade=by_dec(phi, sN),
                         ratio=bestc['W_M'] / res['W_M_psi0'])
    log('C2', nm, 'best a', bestc['a'], 'W_M', bestc['W_M'], 'ratio', res['C2'][nm]['ratio'])
json.dump(res, open(OUT, 'w'), indent=1)

# ------------------------------------------------------------------ C4: cluster frames
full = np.zeros(1 << N); full[basis] = psi
def frame_test(cx, cy, steps=300):
    ncx, ncy = LX // cx, LY // cy; nC = ncx * ncy; m = cx * cy; dloc = 1 << m
    cl = [[site(i * cx + u, j * cy + v) for v in range(cy) for u in range(cx)] for j in range(ncy) for i in range(ncx)]
    # bit permutation: full index -> tensor index (cluster k occupies digits 4k.., axis k)
    idx = np.zeros(1 << N, np.int64); ax = np.arange(1 << N, dtype=np.int64)
    for k, sites in enumerate(cl):
        loc = np.zeros(1 << N, np.int64)
        for q, s_ in enumerate(sites): loc |= ((ax >> s_) & 1) << q
        idx |= loc << (m * (nC - 1 - k))
    T = np.zeros(1 << N); T[idx] = full; T = T.reshape((dloc,) * nC); del idx, ax
    # local Hamiltonian of a cluster (bonds inside it), Sz blocks
    sset = {s_: q for q, s_ in enumerate(cl[0])}
    h = np.zeros((dloc, dloc))
    for (bonds, J) in ((NN, 1.0), (NNN, J2)):
        for (i, j) in bonds:
            if i in sset and j in sset:
                qi, qj = sset[i], sset[j]
                for c in range(dloc):
                    bi, bj = (c >> qi) & 1, (c >> qj) & 1
                    h[c, c] += J * (0.25 if bi == bj else -0.25)
                    if bi != bj: h[c ^ (1 << qi) ^ (1 << qj), c] += 0.5 * J
    nup = np.array([bin(c).count('1') for c in range(dloc)])
    blockmask = nup[:, None] == nup[None, :]
    U0 = np.zeros((dloc, dloc)); evals = np.zeros(dloc)
    for n_ in range(m + 1):
        ii = np.nonzero(nup == n_)[0]; w_, v_ = eigh(h[np.ix_(ii, ii)]); U0[np.ix_(ii, ii)] = v_; evals[ii] = w_
    def apply(Rs):                                             # psi' = (R_k^T on every axis k) psi
        X = T
        for k in range(nC): X = np.moveaxis(np.tensordot(Rs[k].T, X, axes=([1], [k])), 0, k)
        return X
    def F(X): return float(np.sum(X * np.abs(X)))
    A_loc = [q for q, s_ in enumerate(cl[0]) if (s_ % LX + s_ // LX) % 2 == 0]
    MG = np.diag([(-1.0) ** sum((c >> q) & 1 for q in A_loc) for c in range(dloc)])
    def run(free, start='eig'):
        Rs = [(U0 if start == 'eig' else MG).copy() for _ in range(nC)]
        # best sign gauge of the eigenbasis columns (greedy, shared or per cluster)
        X = apply(Rs); f = F(X); imp = True
        while imp:
            imp = False
            for k in (range(nC) if free else [None]):
                for c in range(dloc):
                    Rn = [R.copy() for R in Rs]
                    for kk in (range(nC) if k is None else [k]): Rn[kk][:, c] *= -1
                    fn = F(apply(Rn))
                    if fn > f + 1e-14: Rs, f, imp = Rn, fn, True
        f_eig = f; eta = 0.05
        for st in range(steps):
            X = apply(Rs); G = []
            for k in range(nC):
                axs = [a for a in range(nC) if a != k]
                Mk = np.tensordot(2 * np.abs(X), X, axes=(axs, axs))     # (a, b)
                G.append(-(Mk - Mk.T) / 2 * blockmask)
            if not free: G = [sum(G) / nC] * nC
            gn = np.sqrt(sum(np.sum(g * g) for g in G))
            if gn < 1e-10: break
            best_f, best_R = f, None
            for e in (eta * 4, eta, eta / 4):
                Rn = [R @ expm(-e * g / gn) for R, g in zip(Rs, G)]   # sign chosen for ascent; checked by F below
                fn = F(apply(Rn))
                if fn > best_f: best_f, best_R, best_e = fn, Rn, e
                Rn = [R @ expm(e * g / gn) for R, g in zip(Rs, G)]
                fn = F(apply(Rn))
                if fn > best_f: best_f, best_R, best_e = fn, Rn, e
            if best_R is None:
                eta /= 4
                if eta < 1e-6: break
                continue
            Rs, f, eta = best_R, best_f, best_e
        X = apply(Rs); p = X.ravel() ** 2
        d = np.floor(np.log10(np.maximum(p, 1e-300))).astype(int); neg = (X.ravel() < 0)
        bd = {str(k): [float(p[d == k].sum()), float((p * neg)[d == k].sum() / max(p[d == k].sum(), 1e-300))]
              for k in range(d[p > 0].max(), max(d[p > 0].min(), -20) - 1, -1) if (d == k).any()}
        return dict(W_neg_eigbasis=(1 - f_eig) / 2, W_neg=(1 - f) / 2, steps=st + 1, by_decade=bd,
                    orth_err=float(max(np.abs(R.T @ R - np.eye(dloc)).max() for R in Rs)))
    out = dict(cluster=[cx, cy], n_clusters=nC, local_evals=evals.tolist(), check_norm=float(np.sum(T * T)))
    for nm_, fr_, st_ in (('shared', False, 'eig'), ('free', True, 'eig'), ('shared_from_marshall', False, 'marshall'),
                          ('free_from_marshall', True, 'marshall')):
        out[nm_] = run(fr_, st_); log('C4', cx, cy, nm_, out[nm_]['W_neg_eigbasis'], '->', out[nm_]['W_neg'])
    return out
res['C4'] = {}
for (cx, cy) in ((2, 2), (2, 1)):
    if LX % cx or LY % cy: continue
    res['C4'][f'{cx}x{cy}'] = frame_test(cx, cy)
    for k in ('shared', 'free', 'shared_from_marshall', 'free_from_marshall'): res['C4'][f'{cx}x{cy}'][k]['ratio'] = res['C4'][f'{cx}x{cy}'][k]['W_neg'] / res['W_M_psi0']
    json.dump(res, open(OUT, 'w'), indent=1)
res['sec'] = time.time() - T0; json.dump(res, open(OUT, 'w'), indent=1); log('DONE')
