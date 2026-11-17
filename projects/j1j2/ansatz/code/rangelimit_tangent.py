"""rangelimit_tangent.py: RUNG 2b (STORAGE_PROBLEM.md): tangent share of the clipped target u = clip(r* - m, -4.88, 4.88) for the
symmetric real RBM head r_theta(x) = mean_g sum_k ln cosh(b_k + W_k . z_g), z_g = FL_g x[IMG_g], the 144 'simple' images of
c2l_rbmpp_lib.Lattice(6, proj='simple'). No training. CPU, numpy. usage: python3 rangelimit_tangent.py RL_DIR OUTDIR [smoke]
Inputs from rung 2 (RL_DIR = rangelimit_6x6): p.npy (amplitude per orbit, sum n p^2 = 1), rstar.npy, rangelimit_A.json (m).
Sample: 1e5 orbits, stratified by decade d = clip(floor(log10(p/p_max)), -12, 0); per decade k_d = max(1000, proportional to its n p^2 weight)
(capped by the decade's orbit count), drawn with replacement prop. to n p^2 inside the decade; weight omega = W_d / k_d (sum omega = 1).
r_theta is evaluated on the orbit representative x_r (reps of psi0_6x6_table.npz; the head is not orbit-invariant under reflections).
(i)  L2:  |v|^2 = sum_S omega v^2 (estimates sum n p^2 v^2).
(ii) Q:   <u,v>_Q = sum_x sum_{kept bonds b of x_r} n_x h_b p_x p_y (u_x - u_y)(v_x - v_y), kept = s2_x s2_y < 0 (as BS.build), y = orbit of
     flip_b(x_r). Estimated by the rows x in S with weight omega / (n_x p_x^2); every kept neighbour y of a sampled x enters with its exact
     u_y and its exact J(y) (computed on y's representative), so the only restriction is on the row sum, not on the neighbours.
     Exact Q(u,u) and |u|^2 over all rows are computed too, as a check of the sample estimates.
Share = 1 - |u - J c|^2 / |u|^2, c = (J^T Om J + lam tr/P I)^-1 J^T Om u (ridge scan); at lam -> 0 equals |proj_J u|^2/|u|^2.
Break tests per (M, seed): u -> xi (iid normal on all orbits, rescaled to |u| in each norm), u -> r_theta' (another init, seed + 1000)."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
RL, OUT = sys.argv[1], sys.argv[2]; SMOKE = len(sys.argv) > 3; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
L_CLIP = 4.88; NS_ = 2000 if SMOKE else 100000; FLOOR = 50 if SMOKE else 1000; MS = [8] if SMOKE else [8, 16, 32]; SEEDS = [0] if SMOKE else [0, 1, 2]
LAMS = [0.0] + [10.0 ** k for k in range(-14, -1, 2)]
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s = np.load(BS.S2).astype(np.float64); D = len(n)
reps = np.load('/project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz')['reps']; assert len(reps) == D
p = np.load(f'{RL}/p.npy'); rs = np.load(f'{RL}/rstar.npy'); m = json.load(open(f'{RL}/rangelimit_A.json'))['m']
u_all = np.clip(rs - m, -L_CLIP, L_CLIP); w_all = n * p * p; assert abs(w_all.sum() - 1) < 1e-9
res = dict(host=os.uname().nodename, RL=RL, L=L_CLIP, m=m, Nsample=NS_, lams=LAMS, smoke=SMOKE)
RJ = f'{OUT}/tangent.json'
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
# --- images (as c2l_rbmpp_lib.Lattice(6, proj='simple'): point index 0 and 2 of each translation, x {+1, -1})
Lsz = 6; N = 36
def site(x, y): return (x % Lsz) + Lsz * (y % Lsz)
def pm(f):
    q = np.empty(N, np.int64)
    for y in range(Lsz):
        for x in range(Lsz): xx, yy = f(x, y); q[site(x, y)] = site(xx, yy)
    return q
c4 = pm(lambda x, y: (-y, x)); sx = pm(lambda x, y: (-x, y)); r_ = np.arange(N); pts = []
for k in range(4):
    for refl in (False, True):
        q = r_.copy()
        for _ in range(k): q = c4[q]
        if refl: q = sx[q]
        pts.append(q)
G = np.array([pm(lambda x, y, tx=tx, ty=ty: (x + tx, y + ty))[q] for ty in range(Lsz) for tx in range(Lsz) for q in pts])
P_ = np.concatenate([G[8 * t + k][None] for t in range(N) for k in (0, 2)])
IMG = np.concatenate([P_, P_]); FL = np.concatenate([np.ones(len(P_)), -np.ones(len(P_))]); NI = len(IMG); assert NI == 144
try:
    import c2l_rbmpp_lib as RL_
    lat = RL_.Lattice(6, proj='simple'); assert np.array_equal(np.asarray(lat.IMG), IMG) and np.array_equal(np.asarray(lat.FL), FL); res['img_check'] = 'equal to c2l_rbmpp_lib'
except Exception as ex: res['img_check'] = f'not compared: {ex!r}'
log(f"images {NI}: {res['img_check']}")
def spins(idx):
    x = reps[idx].astype(np.uint64); return np.where(((x[:, None] >> np.arange(N, dtype=np.uint64)[None, :]) & np.uint64(1)) == 1, 1.0, -1.0)
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
IMGj = jnp.asarray(IMG); FLj = jnp.asarray(FL)
@jax.jit
def _head(X, b, W):
    Z = FLj[None, :, None] * X[:, IMGj]; th = b[None, None, :] + jnp.einsum('bgi,ik->bgk', Z, W)
    a = jnp.abs(th); r = (a + jnp.log1p(jnp.exp(-2 * a)) - jnp.log(2.0)).sum(2).mean(1)
    T = jnp.tanh(th); JW = jnp.einsum('bgi,bgk->bik', Z, T) / NI
    return r, jnp.concatenate([T.mean(1), JW.reshape(X.shape[0], -1)], 1)
def head(X, b, W, jac=True):
    """X (B, 36) -> r (B,), J (B, M + 36 M) [d/db, d/dW (row-major i, k)] (jax, x64)."""
    r, J = _head(jnp.asarray(X), jnp.asarray(b), jnp.asarray(W)); return np.asarray(r), (np.asarray(J) if jac else None)
def head_np(X, b, W):   # numpy reference, used once as a check
    Z = FL[None, :, None] * X[:, IMG]; th = b[None, None, :] + Z @ W; a = np.abs(th); r = (a + np.log1p(np.exp(-2 * a)) - np.log(2.0)).sum(2).mean(1)
    T = np.tanh(th); return r, np.concatenate([T.mean(1), (np.matmul(Z.transpose(0, 2, 1), T) / NI).reshape(len(X), -1)], 1)
_X = np.where(np.random.default_rng(7).random((64, N)) < 0.5, 1.0, -1.0); _b = np.full(4, 0.3); _W = np.random.default_rng(8).normal(0, 0.3, (N, 4))
_r, _J = head(_X, _b, _W); _r2, _J2 = head_np(_X, _b, _W); _e = 1e-6; _b2 = _b.copy(); _b2[1] += _e
_fd = (head_np(_X, _b2, _W)[0] - _r2) / _e
res['head_check'] = dict(jax_vs_np_r=float(np.abs(_r - _r2).max()), jax_vs_np_J=float(np.abs(_J - _J2).max()), fd_db1=float(np.abs(_fd - _J2[:, 1]).max()))
log(f"head check {res['head_check']}")
# --- sample
rng = np.random.default_rng(12345)
dec = np.clip(np.floor(np.log10(p / p.max())), -12, 0).astype(int)
Wd = {d: float(w_all[dec == d].sum()) for d in range(-12, 1)}; cnt = {d: int((dec == d).sum()) for d in range(-12, 1)}
k = {d: (min(cnt[d], max(FLOOR, int(round(NS_ * Wd[d])))) if cnt[d] else 0) for d in Wd}
ex = sum(k.values()) - NS_; big = max(k, key=lambda d: k[d]); k[big] -= max(0, ex)
idx, om = [], []
for d in Wd:
    if k[d] == 0: continue
    I = np.nonzero(dec == d)[0]; pr = w_all[I] / w_all[I].sum(); sel = I[rng.choice(len(I), size=k[d], replace=True, p=pr)]
    idx.append(sel); om.append(np.full(k[d], Wd[d] / k[d]))
S = np.concatenate(idx); om = np.concatenate(om); np.savez(f'{OUT}/sample.npz', idx=S, omega=om, dec=dec[S])
res['strata'] = {str(d): dict(weight=Wd[d], orbits=cnt[d], k=k[d]) for d in Wd}; res['n_sample'] = len(S); res['n_unique'] = int(len(np.unique(S))); dump()
log(f"sample {len(S)} ({res['n_unique']} unique) strata { {d: k[d] for d in k if k[d]} }")
# --- kept edges of sampled rows
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
Ssort = np.argsort(S); tS = np.empty((len(S), 144), np.int64); tS[Ssort] = np.concatenate([t1[S[Ssort]], t2[S[Ssort]]], 1)
valid = tS >= 0; tt = np.where(valid, tS, 0); kept = valid & (s[S][:, None] * s[tt] < 0)
er, ek = np.nonzero(kept); ey = tt[er, ek]; ex_ = S[er]
ew = (om[er] / (n[ex_] * p[ex_] ** 2)) * n[ex_] * BS.HW[ek] * p[ex_] * p[ey]   # omega/(n p^2) * n_x h p_x p_y
res['n_edges'] = int(len(er)); log(f'kept edges {len(er)}')
log(f'distinct neighbour orbits: {len(np.unique(ey))}')
# --- exact |u|^2 and Q(u,u) on all rows (check of the estimators)
def exactQ(v):
    tot = 0.0
    for i0 in range(0, D, 262144):
        i1 = min(D, i0 + 262144); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); va = t >= 0; tq = np.where(va, t, 0)
        kp = va & (s[i0:i1, None] * s[tq] < 0); rr, kk = np.nonzero(kp); x = rr + i0; y = tq[rr, kk]
        tot += float((n[x] * BS.HW[kk] * p[x] * p[y] * (v[x] - v[y]) ** 2).sum())
        if SMOKE and i0 > 2e6: return float('nan')
    return tot
res['exact'] = dict(L2_uu=float((w_all * u_all ** 2).sum()), Q_uu=exactQ(u_all))
res['estimate'] = dict(L2_uu=float((om * u_all[S] ** 2).sum()), Q_uu=float((ew * (u_all[ex_] - u_all[ey]) ** 2).sum())); dump()
log(f"exact {res['exact']} estimate {res['estimate']}")
xi_all = rng.standard_normal(D)
def solve_share(Gm, gs, yys):
    tr = np.trace(Gm) / Gm.shape[0]; ev = np.linalg.eigvalsh(Gm); out = {}
    for nm in gs:
        sh = []
        for lam in LAMS:
            try: c = np.linalg.solve(Gm + lam * tr * np.eye(len(gs[nm])), gs[nm]) if lam > 0 else np.linalg.lstsq(Gm, gs[nm], rcond=None)[0]
            except np.linalg.LinAlgError: sh.append(float('nan')); continue
            sh.append(float(1 - (yys[nm] - 2 * c @ gs[nm] + c @ Gm @ c) / yys[nm]))
        out[nm] = dict(share=sh, yy=yys[nm])
    out['eig'] = dict(max=float(ev[-1]), n_gt_1e_8max=int((ev > 1e-8 * ev[-1]).sum()), n_gt_1e_12max=int((ev > 1e-12 * ev[-1]).sum()))
    return out
res['runs'] = []; CH = 8192
for M in MS:
    for sd in SEEDS:
        r0 = np.random.default_rng(sd); b = r0.uniform(-0.01, 0.01, M); W = r0.uniform(-0.01, 0.01, (N, M))
        r1 = np.random.default_rng(sd + 1000); b1 = r1.uniform(-0.01, 0.01, M); W1 = r1.uniform(-0.01, 0.01, (N, M))
        P = M + N * M; t = time.time(); JS = np.empty((len(S), P)); r1S = np.empty(len(S))
        for i0 in range(0, len(S), 4096):
            X = spins(S[i0:i0 + 4096]); _, JS[i0:i0 + 4096] = head(X, b, W); r1S[i0:i0 + 4096] = head(X, b1, W1, jac=False)[0]
        T_S = dict(u=u_all[S], xi=xi_all[S], rtheta_other=r1S)
        GL = (JS * om[:, None]).T @ JS; gL = {k_: (JS * om[:, None]).T @ v for k_, v in T_S.items()}; yL = {k_: float((om * v * v).sum()) for k_, v in T_S.items()}
        GQ = np.zeros((P, P)); gQ = {k_: np.zeros(P) for k_ in T_S}; yQ = {k_: 0.0 for k_ in T_S}
        for i0 in range(0, len(er), CH):
            e = slice(i0, i0 + CH); X = spins(ey[e]); r1y = head(X, b1, W1, jac=False)[0]; _, Jy = head(X, b, W)
            A = JS[er[e]] - Jy; ww = ew[e]; Aw = A * ww[:, None]; GQ += Aw.T @ A
            for k_, dy in (('u', u_all[ex_[e]] - u_all[ey[e]]), ('xi', xi_all[ex_[e]] - xi_all[ey[e]]), ('rtheta_other', r1S[er[e]] - r1y)):
                gQ[k_] += Aw.T @ dy; yQ[k_] += float((ww * dy * dy).sum())
        rec = dict(M=M, seed=sd, P=P, jac_sec=time.time() - t, L2=solve_share(GL, gL, yL), Q=solve_share(GQ, gQ, yQ))
        res['runs'].append(rec); dump()
        f = lambda q: max(x for x in q if x == x)
        log(f"M={M} seed={sd} P={P}: L2 u {f(rec['L2']['u']['share']):.4f} xi {f(rec['L2']['xi']['share']):.4f} r' {f(rec['L2']['rtheta_other']['share']):.4f} | Q u {f(rec['Q']['u']['share']):.4f} xi {f(rec['Q']['xi']['share']):.4f} r' {f(rec['Q']['rtheta_other']['share']):.4f} | rank(1e-8) L2 {rec['L2']['eig']['n_gt_1e_8max']} Q {rec['Q']['eig']['n_gt_1e_8max']} {rec['jac_sec']:.0f}s")
        del JS
res['status'] = 'DONE'; dump(); log('DONE')
