"""rangelimit_tangent_full.py: RUNG 2b exact full-table tangent shares (STORAGE_PROBLEM.md 'RUNG 2b RETRY RESULT'). No sampling, no split.
Head, images, target as rangelimit_tangent_retry.py. usage: python3 rangelimit_tangent_full.py RL_DIR OUTDIR M SCALE SEED BLOCK NBLOCK
One process computes the column block BLOCK (of NBLOCK equal blocks of the P parameters) of every P x P matrix; rangelimit_tangent_full_combine.py
assembles. All sums run over all 15,804,956 orbits r, J_r = J on the representative x_r (as in all rung-2b scripts).
L2:  W = diag(n p^2):  G_L2 = J^T W J,  H_L2 = J^T W^2 J (for the random-vector expectation),  b = J^T W t,  yy = t^T W t.
Q:   A_xy = sum_{kept bonds k of x_r with flip_k(x_r) in orbit y} n_x h_k p_x p_y (kept: s2_x s2_y < 0, as BS.build), symmetric
     (= sum over configuration pairs); d_x = sum_y A_xy; Laplacian Lq = diag(d) - A, so t^T Lq t = sum over each kept configuration-pair edge ONCE
     of h p p (t - t')^2 (= half of the 'exact Q_uu' 4.7824e-3 printed by rangelimit_tangent.py, which counted both ends).
     G_Q = J^T diag(d) J - J^T A J,  b = J^T (Lq t),  yy = t^T Lq t.  (A J_block) is gathered from J_block stored in RAM (float32).
Targets t: u = clip(r* - m, -4.88, 4.88); xi iid normal per orbit (seed 777); r_theta at another init (seed + 1000, same scale)."""
import os, sys, json, time
import numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
RL, OUT = sys.argv[1], sys.argv[2]; M, SC, SD, BLK, NBLK = int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
LIMIT = int(os.environ.get('RTF_LIMIT', '0'))          # >0: only the first LIMIT rows (smoke)
os.makedirs(OUT, exist_ok=True); T0 = time.time(); TAG = f'M{M}_s{SC}_seed{SD}_b{BLK}of{NBLK}'
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
N = 36; P = M + N * M; cols = np.array_split(np.arange(P), NBLK)[BLK]; c0, c1 = cols[0], cols[-1] + 1; nc = c1 - c0
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s = np.load(BS.S2).astype(np.float64); D = len(n); DR = LIMIT or D
reps = np.load('/project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz')['reps']; assert len(reps) == D
p = np.load(f'{RL}/p.npy'); rs = np.load(f'{RL}/rstar.npy'); m = json.load(open(f'{RL}/rangelimit_A.json'))['m']
u_all = np.clip(rs - m, -4.88, 4.88); w_all = n * p * p; assert abs(w_all.sum() - 1) < 1e-9
xi_all = np.random.default_rng(777).standard_normal(D)
# images (as the retry script; checked equal to c2l_rbmpp_lib)
Lsz = 6
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
G8 = np.array([pm(lambda x, y, tx=tx, ty=ty: (x + tx, y + ty))[q] for ty in range(Lsz) for tx in range(Lsz) for q in pts])
P_ = np.concatenate([G8[8 * t + k][None] for t in range(N) for k in (0, 2)])
IMG = np.concatenate([P_, P_]); FL = np.concatenate([np.ones(len(P_)), -np.ones(len(P_))]); NI = 144
import c2l_rbmpp_lib as RLB
lat = RLB.Lattice(6, proj='simple'); assert np.array_equal(np.asarray(lat.IMG), IMG) and np.array_equal(np.asarray(lat.FL), FL)
IMGj = jnp.asarray(IMG); FLj = jnp.asarray(FL)
@jax.jit
def _head(X, b, W):
    Z = FLj[None, :, None] * X[:, IMGj]; th = b[None, None, :] + jnp.einsum('bgi,ik->bgk', Z, W)
    T = jnp.tanh(th); JW = jnp.einsum('bgi,bgk->bik', Z, T) / NI
    return jnp.concatenate([T.mean(1), JW.reshape(X.shape[0], -1)], 1)
@jax.jit
def _ronly(X, b, W):
    th = b[None, None, :] + jnp.einsum('bgi,ik->bgk', FLj[None, :, None] * X[:, IMGj], W); a = jnp.abs(th)
    return (a + jnp.log1p(jnp.exp(-2 * a)) - jnp.log(2.0)).sum(2).mean(1)
def spins(i0, i1):
    x = reps[i0:i1].astype(np.uint64); return jnp.asarray(np.where(((x[:, None] >> np.arange(N, dtype=np.uint64)[None, :]) & np.uint64(1)) == 1, 1.0, -1.0))
r0 = np.random.default_rng(SD); b = jnp.asarray(r0.uniform(-SC, SC, M)); W = jnp.asarray(r0.uniform(-SC, SC, (N, M)))
r1 = np.random.default_rng(SD + 1000); b1 = jnp.asarray(r1.uniform(-SC, SC, M)); W1 = jnp.asarray(r1.uniform(-SC, SC, (N, M)))
R = 4096
# pass 1: store J[:, block] (float32) for all orbits and r_theta' for all orbits
Jb = np.empty((D, nc), np.float32); rp_all = np.zeros(D)
for i0 in range(0, DR, R):
    i1 = min(DR, i0 + R); X = spins(i0, i1); Jb[i0:i1] = np.asarray(_head(X, b, W)[:, c0:c1]); rp_all[i0:i1] = np.asarray(_ronly(X, b1, W1))
    if (i0 // R) % 500 == 0: log(f'pass1 {i1}/{DR}')
log('pass1 done')
TG = dict(u=u_all, xi=xi_all, rtheta_other=rp_all)
GL = np.zeros((P, nc)); HL = np.zeros((P, nc)); GD = np.zeros((P, nc)); GA = np.zeros((P, nc))
bL = {k: np.zeros(P) for k in TG}; bQ = {k: np.zeros(P) for k in TG}; yL = {k: 0.0 for k in TG}; yQ = {k: 0.0 for k in TG}
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r'); SW = 0.0; SDg = 0.0
for i0 in range(0, DR, R):
    i1 = min(DR, i0 + R); X = spins(i0, i1); J = np.asarray(_head(X, b, W))
    t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); va = t >= 0; tq = np.where(va, t, 0)
    kp = va & (s[i0:i1, None] * s[tq] < 0); rr, kk = np.nonzero(kp); x = rr + i0; y = tq[rr, kk]
    if LIMIT: keep = y < DR; rr, kk, x, y = rr[keep], kk[keep], x[keep], y[keep]
    a = n[x] * BS.HW[kk] * p[x] * p[y]
    Ach = sp.csr_matrix((a, (rr, np.arange(len(rr)))), shape=(i1 - i0, len(rr)))
    d = np.asarray(Ach.sum(1)).ravel()
    AJb = Ach @ Jb[y].astype(np.float64)
    w = w_all[i0:i1]; SW += float(w.sum()); SDg += float(d.sum())
    GL += (J * w[:, None]).T @ J[:, c0:c1]; HL += (J * (w * w)[:, None]).T @ J[:, c0:c1]
    GD += (J * d[:, None]).T @ J[:, c0:c1]; GA += J.T @ AJb
    for k_, tv in TG.items():
        tx = tv[i0:i1]; Lt = d * tx - Ach @ tv[y]
        bL[k_] += J.T @ (w * tx); yL[k_] += float((w * tx * tx).sum()); bQ[k_] += J.T @ Lt; yQ[k_] += float((tx * Lt).sum())
    if (i0 // R) % 500 == 0: log(f'pass2 {i1}/{DR}')
np.savez(f'{OUT}/blk_{TAG}.npz', cols=cols, GL=GL, HL=HL, GD=GD, GA=GA, **{f'bL_{k}': v for k, v in bL.items()}, **{f'bQ_{k}': v for k, v in bQ.items()})
json.dump(dict(M=M, scale=SC, seed=SD, block=BLK, nblock=NBLK, P=P, cols=[int(c0), int(c1)], rows=DR, sum_w=SW, sum_d=SDg, yL=yL, yQ=yQ, sec=time.time() - T0, host=os.uname().nodename),
          open(f'{OUT}/blk_{TAG}.json', 'w'), indent=1)
log(f'yL {yL} yQ {yQ}'); log('BLOCK DONE')
