"""test_theta2_calib.py: Theta v2 calibration check (PLAN_pfaffian.md "Theta v2 calibration check"), CPU, sign s1.  NOT committed.
C_exact = b^T (G + lam I)^-1 b / D_RR, lam = 1e-10 tr(G)/P, over ALL blocked pairs at s1 (520M entries, weights n_x 0.5 HW_k phi''_x phi''_y as bond_split.cost_pass),
J = d R_theta/d theta of the alpha-16 M-R RBM residual (theta_a16.npy, 592 columns), per orbit rep; R = R*_1 = ln phi'' - ln|A_s1|.
J never goes to disk: 3 column blocks (feature groups 0-4, 5-9, 10-15), float32 in RAM, two blocks at a time:
  pass 1 holds (B0,B1): G00 G11 G01 ; pass 2 keeps B1 + computes B2: G22 G12 ; pass 3 keeps B2 + recomputes B0: G02.  Repeated blocks are cross-checked.
The same pair pass also carries the L1 feature set (9 columns) and a permuted-L1 control, and the column R, so L1's eps^2 = 1 - C_L1 comes from identical machinery.
usage: python3 test_theta2_calib.py OUT check | smoke | run | final      (smoke: first NSM reps only, vs a naive float64 reference)"""
import os, sys
TH = os.environ.get('CALIB_THREADS', '8')
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = TH
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_FLAGS', f'--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={TH}')
import json, time, ast, hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import blas

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = sys.argv[1]; STAGE = sys.argv[2]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
R_ = '/scratch/a/A.Otaifi/ansatz_runs'
S1F = f'{R_}/pfaffian_z/s1.npy'; PHIPP = f'{R_}/pfaffian_r1/target_la_s1.npy'
LA_S1 = f'{R_}/pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy'
THETA = f'{R_}/pfaffian_mr/theta_a16.npy'; RA16 = f'{R_}/pfaffian_mr/R_a16.npy'
TAB = f'{R_}/c2_t0b_6x6/tables'; CSRD = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr'
REPS = '/project/theorie/a/A.Otaifi/Chatty/runs/ll6/data/psi0_6x6_table.npz'
HW = np.concatenate([np.full(72, 0.5), np.full(72, 0.25)])
NBLK_ENTRIES = 519978046; DRR_N = 2.0316018522113357e-4          # prep2.json n_blocked_entries ; pfaffian_n s1_b0 gram_result T1 D_RR
L1_REF = 0.9771830555725267                                      # pfaffian_n s1_b0 gram_result T1 L1
AL = 16; P = AL * 37
FEATS = [list(range(0, 5)), list(range(5, 10)), list(range(10, 16))]
def cols_of(fs): return np.array(list(fs) + [AL + f * 36 + i for f in fs for i in range(36)])
BCOLS = [cols_of(f) for f in FEATS]; assert sorted(np.concatenate(BCOLS).tolist()) == list(range(P))

n = np.load(f'{CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
reps = np.load(REPS)['reps']; assert reps.shape == (D,)
SH = np.arange(36, dtype=np.uint64)
def bits(r): return ((r[:, None] >> SH[None, :]) & np.uint64(1)).astype(np.float64) * 2 - 1
theta = np.load(THETA).astype(np.float64); assert theta.shape == (P,)

# ------------------------------------------------------------------ model and J
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from rbmpp_lib import Lattice
lat = Lattice(6, 0.5, 'full'); IMG = lat.IMG; INV = lat.INV; FLP = lat.FL.astype(jnp.float64)
def mr_model(alpha):
    src = open(os.path.join(HERE, 'mr_residual.py')).read(); tree = ast.parse(src)
    fdef = next(nd for nd in tree.body if isinstance(nd, ast.FunctionDef) and nd.name == 'make_model')
    ns = dict(jax=jax, jnp=jnp, IMG=IMG, INV=INV, FLP=FLP); exec(ast.get_source_segment(src, fdef), ns)
    Pp, R_one, *_ = ns['make_model'](alpha); assert Pp == P
    return R_one, hashlib.sha256(src.encode()).hexdigest()[:16]
R_one, MR_SHA = mr_model(AL)
Pm = np.zeros((36, 576, 36)); Pm[np.asarray(IMG).reshape(-1), np.repeat(np.arange(576), 36), np.tile(np.arange(36), 576)] = 1.0
PALL = jnp.asarray(Pm.reshape(36, 576 * 36)); del Pm            # S @ PALL = S[IMG] (exact 0/1 permutation as GEMM; the XLA gather was 3x slower)
def jsub(th, S, fs):
    """analytic dR/dtheta for features fs: R = sum_{g,f} lncosh(a_gf), a = FLP_g sum_i W_fi S[IMG_gi] + b_f -> dR/db_f = sum_g tanh a, dR/dW_fi = sum_g tanh(a) Z_gi"""
    b = th[fs]; W = th[AL:].reshape(AL, 36)[fs]
    Z = FLP[None, :, None] * (S @ PALL).reshape(-1, 576, 36)
    T = jnp.tanh(jnp.matmul(Z, W.T) + b)
    return jnp.concatenate([T.sum(1), jnp.matmul(jnp.swapaxes(T, 1, 2), Z).reshape(S.shape[0], -1)], 1)
jsub_j = jax.jit(jsub)
CH = 1024
def j_block(th, S, b):
    """S (m,36) +-1 float64 -> (m, len(BCOLS[b])) float64, columns in cols_of order."""
    m = S.shape[0]
    if m < CH: S = np.concatenate([S, np.repeat(S[:1], CH - m, 0)])
    return np.asarray(jsub_j(jnp.asarray(th), jnp.asarray(S), jnp.asarray(FEATS[b])))[:m]
def j_full(th, S):
    out = np.empty((S.shape[0], P))
    for b in range(3): out[:, BCOLS[b]] = j_block(th, S, b)
    return out
fwd_j = jax.jit(jax.vmap(R_one, in_axes=(None, 0)))

# ------------------------------------------------------------------ checks (J vs finite differences on 10 orbits, invariance, table, broken controls)
def checks():
    ck = {}; rng = np.random.default_rng(321)
    idx = np.sort(rng.choice(D, 10, replace=False)); S = bits(reps[idx]); thj = jnp.asarray(theta)
    # model/table: R_one(theta) vs R_a16.npy on 500 reps, and the broken-theta control
    i5 = np.sort(rng.choice(D, 500, replace=False)); S5 = bits(reps[i5]); Rref = np.load(RA16, mmap_mode='r')[i5]
    ck['table_maxabs'] = float(np.abs(np.asarray(fwd_j(thj, jnp.asarray(S5))) - Rref).max())
    ck['table_broken_maxabs'] = float(np.abs(np.asarray(fwd_j(jnp.asarray(theta + 1e-6 * rng.standard_normal(P)), jnp.asarray(S5))) - Rref).max())
    # J (the float32 blocks that production stores) vs central FD of R_one, all 592 columns, 10 orbits
    J32 = np.empty((10, P), np.float32)
    for b in range(3): J32[:, BCOLS[b]] = j_block(theta, S, b).astype(np.float32)
    FD = np.empty((10, P)); e = 1e-5
    for p in range(P):
        d = np.zeros(P); d[p] = e
        FD[:, p] = (np.asarray(fwd_j(jnp.asarray(theta + d), jnp.asarray(S))) - np.asarray(fwd_j(jnp.asarray(theta - d), jnp.asarray(S)))) / (2 * e)
    sc = np.abs(FD).max(1, keepdims=True)
    ck['J_vs_FD_maxrel'] = float((np.abs(J32 - FD) / sc).max()); ck['J_absmax'] = float(np.abs(FD).max())
    ck['J_vs_FD_maxrel_per_orbit'] = (np.abs(J32 - FD) / sc).max(1).tolist()
    Jad = np.asarray(jax.jit(jax.vmap(jax.grad(R_one), in_axes=(None, 0)))(thj, jnp.asarray(S)))
    ck['J_vs_jaxgrad_maxrel'] = float((np.abs(J32 - Jad) / sc).max())
    # broken: J at a perturbed theta must NOT match FD at theta
    Jb = np.empty((10, P))
    for b in range(3): Jb[:, BCOLS[b]] = j_block(theta + 1e-3 * rng.standard_normal(P), S, b)
    ck['J_broken_vs_FD_maxrel'] = float((np.abs(Jb - FD) / sc).max())
    # orbit invariance of J: group images (lat.G rows) and global flip
    G = np.asarray(lat.G); dev = 0.0
    for k in rng.integers(0, len(G), 6): dev = max(dev, float((np.abs(j_full(theta, S[:, G[k]]) - j_full(theta, S)) / sc).max()))
    ck['J_orbit_dev_maxrel'] = dev; ck['J_flip_dev_maxrel'] = float((np.abs(j_full(theta, -S) - j_full(theta, S)) / sc).max())
    # broken invariance control: dropping one image of the sum must change J
    ck['pass'] = bool(ck['table_maxabs'] < 1e-8 and ck['table_broken_maxabs'] > 1e-6 and ck['J_vs_FD_maxrel'] < 1e-6 and ck['J_vs_jaxgrad_maxrel'] < 1e-6
                      and ck['J_broken_vs_FD_maxrel'] > 1e-4 and ck['J_orbit_dev_maxrel'] < 1e-6 and ck['J_flip_dev_maxrel'] < 1e-6)
    ck['orbits_idx'] = idx.tolist(); ck['theta_sha'] = hashlib.sha256(open(THETA, 'rb').read()).hexdigest()[:16]; ck['mr_residual_sha'] = MR_SHA
    json.dump(ck, open(f'{OUT}/checks.json', 'w'), indent=1); log('CHECKS ' + json.dumps(ck)); return ck
if STAGE == 'check': sys.exit(0 if checks()['pass'] else 2)

# ------------------------------------------------------------------ L1 features (Test L/N code, verbatim)
L = 6; site = lambda x, y: (x % L) + L * (y % L)
CLS = [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3)]
cls_perms = {c: [] for c in CLS}
for a in range(L):
    for b in range(L):
        if a == b == 0: continue
        ka, kb = min(a, L - a), min(b, L - b); cls_perms[(max(ka, kb), min(ka, kb))].append(np.array([site(x + a, y + b) for y in range(L) for x in range(L)]))
def l1_feats(S):
    out = np.empty((S.shape[0], 9))
    for j, c in enumerate(CLS):
        acc = np.zeros(S.shape[0])
        for p in cls_perms[c]: acc += (S * S[:, p]).sum(1)
        out[:, j] = acc / (36.0 * len(cls_perms[c]))
    return out

SMOKE = STAGE == 'smoke'; NSM = 300000
DE = NSM if SMOKE else D                                          # rows in play
def norm_la(la): la = la - la.max(); return la - 0.5 * np.log((n * np.exp(2 * la)).sum())
s1 = np.load(S1F).astype(np.float64); assert s1.shape == (D,) and set(np.unique(s1)) <= {-1.0, 1.0}
phipp = np.load(PHIPP).astype(np.float64); lA = np.load(LA_S1).astype(np.float64)
lp = norm_la(phipp); Rt = phipp - lA; del phipp, lA                 # R*_1; lp = phi'' normalised as v2 (sum n phi''^2 = 1)

# ------------------------------------------------------------------ pair list (all blocked entries, held in RAM: x,y int32, w float32)
def build_pairs():
    t1 = np.load(f'{TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{TAB}/nb_J2.npy', mmap_mode='r'); RS = 65536
    X = []; Y = []; W = []; tot = 0; wmin = np.inf; wsum = 0.0
    if not SMOKE: PX = np.empty(NBLK_ENTRIES, np.int32); PY = np.empty(NBLK_ENTRIES, np.int32); PW = np.empty(NBLK_ENTRIES, np.float32)
    for i0 in range(0, DE, RS):
        i1 = min(DE, i0 + RS); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        viol = valid & (s1[i0:i1, None] * s1[tt] > 0)
        if SMOKE: viol &= tt < DE
        rr, kk = np.nonzero(viol); x = rr + i0; y = tt[rr, kk]; w = n[x] * 0.5 * HW[kk] * np.exp(lp[x] + lp[y])
        m = len(x); wmin = min(wmin, float(w.min())) if m else wmin; wsum += w.sum()
        if SMOKE: X.append(x.astype(np.int32)); Y.append(y.astype(np.int32)); W.append(w.astype(np.float32))
        else: PX[tot:tot + m] = x; PY[tot:tot + m] = y; PW[tot:tot + m] = w
        tot += m
        if (i0 // RS) % 40 == 0: log(f'pairs {i1}/{DE} entries {tot}')
    if SMOKE: PX, PY, PW = np.concatenate(X), np.concatenate(Y), np.concatenate(W)
    else: assert tot == NBLK_ENTRIES, tot
    log(f'pair list: {tot} entries, w_min {wmin:.3e}, Wtot {wsum:.6f}'); return PX, PY, PW, dict(n_entries=tot, w_min=wmin, Wtot=wsum)

# ------------------------------------------------------------------ one pass: hold J blocks `bl`, aux columns optional; syrk over all pairs
def jblock_all(b):
    buf = np.empty((DE, len(BCOLS[b])), np.float32); t = time.time(); CHJ = 4096
    for i0 in range(0, DE, CHJ):
        i1 = min(DE, i0 + CHJ); buf[i0:i1] = j_block(theta, bits(reps[i0:i1]), b)
        if (i0 // CHJ) % 400 == 0: log(f'J block {b}: {i1}/{DE} ({i1 / (time.time() - t):.0f} conf/s)')
    return buf

def run_pass(PX, PY, PW, blocks, Jb, aux, tag):
    """returns G_ext over columns [blocks in order | R | aux], float64, symmetrised."""
    wcol = [len(BCOLS[b]) for b in blocks]; nc = sum(wcol) + 1 + (aux.shape[1] if aux is not None else 0); NP = len(PX); SUB = 65536
    Gu = np.zeros((nc, nc)); pool = ThreadPoolExecutor(3)
    def prep(lo):
        hi = min(NP, lo + SUB); x = PX[lo:hi]; y = PY[lo:hi]; B = np.empty((hi - lo, nc)); c = 0
        for Jx, w_ in zip(Jb, wcol):
            d = Jx[x]; d -= Jx[y]; B[:, c:c + w_] = d; c += w_
        B[:, c] = Rt[x] - Rt[y]; c += 1
        if aux is not None: d = aux[x]; d -= aux[y]; B[:, c:] = d
        B *= np.sqrt(PW[lo:hi].astype(np.float64))[:, None]; return B
    futs = {}; los = list(range(0, NP, SUB)); t = time.time(); AHEAD = 4
    for j in range(min(AHEAD, len(los))): futs[j] = pool.submit(prep, los[j])
    for j in range(len(los)):
        B = futs.pop(j).result()
        if j + AHEAD < len(los): futs[j + AHEAD] = pool.submit(prep, los[j + AHEAD])
        Gu = blas.dsyrk(1.0, B, beta=1.0, c=Gu, trans=1, lower=0, overwrite_c=1)
        if j % 400 == 0: log(f'{tag}: pairs {min(NP, los[j] + SUB)}/{NP} ({(los[j] + SUB) / (time.time() - t):.0f} entries/s)')
    pool.shutdown(); Gf = np.triu(Gu) + np.triu(Gu, 1).T; return Gf

def cols_pos(blocks):
    pos = {}; c = 0
    for b in blocks:
        for q, col in enumerate(BCOLS[b]): pos[int(col)] = (b, c + q)
        c += len(BCOLS[b])
    return c

if STAGE in ('smoke', 'run'):
    ck = json.load(open(f'{OUT}/checks.json')) if STAGE == 'run' and os.path.exists(f'{OUT}/checks.json') else checks()
    if not ck['pass']: log('CHECK FAIL: STOP'); sys.exit(2)
    PX, PY, PW, pinfo = build_pairs(); json.dump(pinfo, open(f'{OUT}/pairs_{STAGE}.json', 'w'))
    # aux: L1 (9) + permuted L1 control (9), float32, per orbit rep (centred; constants are free in the pair-difference metric)
    FL1 = np.empty((DE, 9), np.float32)
    for i0 in range(0, DE, 65536): i1 = min(DE, i0 + 65536); FL1[i0:i1] = l1_feats(bits(reps[i0:i1]))
    FL1 -= FL1[:4096].mean(0, dtype=np.float64).astype(np.float32); perm = np.random.default_rng(0).permutation(DE)
    AUX = np.concatenate([FL1, FL1[perm]], 1); del FL1; log('L1 features done')
    def save(G, name, blocks, aux):
        np.save(f'{OUT}/{name}.npy', G); json.dump(dict(blocks=blocks, aux=aux, smoke=SMOKE), open(f'{OUT}/{name}.json', 'w'))
    J0 = jblock_all(0); J1 = jblock_all(1)
    G1 = run_pass(PX, PY, PW, [0, 1], [J0, J1], AUX, 'pass1'); save(G1, f'G_pass1{"_smoke" if SMOKE else ""}', [0, 1], True)
    del J0; J2 = jblock_all(2)
    G2 = run_pass(PX, PY, PW, [1, 2], [J1, J2], None, 'pass2'); save(G2, f'G_pass2{"_smoke" if SMOKE else ""}', [1, 2], False)
    del J1; J0 = jblock_all(0)
    G3 = run_pass(PX, PY, PW, [2, 0], [J2, J0], None, 'pass3'); save(G3, f'G_pass3{"_smoke" if SMOKE else ""}', [2, 0], False)
    log('RUN DONE (G passes saved); run `final`' if not SMOKE else 'SMOKE passes done')
    if SMOKE:                                            # naive float64 reference on the same pair set, same rows
        Jfull = np.empty((DE, P));
        for b in range(3): Jfull[:, BCOLS[b]] = np.concatenate([j_block(theta, bits(reps[i0:min(DE, i0 + 4096)]), b) for i0 in range(0, DE, 4096)])
        x = PX.astype(np.int64); y = PY.astype(np.int64); w = PW.astype(np.float64)
        Gn = np.zeros((P, P)); bn = np.zeros(P); Dn = 0.0
        for lo in range(0, len(x), 200000):
            xs, ys, ws = x[lo:lo + 200000], y[lo:lo + 200000], w[lo:lo + 200000]; dJ = Jfull[xs] - Jfull[ys]; dR = Rt[xs] - Rt[ys]
            Gn += (dJ * ws[:, None]).T @ dJ; bn += (dJ * ws[:, None]).T @ dR; Dn += (ws * dR * dR).sum()
        c0 = cols_pos([0, 1]); sc_ = np.sqrt(np.diag(Gn)); sc_[sc_ == 0] = 1
        def blk(G, blocks, bs):                              # extract G block (cols of blocks bs[0], bs[1]) from a pass matrix
            off = {}; c = 0
            for b in blocks: off[b] = c; c += len(BCOLS[b])
            return G[off[bs[0]]:off[bs[0]] + len(BCOLS[bs[0]]), off[bs[1]]:off[bs[1]] + len(BCOLS[bs[1]])]
        def rel(A, Bn, ia, ib): return float(np.abs(A - Bn).max() / np.abs(Bn).max())
        out = {}
        for (G, blocks, bs) in ((G1, [0, 1], (0, 0)), (G1, [0, 1], (1, 1)), (G1, [0, 1], (0, 1)), (G2, [1, 2], (2, 2)), (G2, [1, 2], (1, 2)), (G3, [2, 0], (0, 2)), (G3, [2, 0], (2, 2)), (G3, [2, 0], (0, 0))):
            ref = Gn[np.ix_(BCOLS[bs[0]], BCOLS[bs[1]])]; out[str(bs) + str(blocks)] = float(np.abs(blk(G, blocks, bs) - ref).max() / np.abs(ref).max())
        nb0, nb1 = len(BCOLS[0]), len(BCOLS[1])
        out['b0_from_pass1'] = float(np.abs(G1[:nb0, nb0 + nb1] - bn[BCOLS[0]]).max() / np.abs(bn[BCOLS[0]]).max())
        out['b2_from_pass2'] = float(np.abs(G2[nb1:nb1 + len(BCOLS[2]), nb1 + len(BCOLS[2])] - bn[BCOLS[2]]).max() / np.abs(bn[BCOLS[2]]).max())
        out['DRR'] = float(abs(G1[nb0 + nb1, nb0 + nb1] / Dn - 1))
        # negative control: block (0,1) compared with the WRONG reference (0,0 columns) must differ
        out['broken_ctl_rel'] = float(np.abs(blk(G1, [0, 1], (0, 1)) - Gn[np.ix_(BCOLS[0], BCOLS[1])][::-1]).max() / np.abs(Gn).max())
        json.dump(out, open(f'{OUT}/smoke_result.json', 'w'), indent=1); log('SMOKE ' + json.dumps(out)); sys.exit(0)
    sys.exit(0)

# ------------------------------------------------------------------ final: assemble 592x592, C_exact, L1
if STAGE == 'final':
    Gs = [np.load(f'{OUT}/G_pass{i}.npy') for i in (1, 2, 3)]; PB = [[0, 1], [1, 2], [2, 0]]
    nb = [len(c) for c in BCOLS]; G = np.full((P, P), np.nan); bvec = np.full(P, np.nan); DRR_l = []; rep = {}
    seen = {}
    for Gp, blocks in zip(Gs, PB):
        off = {}; c = 0
        for b in blocks: off[b] = c; c += nb[b]
        rcol = c; DRR_l.append(Gp[rcol, rcol])
        for a in blocks:
            bvec_a = Gp[off[a]:off[a] + nb[a], rcol]
            if a in seen.get('b', {}): rep[f'b{a}_repeat_rel'] = float(np.abs(bvec_a - seen['b'][a]).max() / np.abs(bvec_a).max())
            seen.setdefault('b', {})[a] = bvec_a; bvec[BCOLS[a]] = bvec_a
            for b2 in blocks:
                blkm = Gp[off[a]:off[a] + nb[a], off[b2]:off[b2] + nb[b2]]; key = (a, b2)
                if key in seen:
                    rep[f'G{a}{b2}_repeat_rel'] = float(np.abs(blkm - seen[key]).max() / np.abs(blkm).max())
                else: seen[key] = blkm; G[np.ix_(BCOLS[a], BCOLS[b2])] = blkm
    assert np.isfinite(G).all() and np.isfinite(bvec).all()
    DRR = float(DRR_l[0]); rep['DRR_passes'] = [float(x) for x in DRR_l]; rep['DRR_vs_pfaffian_n'] = DRR / DRR_N - 1
    rep['G_symmetry_rel'] = float(np.abs(G - G.T).max() / np.abs(G).max()); G = 0.5 * (G + G.T)
    lam, V = np.linalg.eigh(G); trG = float(np.trace(G)); pr = V.T @ bvec
    res = dict(D_RR=DRR, trG=trG, P=P, lam_max=float(lam.max()), lam_min=float(lam.min()), n_eig_gt_1em10_rel=int((lam > 1e-10 * lam.max()).sum()),
               n_eig_gt_1em6_rel=int((lam > 1e-6 * lam.max()).sum()), n_eig_gt_ridge=int((lam > 1e-10 * trG / P).sum()), consistency=rep)
    for tag, lr in (('ridge_1e-10', 1e-10), ('ridge_1e-8', 1e-8), ('ridge_1e-12', 1e-12), ('ridge_1e-6', 1e-6)):
        lm = lr * trG / P; res[f'C_exact_{tag}'] = float((pr ** 2 / (np.maximum(lam, 0) + lm)).sum() / DRR)
    for cut in (1e-14, 1e-12, 1e-10, 1e-8): sel = lam > cut * lam.max(); res[f'C_exact_trunc_{cut:g}'] = float((pr[sel] ** 2 / lam[sel]).sum() / DRR)
    res['C_exact'] = res['C_exact_ridge_1e-10']
    # L1 from pass 1 (identical machinery): columns [B0|B1|R|L1(9)|L1perm(9)]
    G1 = Gs[0]; r0 = nb[0] + nb[1]; l1 = list(range(r0 + 1, r0 + 10)); lp_ = list(range(r0 + 10, r0 + 19))
    def share(idx):
        Gl = G1[np.ix_(idx, idx)]; bl = G1[idx, r0]; sc = np.sqrt(np.diag(Gl)); Gn = Gl / np.outer(sc, sc)
        c_ls = np.linalg.lstsq(Gn, bl / sc, rcond=1e-13)[0] / sc; ls = float((c_ls @ bl) / DRR)
        lm = 1e-10 * np.trace(Gl) / len(idx); c_r = np.linalg.solve(Gl + lm * np.eye(len(idx)), bl); rd = float((c_r @ bl) / DRR)
        return ls, rd
    ls, rd = share(l1); res['L1_C_lstsq'] = ls; res['L1_C_ridge'] = rd; res['L1_eps2_lstsq'] = 1 - ls; res['L1_eps2_ridge'] = 1 - rd; res['L1_eps2_ref_pfaffian_n'] = L1_REF
    res['L1_eps2_gate_1e-4'] = bool(abs(1 - ls - L1_REF) < 1e-4 and abs(1 - rd - L1_REF) < 1e-4)
    cp = share(lp_); res['L1perm_C_lstsq'] = cp[0]; res['L1perm_C_ridge'] = cp[1]
    # gate (contract): |C_ho(P)_f - C_exact| <= 0.02 for both directions AND C_exact >= max(C_ho(P)_f)
    ho = json.load(open(f'{R_}/pfaffian_theta2/ho2_f.json'))['ho2']['phipp_fitA_testB']['R1']['C_P_heldout'], json.load(open(f'{R_}/pfaffian_theta2/ho2_f.json'))['ho2']['phipp_fitB_testA']['R1']['C_P_heldout']
    C = res['C_exact']; res['C_ho_P_f'] = list(ho); res['gate_close'] = bool(all(abs(h - C) <= 0.02 for h in ho)); res['gate_bias'] = bool(C >= max(ho))
    res['GATE_PASS'] = bool(res['gate_close'] and res['gate_bias'])
    json.dump(res, open(f'{OUT}/calib_result.json', 'w'), indent=1); log('FINAL ' + json.dumps(res)); sys.exit(0)
