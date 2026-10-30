"""test_theta2.py: Test Theta v2 (PLAN_pfaffian.md "Test Θ v2"), CPU, JAX float64, sign s1. Derived from test_theta.py (sha256 c772978b...):
J code, pair sampler, targets, table/FD checks are verbatim; the held-out pass is streamed (J never stored).
usage:  python3 test_theta2.py OUT prep2                (two pair sets A (seed 10), B (seed 11) of NP2=524288 blocked pairs per target-phi weight)
        python3 test_theta2.py OUT cand NAME            (NAME in b e f; streams G=mean dJ dJ^T, b=mean dJ dR per set, then C_ho(P), C_ho(GN), C_ho(200))
Sign s1. lam_rel (GN): b 0.0625, e 1e-2, f 1e-2 (brief)."""
import os, sys
TH = os.environ.get('THETA_THREADS', '8')
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(k, TH)
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_FLAGS', f'--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={TH}')
import json, time, ast, textwrap, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = sys.argv[1]; STAGE = sys.argv[2]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
R_ = '/scratch/a/A.Otaifi/ansatz_runs'
Z = f'{R_}/pfaffian_z'; SC = f'{R_}/pfaffian_screen'
S1F = f'{Z}/s1.npy'; PHIPP = f'{R_}/pfaffian_r1/target_la_s1.npy'; PHI2 = f'{SC}/ln_phi2.npy'; FJ = f'{Z}/fJ.npy'
LA_S1 = f'{R_}/pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy'
TAB = '/scratch/a/A.Otaifi/ansatz_runs/c2_t0b_6x6/tables'; CSRD = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr'
REPS = '/project/theorie/a/A.Otaifi/Chatty/runs/ll6/data/psi0_6x6_table.npz'
HW = np.concatenate([np.full(72, 0.5), np.full(72, 0.25)])           # bond_split.HW
NP_PAIRS = 32768; SEEDS = (0, 1); NP2 = 524288; SETS2 = (('A', 10), ('B', 11)); CHP = 16384; KS = (10, 50, 200, 1000); RELTHR = 1e-12
def sha(f): return hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16]
def norm_la(la, n): la = la - la.max(); return la - 0.5 * np.log((n * np.exp(2 * la)).sum())

# =============================================================================== prep: row weights + pair samples
if STAGE == 'prep2':
    n = np.load(f'{CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
    s1 = np.load(S1F).astype(np.float64); assert s1.shape == (D,) and set(np.unique(s1)) <= {-1.0, 1.0}
    LP = dict(phipp=norm_la(np.load(PHIPP).astype(np.float64), n), phi2=norm_la(np.load(PHI2).astype(np.float64), n))
    log('loading neighbour tables into RAM'); t1 = np.load(f'{TAB}/nb_J1.npy'); t2 = np.load(f'{TAB}/nb_J2.npy'); log('tables loaded')
    WF = f'{R_}/pfaffian_theta/rowweights.npz'          # v1 row weights
    if os.path.exists(WF): z = np.load(WF); Wr = {k: z[k] for k in LP}; npair = int(z['npair']); log('row weights loaded')
    else:
        Wr = {k: np.zeros(D) for k in LP}; npair = 0; RS = 65536
        for i0 in range(0, D, RS):
            i1 = min(D, i0 + RS); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
            viol = valid & (s1[i0:i1, None] * s1[tt] > 0); rr, kk = np.nonzero(viol); x = rr + i0; y = tt[rr, kk]; npair += rr.size
            for k, lp in LP.items(): Wr[k][i0:i1] = np.bincount(rr, weights=n[x] * 0.5 * HW[kk] * np.exp(lp[x] + lp[y]), minlength=i1 - i0)
            if (i0 // RS) % 40 == 0: log(f'weights {i1}/{D}')
        np.savez(WF, npair=npair, **Wr)
    info = dict(n_blocked_entries=npair, Wtot={k: float(v.sum()) for k, v in Wr.items()}, sha=dict(s1=sha(S1F), phipp=sha(PHIPP), phi2=sha(PHI2)))
    log(f'weights {info}')
    def draw_pairs(rng, M, cumWr, lp):                  # mr_residual.draw_pairs with s -> s1 and lp -> the target's phi
        r = np.minimum(np.searchsorted(cumWr, rng.random(M)), D - 1)
        order = np.argsort(r); r = r[order]
        t = np.concatenate([t1[r], t2[r]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        viol = valid & (s1[r, None] * s1[tt] > 0)
        w = np.where(viol, n[r, None] * 0.5 * HW[None, :] * np.exp(lp[r, None] + lp[tt]), 0.0)
        cw = np.cumsum(w, 1); u = rng.random(M) * cw[:, -1]
        k = np.minimum((cw < u[:, None]).sum(1), 143)
        assert viol[np.arange(M), k].all()
        return r, tt[np.arange(M), k], k
    info['WF_sha'] = sha(WF)
    for k, lp in LP.items():
        cw = np.cumsum(Wr[k]); cw /= cw[-1]
        for sn, sd in SETS2:
            rng = np.random.default_rng(sd if k == 'phipp' else [sd, 1])
            x, y, kk = draw_pairs(rng, NP2, cw, lp)
            np.savez(f'{OUT}/pairs2_{k}_{sn}.npz', x=x, y=y, k=kk)
            info[f'pairs2_{k}_{sn}'] = dict(seed=sd, unique_x=int(np.unique(x).size), unique_xy=int(np.unique(np.concatenate([x, y])).size))
            log(f'pairs {k} set {sn}: {info[f"pairs2_{k}_{sn}"]}')
    json.dump(info, open(f'{OUT}/prep2.json', 'w'), indent=1); log('PREP2 DONE'); sys.exit(0)

# =============================================================================== models (verbatim source blocks)
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from rbmpp_lib import Lattice
from scipy.stats import spearmanr
lat = Lattice(6, 0.5, 'full'); IMG = lat.IMG; INV = lat.INV; FLP = lat.FL.astype(jnp.float64); N = 36; NH = 18
xy_ = np.array([(i % 6, i // 6) for i in range(N)])
DISP = jnp.asarray(np.array([[((xy_[j, 0] - xy_[i, 0]) % 6) + 6 * ((xy_[j, 1] - xy_[i, 1]) % 6) for j in range(N)] for i in range(N)], np.int32))
BFW = 3; HWB = 1; ND9 = 9
NBRW = jnp.asarray(np.array([[((x + dx) % 6) + 6 * ((y + dy) % 6) for dy in range(-HWB, HWB + 1) for dx in range(-HWB, HWB + 1)]
                             for y in range(6) for x in range(6)], np.int32))
NP_REAL = N * N + 2 * ND9 * N

def model_block(fn):
    src = open(os.path.join(HERE, fn)).read().split('\n')
    i0 = next(i for i, l in enumerate(src) if '# C1 MODEL BEGIN' in l); i1 = next(i for i, l in enumerate(src) if '# C1 MODEL END' in l)
    return textwrap.dedent('\n'.join(src[i0 + 1:i1])), hashlib.sha256(open(os.path.join(HERE, fn), 'rb').read()).hexdigest()[:16]

def pp_model(fn, rbmpp):
    code, h = model_block(fn)
    ns = dict(jax=jax, jnp=jnp, N=N, NH=NH, NBRW=NBRW, DISP=DISP, IMG=IMG, FLP=FLP, NP_REAL=NP_REAL, ND9=ND9,
              COMPLEX=True, BF=not rbmpp, RBMPP=rbmpp, NA=4, BREAK_R=False)
    exec(code, ns)
    return ns['logA_one'], h

def mr_model(alpha):
    src = open(os.path.join(HERE, 'mr_residual.py')).read(); tree = ast.parse(src)
    fdef = next(nd for nd in tree.body if isinstance(nd, ast.FunctionDef) and nd.name == 'make_model')
    ns = dict(jax=jax, jnp=jnp, IMG=IMG, INV=INV, FLP=FLP)
    exec(ast.get_source_segment(src, fdef), ns)
    P, R_one, *_ = ns['make_model'](alpha)
    return R_one, P, hashlib.sha256(src.encode()).hexdigest()[:16]

LS = f'{R_}/pfaffian_lswo/run/round_1'
CAND = dict(
    a=dict(theta=f'{R_}/pfaffian_c/runs/c1_cplx_s1/round_1/theta.npy', run=f'{R_}/pfaffian_c/runs/c1_cplx_s1/round_1', src='cx_c1fit.py', kind='ppbf'),
    b=dict(theta=f'{LS}/theta.npy', run=LS, src='cx_r1fit.py', kind='rbmpp'),
    c=dict(theta=f'{R_}/pfaffian_r1/run_a40/round_1/theta.npy', run=f'{R_}/pfaffian_r1/run_a40/round_1', src='cx_r1fit.py', kind='rbmpp'),
    d=dict(theta=f'{R_}/pfaffian_r1/run_b/round_1/theta.npy', run=f'{R_}/pfaffian_r1/run_b/round_1', src='cx_r1fit.py', kind='rbmpp'),
    e=dict(theta=f'{LS}/theta.npy', run=LS, src='cx_r1fit.py', kind='rbmpp', rbm_init=True),
    f=dict(theta=f'{R_}/pfaffian_mr/theta_a16.npy', run=f'{R_}/pfaffian_mr', src='mr_residual.py', kind='mr'))

NAME = sys.argv[3]; C = CAND[NAME]
res = dict(candidate=NAME, spec=C, host=os.uname().nodename, threads=TH, sign='s1', s1_sha=sha(S1F), N_pairs=NP_PAIRS)
RJ = f'{OUT}/ho2_{NAME}.json'
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)

theta = np.load(C['theta']).astype(np.float64); res['theta_sha'] = sha(C['theta'])
if C['kind'] == 'mr':
    one, P_, h = mr_model(16); assert theta.size == P_
    res['code_sha'] = h; la_ref_file = f'{C["run"]}/R_a16.npy'
else:
    one, h = pp_model(C['src'], C['kind'] == 'rbmpp'); res['code_sha'] = h; la_ref_file = f'{C["run"]}/la_table.npy'
    if C['kind'] == 'rbmpp':
        assert theta.size == 2 * N * N + 4 * 37, theta.size
        rbm_blk = theta[2 * N * N:]; res['rbm_block_maxabs_in_file'] = float(np.abs(rbm_blk).max())
        if C.get('rbm_init'):
            theta = theta.copy(); theta[2 * N * N:] = 0.01 * np.random.default_rng(0).standard_normal(4 * 37)   # (e): W, b ~ N(0, 0.01^2), seed 0
    else: assert theta.size == 2 * NP_REAL
P = theta.size; res['P'] = P
thj = jnp.asarray(theta)
reps = np.load(REPS)['reps']; D = reps.shape[0]
def bits_of(idx): return (((reps[idx][:, None] >> np.arange(36, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.float64) * 2 - 1)
CH = 64 if C['kind'] != 'mr' else 256
fwd_j = jax.jit(jax.vmap(one, in_axes=(None, 0)))
jac_j = jax.jit(jax.vmap(jax.grad(one), in_axes=(None, 0)))
def run_chunks(f, th, idx, width=None):
    out = []
    for i in range(0, len(idx), CH):
        S = bits_of(idx[i:i + CH]); m = S.shape[0]
        if m < CH: S = np.concatenate([S, np.repeat(S[:1], CH - m, 0)])
        out.append(np.asarray(f(th, jnp.asarray(S)))[:m])
    return np.concatenate(out)

# ---- table check (1000 reps) and its negative control, then a finite-difference check of J
ck = np.sort(np.random.default_rng(123).choice(D, 1000, replace=False))
la_ref = np.load(la_ref_file)[ck]; res['la_ref_file'] = la_ref_file
if C.get('rbm_init'):                                          # (e): the reference is (b)'s table + the RBM term, numpy, independently
    wb = theta[2 * N * N:].reshape(4, 37)
    Sg = bits_of(ck); Zimg = np.asarray(FLP)[None, :, None] * Sg[:, np.asarray(IMG)]          # (1000, 576, 36)
    a_ = Zimg @ wb[:, :36].T + wb[:, 36][None, None, :]
    la_ref = la_ref + np.sum(np.logaddexp(a_, -a_) - np.log(2.0), axis=(1, 2)); res['la_ref_note'] = "(b) la_table + sum ln cosh(b + W z_g) (numpy)"
t = time.time(); la_net = run_chunks(fwd_j, thj, ck); res['fwd_sec_1000'] = time.time() - t
res['check_table_maxabs'] = float(np.abs(la_net - la_ref).max()); res['check_table_pass'] = res['check_table_maxabs'] < 1e-8
thb = jnp.asarray(theta + 1e-6 * np.random.default_rng(7).standard_normal(P))
res['check_table_broken_maxabs'] = float(np.abs(run_chunks(fwd_j, thb, ck[:128]) - la_ref[:128]).max())
log(f'{NAME}: table check max|diff| {res["check_table_maxabs"]:.2e} (broken theta: {res["check_table_broken_maxabs"]:.2e}); {res["fwd_sec_1000"]:.1f}s/1000 fwd')
dump()
if not res['check_table_pass'] or res['check_table_broken_maxabs'] < 1e-8: log('CHECK FAIL: STOP'); res['status'] = 'CHECK FAIL'; dump(); sys.exit(2)
fdi = ck[:2]; Jt = run_chunks(jac_j, thj, fdi); pidx = np.random.default_rng(11).choice(np.nonzero(np.abs(Jt).sum(0) > 0)[0], 4, replace=False); fd = []
for p in pidx:
    e = np.zeros(P); e[p] = 1e-5
    fd.append((run_chunks(fwd_j, jnp.asarray(theta + e), fdi) - run_chunks(fwd_j, jnp.asarray(theta - e), fdi)) / 2e-5)
fd = np.array(fd).T; res['check_fd_maxrel'] = float(np.abs(fd - Jt[:, pidx]).max() / np.abs(Jt[:, pidx]).max()); log(f'FD check rel {res["check_fd_maxrel"]:.1e}'); dump()
assert res['check_fd_maxrel'] < 1e-5

if False:
    nc = int(sys.argv[4]); idx = np.sort(np.random.default_rng(5).choice(D, nc, replace=False))
    t = time.time(); _ = run_chunks(jac_j, thj, idx[:CH]); t1_ = time.time() - t
    t = time.time(); _ = run_chunks(jac_j, thj, idx); dt = time.time() - t
    res['probe'] = dict(n=nc, sec=dt, conf_per_sec=nc / dt, first_chunk_sec_incl_compile=t1_); dump(); log(f'probe {res["probe"]}'); sys.exit(0)

# =============================================================================== streamed held-out analysis
n = np.load(f'{CSRD}/norb.npy').astype(np.float64)
lA = np.load(LA_S1).astype(np.float64); fJ = np.load(FJ).astype(np.float64)
TG = {'R1': np.load(PHIPP).astype(np.float64) - lA, 'R2': np.load(PHI2).astype(np.float64) - (lA + 3.9 * fJ), 'fJ': fJ,
      'noise': np.random.default_rng(999).standard_normal(D)}
del lA
res['targets_sha'] = dict(phipp=sha(PHIPP), phi2=sha(PHI2), fJ=sha(FJ), la_A_s1=sha(LA_S1))
LAMREL = dict(b=0.0625, e=1e-2, f=1e-2)[NAME]; res['lam_rel_GN'] = LAMREL; res['lam_rel_P'] = 1e-6
NTOP = 200
ONLY = sys.argv[4].split(',') if len(sys.argv) > 4 else ['phipp', 'phi2']          # optional: restrict to some weight kinds

def accumulate(wk, sn, names):
    """G = mean dJ dJ^T, b = mean dJ dR, D_RR over the pair set, accumulated in chunks of CHP pairs (dJ never stored).
    DEVIATION from the brief: J of the set's UNIQUE configurations (measured 2.6e5 per set, ~6-8 GB float64) is held in RAM so every
    configuration is evaluated once; re-evaluating per chunk costs 2.7x more. J is never stored per pair and never written to disk."""
    z = np.load(f'{OUT}/pairs2_{wk}_{sn}.npz'); x, y = z['x'], z['y']; M = len(x)
    dR = np.stack([TG[k][x] - TG[k][y] for k in names], 1)
    u, inv = np.unique(np.concatenate([x, y]), return_inverse=True); ix, iy = inv[:M], inv[M:]
    Ju = np.empty((u.size, P)); t = time.time(); tl = t
    for j0 in range(0, u.size, 4096):
        Ju[j0:j0 + 4096] = run_chunks(jac_j, thj, u[j0:j0 + 4096]); j1 = min(j0 + 4096, u.size); el = time.time() - t
        log(f'{NAME} {wk} {sn}: J {j1}/{u.size} conf; {j1 / el:.2f} conf/s cumulative; last chunk {4096 / (time.time() - tl):.2f} conf/s; ETA {(u.size - j1) / (j1 / el) / 3600:.2f} h')
        if j0 == 0: res.setdefault('J_rate_first_chunk_conf_per_s', {})[f'{wk}_{sn}'] = (j1) / el; dump()
        tl = time.time()
    tj = time.time() - t
    G = np.zeros((P, P)); bb = np.zeros((P, len(names)))
    for i0 in range(0, M, CHP):
        i1 = min(i0 + CHP, M); dJ = Ju[ix[i0:i1]] - Ju[iy[i0:i1]]
        G += dJ.T @ dJ; bb += dJ.T @ dR[i0:i1]
    del Ju, dJ
    log(f'{NAME} {wk} {sn}: G, b accumulated from {M} pairs, {u.size} unique configs, J {tj:.0f}s')
    return G / M, bb / M, (dR * dR).mean(0), tj, int(u.size)

def estimators(Gt, bt, Dt, Ge, be, De, j):
    lam, V = np.linalg.eigh(Gt); trG = np.trace(Gt)
    pr = V.T @ bt[:, j]
    lamP = 1e-6 * trG / P; lamG = LAMREL * trG / P
    cs = dict(P=V @ (pr / (np.maximum(lam, 0) + lamP)), GN=V @ (pr / (np.maximum(lam, 0) + lamG)))
    top = np.argsort(lam)[::-1][:NTOP]; cs['200'] = V[:, top] @ (pr[top] / lam[top])
    out = {}
    for k, c in cs.items():
        out[f'C_{k}_heldout'] = float(1 - (De[j] - 2 * c @ be[:, j] + c @ Ge @ c) / De[j])
        out[f'C_{k}_in_overfit_not_quotable'] = float(1 - (Dt[j] - 2 * c @ bt[:, j] + c @ Gt @ c) / Dt[j])
    out['trG'] = float(trG); out['lam_max'] = float(lam.max()); out['lamP'] = lamP; out['lamGN'] = lamG
    return out

res['ho2'] = {}
for wk, names in (('phipp', ['R1', 'fJ', 'noise']), ('phi2', ['R2'])):
    if wk not in ONLY: continue
    AB = {}
    for sn, sd in SETS2:
        G, bb, Dr, ts, nev = accumulate(wk, sn, names); AB[sn] = (G, bb, Dr)
        res.setdefault('J_stats', {})[f'{wk}_{sn}'] = dict(sec_J=ts, n_J_evals=nev, conf_per_s=nev / ts)
        np.savez(f'{OUT}/Gb_{NAME}_{wk}_{sn}.npz', G=G, b=bb, DRR=Dr, names=np.array(names)); dump()
    for tr, te in (('A', 'B'), ('B', 'A')):
        Gt, bt, Dt = AB[tr]; Ge, be, De = AB[te]
        res['ho2'][f'{wk}_fit{tr}_test{te}'] = {nm: dict(estimators(Gt, bt, Dt, Ge, be, De, j), D_RR_test=float(De[j])) for j, nm in enumerate(names)}
        dump(); log(f'{NAME} {wk} fit {tr} test {te}: ' + json.dumps({nm: {k: round(v, 4) for k, v in d.items() if k.startswith('C_') and 'heldout' in k} for nm, d in res['ho2'][f'{wk}_fit{tr}_test{te}'].items()}))
res['status'] = 'DONE'; dump(); log(f'THETA2 DONE {NAME}')
