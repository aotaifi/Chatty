"""test_theta.py: pre-registered Test Theta (PLAN_pfaffian.md "Test Θ" + "Network questions" item 2), CPU, JAX float64, sign s1.
usage:  python3 test_theta.py OUT prep                 (row weights for phi'' and phi_2, pair samples seeds 0, 1; cached)
        python3 test_theta.py OUT cand NAME             (NAME in a b c d e f; one candidate)
        python3 test_theta.py OUT probe NAME NCONF      (J rate only)
Threads: the caller sets OMP/BLAS/XLA threads before python starts (THETA_THREADS also read here before numpy import).
Pairs: blocked pairs (s1_x s1_y > 0) drawn prop. to w = n_x 0.5 HW_k phi_x phi_y (test_k.py / mr_residual.py weights and sampler),
phi = phi'' (pfaffian_r1/target_la_s1.npy) for R*_1 and f_J, phi = phi_2 (pfaffian_screen/ln_phi2.npy) for R*_2.
G = mean dJ dJ^T, b = mean dJ dR, D_RR = mean dR^2 over N_p = 32768 sampled pairs; s_i = (v_i.b)^2 / (lambda_i D_RR), lambda_i > 1e-12 lambda_max.
Models: the 'C1 MODEL' blocks of cx_c1fit.py / cx_r1fit.py and mr_residual.make_model are exec'd verbatim from the source files."""
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
NP_PAIRS = 32768; SEEDS = (0, 1); KS = (10, 50, 200, 1000); RELTHR = 1e-12
def sha(f): return hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16]
def norm_la(la, n): la = la - la.max(); return la - 0.5 * np.log((n * np.exp(2 * la)).sum())

# =============================================================================== prep: row weights + pair samples
if STAGE == 'prep':
    n = np.load(f'{CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
    s1 = np.load(S1F).astype(np.float64); assert s1.shape == (D,) and set(np.unique(s1)) <= {-1.0, 1.0}
    LP = dict(phipp=norm_la(np.load(PHIPP).astype(np.float64), n), phi2=norm_la(np.load(PHI2).astype(np.float64), n))
    log('loading neighbour tables into RAM'); t1 = np.load(f'{TAB}/nb_J1.npy'); t2 = np.load(f'{TAB}/nb_J2.npy'); log('tables loaded')
    WF = f'{OUT}/rowweights.npz'
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
    for k, lp in LP.items():
        cw = np.cumsum(Wr[k]); cw /= cw[-1]
        for sd in SEEDS:
            x, y, kk = draw_pairs(np.random.default_rng(1000 * sd + (0 if k == 'phipp' else 1)), NP_PAIRS, cw, lp)
            np.savez(f'{OUT}/pairs_{k}_seed{sd}.npz', x=x, y=y, k=kk)
            info[f'pairs_{k}_seed{sd}'] = dict(unique_x=int(np.unique(x).size), unique_xy=int(np.unique(np.concatenate([x, y])).size))
            log(f'pairs {k} seed {sd}: {info[f"pairs_{k}_seed{sd}"]}')
    json.dump(info, open(f'{OUT}/prep.json', 'w'), indent=1); log('PREP DONE'); sys.exit(0)

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
RJ = f'{OUT}/theta_{NAME}.json'
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ if STAGE == 'cand' else f'{OUT}/{STAGE}_{NAME}.json', 'w'), indent=1, default=float)

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

if STAGE == 'probe':
    nc = int(sys.argv[4]); idx = np.sort(np.random.default_rng(5).choice(D, nc, replace=False))
    t = time.time(); _ = run_chunks(jac_j, thj, idx[:CH]); t1_ = time.time() - t
    t = time.time(); _ = run_chunks(jac_j, thj, idx); dt = time.time() - t
    res['probe'] = dict(n=nc, sec=dt, conf_per_sec=nc / dt, first_chunk_sec_incl_compile=t1_); dump(); log(f'probe {res["probe"]}'); sys.exit(0)

# =============================================================================== the kernel analysis
n = np.load(f'{CSRD}/norb.npy').astype(np.float64)
lA = np.load(LA_S1).astype(np.float64); fJ = np.load(FJ).astype(np.float64)
TG = {'R1': np.load(PHIPP).astype(np.float64) - lA, 'R2': np.load(PHI2).astype(np.float64) - (lA + 3.9 * fJ), 'fJ': fJ,
      'noise': np.random.default_rng(999).standard_normal(D)}
del lA
res['targets_sha'] = dict(phipp=sha(PHIPP), phi2=sha(PHI2), fJ=sha(FJ), la_A_s1=sha(LA_S1))
LAM_REL = None; lam_src = 'absent -> 1e-2'
cj = f'{C["run"]}/cx_t1s.json'
if C['kind'] != 'mr' and not C.get('rbm_init') and os.path.exists(cj):
    hh = json.load(open(cj))['fit']['hist']; LAM_REL = float(hh[-1]['lam_next']); lam_src = f'{cj}: fit.hist[-1].lam_next (step {hh[-1]["step"]})'
if LAM_REL is None: LAM_REL = 1e-2
res['lam_rel'] = LAM_REL; res['lam_rel_source'] = lam_src
SETS = dict(phipp=['R1', 'fJ', 'R2', 'noise'], phi2=['R2', 'R1'])     # the first target(s) are in their own metric; R2 in phipp and R1 in phi2 = transfer only

def analyse(dJ, dR, names):
    Np = dJ.shape[0]
    G = dJ.T @ dJ / Np; b = dJ.T @ dR / Np; DRR = (dR * dR).mean(0)
    lam, V = np.linalg.eigh(G); lam = lam[::-1]; V = V[:, ::-1]
    keep = lam > RELTHR * lam[0]; lamk = lam[keep]; proj = V[:, keep].T @ b                       # (r, T)
    s = proj ** 2 / (lamk[:, None] * DRR[None, :])
    trG = float(np.trace(G)); lam_damp = LAM_REL * trG / P
    colnorm = np.sqrt(np.diag(G))
    Gp = np.linalg.pinv(G, rcond=RELTHR, hermitian=True); bGb = np.einsum('pt,pq,qt->t', b, Gp, b) / DRR
    c_ls, *_ = np.linalg.lstsq(dJ, dR, rcond=np.sqrt(RELTHR)); r2 = 1 - ((dR - dJ @ c_ls) ** 2).sum(0) / (dR * dR).sum(0)
    out = dict(trG=trG, lam_max=float(lam[0]), rank=int(keep.sum()), P_eff_cols=int((colnorm > 1e-8 * colnorm.max()).sum()),
               lam_damp=lam_damp, n_above_damp=int((lamk > lam_damp).sum()))
    for j, nm in enumerate(names):
        cs = np.cumsum(s[:, j])
        out[nm] = dict(D_RR=float(DRR[j]), C_P=float(cs[-1]), C_P_pinv=float(bGb[j]), C_P_lstsq=float(r2[j]),
                       C_k={str(k): float(cs[min(k, len(cs)) - 1]) for k in KS}, C_GN=float(s[lamk > lam_damp, j].sum()))
    return out, lamk, s, V[:, 0], G

if STAGE == 'ho':      # held-out (added after the contract run): fit c on one pair sample, score it on the other
    def jset(sd, wk, names):
        z = np.load(f'{OUT}/pairs_{wk}_seed{sd}.npz'); x, y = z['x'], z['y']
        u, inv = np.unique(np.concatenate([x, y]), return_inverse=True); Ju = np.empty((u.size, P))
        for i0 in range(0, u.size, 4096): Ju[i0:i0 + 4096] = run_chunks(jac_j, thj, u[i0:i0 + 4096])
        dJ = Ju[inv[:NP_PAIRS]] - Ju[inv[NP_PAIRS:]]; del Ju
        dR = np.stack([TG[k][x] - TG[k][y] for k in names], 1)
        return dJ.T @ dJ / NP_PAIRS, dJ.T @ dR / NP_PAIRS, (dR * dR).mean(0)
    res['ho'] = {}
    for wk, names in dict(phipp=['R1', 'fJ', 'noise'], phi2=['R2']).items():
        GB = {sd: jset(sd, wk, names) for sd in SEEDS}; log(f'{NAME} {wk}: J done')
        for a_, b_ in ((0, 1), (1, 0)):
            G, b, Dr = GB[a_]; Go, bo, Do = GB[b_]
            lam, V = np.linalg.eigh(G); keep = lam > RELTHR * lam.max(); lam_damp = LAM_REL * np.trace(G) / P
            cP = V[:, keep] @ ((V[:, keep].T @ b) / lam[keep][:, None])
            cG = V @ ((V.T @ b) / (np.maximum(lam, 0) + lam_damp)[:, None])
            out = {}
            for j, nm in enumerate(names):
                sc = lambda c: float(1 - (c @ Go @ c - 2 * c @ bo[:, j] + Do[j]) / Do[j])
                out[nm] = dict(C_P_in=float(cP[:, j] @ b[:, j] / Dr[j]), C_P_heldout=sc(cP[:, j]), C_GNstep_in=float(1 - (cG[:, j] @ G @ cG[:, j] - 2 * cG[:, j] @ b[:, j] + Dr[j]) / Dr[j]), C_GNstep_heldout=sc(cG[:, j]))
            res['ho'][f'{wk}_fit{a_}_test{b_}'] = out; dump(); log(f'{NAME} {wk} fit {a_} test {b_}: {out}')
    res['status'] = 'DONE'; dump(); log(f'THETA HO DONE {NAME}'); sys.exit(0)
t_all = time.time(); res['runs'] = {}
for sd in SEEDS:
    for wk, names in SETS.items():
        z = np.load(f'{OUT}/pairs_{wk}_seed{sd}.npz'); x, y = z['x'], z['y']
        u, inv = np.unique(np.concatenate([x, y]), return_inverse=True)
        t = time.time(); Ju = np.empty((u.size, P))
        for i0 in range(0, u.size, 4096):
            Ju[i0:i0 + 4096] = run_chunks(jac_j, thj, u[i0:i0 + 4096])
            log(f'{NAME} seed {sd} {wk}: J {min(i0 + 4096, u.size)}/{u.size}  {(min(i0 + 4096, u.size)) / (time.time() - t):.1f} conf/s')
        tj = time.time() - t
        dJ = Ju[inv[:NP_PAIRS]] - Ju[inv[NP_PAIRS:]]; del Ju
        dR = np.stack([TG[k][x] - TG[k][y] for k in names], 1)
        out, lamk, s, v1, G = analyse(dJ, dR, names)
        # control: R = J v_1 (in the span) -> C(P) = 1
        dc = (dJ @ v1)[:, None]; oc, *_ = analyse(dJ, dc, ['Jv1']); out['ctl_Jv1'] = oc['Jv1']
        rho = spearmanr(s[:, names.index('R1')], s[:, names.index('R2')]).correlation      # transfer, in this pair set's kernel
        out['spearman_R1_R2'] = float(rho); out['J_sec'] = tj; out['n_unique_conf'] = int(u.size); out['J_conf_per_sec'] = u.size / tj
        res['runs'][f'seed{sd}_{wk}'] = out
        np.savez(f'{OUT}/spec_{NAME}_seed{sd}_{wk}.npz', lam=lamk, s=s, names=np.array(names))
        dump(); log(f'{NAME} seed {sd} {wk}: ' + json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ("C_P", "C_P_pinv", "C_P_lstsq", "C_GN")}) for k, v in out.items()}, default=float))
        del dJ, G
res['status'] = 'DONE'; res['analysis_sec'] = time.time() - t_all; dump(); log(f'THETA DONE {NAME}')
