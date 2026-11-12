#!/usr/bin/env python3
"""Stage 1 (ASTRA_paired.md section 4, item 1): rows and finite propagators, in the paired_fn engine itself.

System: Astra's four-spin Sz=0 example (6 states, bonds BI/BJ/JB as check_paired.py), guides a, b (amplitudes), s, sb (one sign flipped);
and Astra's two-state counterexample.  Deterministic checks: rows of the production TableGuide (gfmc_fn_lib, via ConfigGuide) equal the
independent DenseGuide rows; joint split-rate generator marginals; maximal-coupling parent law (exact enumeration and sampled).
Stochastic checks (engine runs, R replicas x M walkers): block-end mixed energies at T = k dtau compared with the exact finite-T
propagator  E(T) = e^T exp(T L) q0 / 1^T exp(T L) q0  (z = (mean - exact) / SE over replicas; finite-M bias O(1/M) is inside tolerance at
M = 2000 here, see the 'bias' check), for both arms and for Delta.  PASS: correct versions |z| < 4.5 at every T; broken versions must
miss at |z| > 8 at some T (and their miss is reported); exact zero controls give Delta == 0 bitwise in every block; reweight_A raises
SupportViolation on a pair with B-only kept edges.
usage: stage1_checks.py OUTDIR
"""
import os, sys, json, time, hashlib
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from scipy.linalg import expm
import providers as P
import engine as E

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
RES = dict(checks={}, failures=[], code_sha256={f: hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest() for f in ('providers.py', 'engine.py', 'stage1_checks.py')})
def log(m): print(f'[{time.time() - T0:7.1f}s] {m}', flush=True)
def dump(): json.dump(RES, open(f'{OUT}/stage1.json', 'w'), indent=1, default=float)
def record(name, passed, **kw):
    RES['checks'][name] = dict(passed=bool(passed), **kw)
    if not passed: RES['failures'].append(name)
    log(f'{"PASS" if passed else "FAIL"} {name} ' + ' '.join(f'{k}={v}' for k, v in kw.items() if not isinstance(v, (list, dict)))); dump()

# ---------------------------------------------------------------- four-spin system (as check_paired.py)
BI = np.array([0, 1, 2, 3, 0, 1]); BJ = np.array([1, 2, 3, 0, 2, 3]); JB = np.array([1., 1., 1., 1., .5, .5])
a = np.array([1., .6, 1.4, .8, 1.2, .7]); b = np.array([.9, .85, 1.1, .65, 1.35, .8])
s = np.array([1., -1., 1., -1., 1., -1.]); sb = s.copy(); sb[2] *= -1
tgA = P.table_guide(4, BI, BJ, JB, s * a)
basis = tgA.g.basis; n = len(basis)
X = ((basis[:, None] >> np.arange(4)) & 1).astype(np.int8) * 2 - 1
dest = np.tile(np.arange(n)[:, None], (1, len(JB))); offd = np.zeros((n, len(JB)))
for x in range(n):
    for k in range(len(JB)):
        if X[x, BI[k]] != X[x, BJ[k]]:
            dest[x, k] = tgA.g.lookup[basis[x] ^ (1 << BI[k]) ^ (1 << BJ[k])]; offd[x, k] = JB[k] / 2
Hd = tgA.g.diag0(X)
def dense(psi, **kw): return P.DenseGuide(Hd, dest, offd, psi, **kw)
GU = dict(A=s * a, amp=s * b, sign=sb * a, both=sb * b)

# 1a rows: production TableGuide == independent DenseGuide, every guide
err = 0.0
for k, psi in GU.items():
    tg = P.table_guide(4, BI, BJ, JB, psi); dg = dense(psi)
    r1 = tg.rows(X); r2 = dg.rows(np.arange(n))
    err = max(err, float(np.max(np.abs(r1.rate - r2.rate))), float(np.max(np.abs(r1.dFN - r2.dFN))))
    # moves agree: ConfigGuide.move then index == dest
    for kk in range(len(JB)):
        anti = X[:, BI[kk]] != X[:, BJ[kk]]
        if anti.any():
            Y = tg.move(X[anti], np.full(anti.sum(), kk)); assert np.array_equal(tg.g.index(Y), dest[anti, kk])
record('rows_production_vs_dense', err < 1e-13, max_err=err)
# exact operator: E_L of the rows equals (H psi)/psi with the full H, independent of the FN construction
Hfull = np.diag(Hd).astype(float)
for x in range(n):
    for kk in range(len(JB)):
        if offd[x, kk]: Hfull[x, dest[x, kk]] += offd[x, kk]
errEL = max(float(np.max(np.abs(dense(psi).EL_all() - (Hfull @ psi) / psi))) for psi in GU.values())
record('EL_equals_Hpsi_over_psi', errEL < 1e-13, max_err=errEL)

# 1b joint split-rate generator marginals from the engine's own provider rows
def joint_marginals(gA, gB):
    rA = gA.rows(np.arange(n)).rate; rB = gB.rows(np.arange(n)).rate; J = np.zeros((n * n, n * n))
    for x in range(n):
        for y in range(n):
            row = x * n + y
            for k in range(len(JB)):
                c = min(rA[x, k], rB[y, k]); xx = dest[x, k]; yy = dest[y, k]
                for tgt, rate in ((xx * n + yy, c), (xx * n + y, rA[x, k] - c), (x * n + yy, rB[y, k] - c)):
                    J[row, tgt] += rate; J[row, row] -= rate
    QA = np.zeros((n, n)); QB = np.zeros((n, n))
    for x in range(n):
        for k in range(len(JB)):
            QA[x, dest[x, k]] += rA[x, k]; QA[x, x] -= rA[x, k]; QB[x, dest[x, k]] += rB[x, k]; QB[x, x] -= rB[x, k]
    fA = np.repeat(np.eye(n), n, 0); fB = np.tile(np.eye(n), (n, 1))
    return float(np.max(np.abs(J @ fA - np.repeat(QA, n, 0)))), float(np.max(np.abs(J @ fB - np.tile(QB, (n, 1)))))
eA_, eB_ = joint_marginals(dense(GU['A']), dense(GU['both']))
record('joint_generator_marginals', max(eA_, eB_) < 1e-12, A_err=eA_, B_err=eB_)

# 1c resampler: exact law by enumeration of the categorical coupling, and sampled marginals; broken force_A must fail
rng = np.random.default_rng(7)
keyA = np.array([0, 0, 1, 2, 3, 3, 4]); keyB = np.array([0, 1, 1, 2, 5, 3, 3])
pA = rng.random(7); pA /= pA.sum(); pB = rng.random(7); pB /= pB.sum()
ND = 400000
iA, iB, alpha, nc = E.resample_coupled(pA, pB, keyA, keyB, ND, np.random.default_rng(11))
fA = np.bincount(iA, minlength=7) / ND; fB = np.bincount(iB, minlength=7) / ND
zA = np.max(np.abs(fA - pA) / np.sqrt(pA * (1 - pA) / ND)); zB = np.max(np.abs(fB - pB) / np.sqrt(pB * (1 - pB) / ND))
piA = {k: pA[keyA == k].sum() for k in set(keyA) | set(keyB)}; piB = {k: pB[keyB == k].sum() for k in piA}
alpha_ex = sum(min(piA[k], piB[k]) for k in piA)
same = float((keyA[iA] == keyB[iB]).mean()); zs = abs(same - alpha_ex) / np.sqrt(alpha_ex * (1 - alpha_ex) / ND)
record('resampler_marginals_and_alpha', zA < 4.5 and zB < 4.5 and abs(alpha - alpha_ex) < 1e-14 and zs < 4.5,
       zA_max=float(zA), zB_max=float(zB), alpha=alpha, alpha_exact=alpha_ex, P_same=same, z_same=float(zs))
iA2, iB2, _, _ = E.resample_coupled(pA, pB, keyA, keyB, ND, np.random.default_rng(12), force_A_law=True)
fB2 = np.bincount(iB2, minlength=7) / ND; zB2 = np.max(np.abs(fB2 - pB) / np.sqrt(pB * (1 - pB) / ND))
record('broken_force_A_parents_resampler_detected', zB2 > 8, zB_max=float(zB2), max_prob_err=float(np.max(np.abs(fB2 - pB))))

# ---------------------------------------------------------------- stochastic finite-T propagator checks
def exact_E(g, q0, Ts):
    L = g.generator(); e = g.EL_all(); out = []
    for T in Ts:
        f = expm(T * L) @ q0; out.append(float(e @ f / f.sum()))
    return np.array(out)

def ens(gA, gB, mode, q0, M=2000, R=60, nblk=20, dtau=0.1, seed0=1000, **kw):
    """R independent replicas; returns per replica emix_A, emix_B (R, nblk)."""
    EA = np.empty((R, nblk)); EB = np.empty((R, nblk))
    for r in range(R):
        rg = np.random.default_rng(seed0 + r); X0 = rg.choice(len(q0), M, p=q0)
        out = E.run(gA, gB, X0, X0, mode=mode, dtau=dtau, burn_blocks=0, measure_blocks=nblk, pair_seed=seed0 + 7919 * r + 1, **kw)
        EA[r] = out['series']['emix_A']; EB[r] = out['series']['emix_B'] if gB is not None else np.nan
    return EA, EB

def zcmp(sim, exact):
    m = sim.mean(0); se = sim.std(0, ddof=1) / np.sqrt(sim.shape[0]); z = (m - exact) / se
    return z, m, se

q0 = np.ones(n) / n; dtau = 0.1; NBLK = 20; Ts = dtau * np.arange(1, NBLK + 1); iT = [1, 4, 9, 19]
def check_pair(tag, gA, gB, mode, expect_ok=True, **kw):
    EA, EB = ens(gA, gB, mode, q0, nblk=NBLK, dtau=dtau, **kw)
    xA = exact_E(gA if not isinstance(gA, tuple) else gA[0], q0, Ts)
    xB = exact_E(TRUE_B[tag] if tag in TRUE_B else gB, q0, Ts)
    zA, mA, _ = zcmp(EA, xA); zB, mB, seB = zcmp(EB, xB); zD, mD, seD = zcmp(EB - EA, xB - xA)
    zmax = float(max(np.max(np.abs(zA)), np.max(np.abs(zB)), np.max(np.abs(zD))))
    j = int(np.argmax(np.abs(zD)))
    info = dict(mode=mode, zA_max=float(np.max(np.abs(zA))), zB_max=float(np.max(np.abs(zB))), zD_max=float(np.max(np.abs(zD))),
                T_worst=float(Ts[j]), D_sim=float(mD[j]), D_exact=float(xB[j] - xA[j]), D_se=float(seD[j]),
                D_at_T=[dict(T=float(Ts[i]), sim=float(mD[i]), se=float(seD[i]), exact=float(xB[i] - xA[i])) for i in iT])
    if expect_ok: record(f'propagator_{tag}', zmax < 4.5, **info)
    else: record(f'broken_misses_{tag}', zmax > 8, miss=float(mD[j] - (xB[j] - xA[j])), **info)
    return EA, EB

TRUE_B = {}
gA = dense(GU['A'])
for pair in ('amp', 'sign', 'both'):
    gB = dense(GU[pair])
    EA, EB = check_pair(f'coupled_{pair}', gA, gB, 'coupled')
    if pair == 'both': EAc, EBc = EA, EB
    check_pair(f'union_{pair}', gA, gB, 'union')
check_pair('reweight_A_amp', gA, dense(GU['amp']), 'reweight_A')
# coupled arms vs standalone arms (same q0, independent seeds): marginal agreement
SA, _ = ens(gA, None, 'single', q0, nblk=NBLK, seed0=5000); SB, _ = ens(dense(GU['both']), None, 'single', q0, nblk=NBLK, seed0=6000)
zmA = (EAc.mean(0) - SA.mean(0)) / np.sqrt(EAc.var(0, ddof=1) / len(EAc) + SA.var(0, ddof=1) / len(SA))
zmB = (EBc.mean(0) - SB.mean(0)) / np.sqrt(EBc.var(0, ddof=1) / len(EBc) + SB.var(0, ddof=1) / len(SB))
vr = float(np.mean(np.var(EBc - EAc, 0) / (np.var(SA, 0) + np.var(SB, 0))))
record('coupled_marginals_vs_standalone', max(np.max(np.abs(zmA)), np.max(np.abs(zmB))) < 4.5, zA_max=float(np.max(np.abs(zmA))),
       zB_max=float(np.max(np.abs(zmB))), var_ratio_paired_over_independent=vr)
# reweight_A must reject an unsupported B (B-only kept edges): sign-changing pairs
for pair in ('sign', 'both'):
    try:
        ens(gA, dense(GU[pair]), 'reweight_A', q0, R=1, nblk=NBLK); record(f'reweight_A_rejects_unsupported_{pair}', False, note='no SupportViolation raised')
    except E.SupportViolation as e:
        record(f'reweight_A_rejects_unsupported_{pair}', True, message=str(e)[:160])
# broken versions must miss
TRUE_B['coupled_inverse_ratio_provider'] = dense(GU['amp'])
check_pair('coupled_inverse_ratio_provider', gA, dense(GU['amp'], broken='inverse_ratio'), 'coupled', expect_ok=False)
TRUE_B['reweight_A_inverse_hop_ratio'] = dense(GU['amp'])
check_pair('reweight_A_inverse_hop_ratio', gA, dense(GU['amp']), 'reweight_A', expect_ok=False, broken='inverse_hop_ratio')
TRUE_B['coupled_missing_wall_no_vsf'] = dense(GU['both'])
check_pair('coupled_missing_wall_no_vsf', gA, dense(GU['both'], broken='no_vsf'), 'coupled', expect_ok=False)
TRUE_B['coupled_changed_walls_from_A'] = dense(GU['sign'])
check_pair('coupled_changed_walls_from_A', gA, dense(GU['sign'], walls_sign=s), 'coupled', expect_ok=False)
TRUE_B['coupled_force_A_parents'] = dense(GU['both'])
check_pair('coupled_force_A_parents', gA, dense(GU['both']), 'coupled', expect_ok=False, broken='force_A_parents')
# the A-only construction on an unsupported pair, support check disabled: must miss (Astra's 'incorrect A-only construction')
TRUE_B['reweight_A_unsupported_unchecked'] = dense(GU['both'])
def ens_nocheck(R=200):
    EA = np.empty((R, NBLK)); EB = np.empty((R, NBLK))
    for r in range(R):
        rg = np.random.default_rng(9000 + r); X0 = rg.choice(n, 2000, p=q0)
        eng = E.PairedFN(gA, dense(GU['both']), X0, X0, mode='reweight_A', dtau=dtau, seed=9100 + r, check_support=False)
        for i in range(NBLK):
            o = eng.block(); EA[r, i] = o['emix_A']; EB[r, i] = o['emix_B']
    return EA, EB
EA, EB = ens_nocheck(); xA = exact_E(gA, q0, Ts); xB = exact_E(dense(GU['both']), q0, Ts)
zD, mD, seD = zcmp(EB - EA, xB - xA); j = int(np.argmax(np.abs(zD)))
# Astra (checks.json): the incorrect A-only construction on this pair misses E_B by -4.02e-2 at T = 2 (q0 uniform, infinite M)
zAstra = float((EB[:, -1].mean() - xB[-1] - (-4.0198e-2)) / (EB[:, -1].std(ddof=1) / np.sqrt(len(EB))))
record('broken_misses_reweight_A_unsupported_unchecked', abs(zD[j]) > 8, T_worst=float(Ts[j]), D_sim=float(mD[j]), D_exact=float(xB[j] - xA[j]), D_se=float(seD[j]),
       EB_miss_T2=float(EB[:, -1].mean() - xB[-1]), z_vs_Astra_miss_T2=zAstra)

# exact zero controls: identical guide, and global sign flip, in every mode: Delta == 0 bitwise in every block
for nm, psiB in (('identical', GU['both']), ('global_flip', -GU['both'])):
    for mode in ('coupled', 'union', 'reweight_A'):
        X0 = np.random.default_rng(3).choice(n, 500, p=q0)
        out = E.run(dense(GU['both']), dense(psiB), X0, X0, mode=mode, dtau=dtau, measure_blocks=200, pair_seed=4)
        d = out['series']['delta']; hops = np.nansum(out['series']['shared']) + np.nansum(out['series']['hops'])
        record(f'exact_zero_{nm}_{mode}', bool(np.all(d == 0.0)) and hops > 0, max_abs_delta=float(np.max(np.abs(d))), events=float(hops),
               same_post_min=float(np.nanmin(out['series']['same_post'])) if mode == 'coupled' else None)

# two-state counterexample: reweight_A rejects; union and coupled give E_B = -1 at T = 20 (exact), A-only construction gives -1.25
H2d = np.array([0., 0.]); dest2 = np.array([[1], [0]]); off2 = np.array([[1.], [1.]])
g2A = P.DenseGuide(H2d, dest2, off2, np.array([1., 2.])); g2B = P.DenseGuide(H2d, dest2, off2, np.array([1., -2.]))
X0 = np.random.default_rng(5).choice(2, 4000, p=[.5, .5])
try:
    E.run(g2A, g2B, X0, X0, mode='reweight_A', measure_blocks=5); record('two_state_reweight_A_rejects', False)
except E.SupportViolation as e:
    record('two_state_reweight_A_rejects', True, message=str(e)[:120])
for mode in ('union', 'coupled'):
    eb = []
    for r in range(8):
        X0r = np.random.default_rng(50 + r).choice(2, 4000, p=[.5, .5])
        out = E.run(g2A, g2B, X0r, X0r, mode=mode, dtau=0.1, measure_blocks=200, pair_seed=60 + r); eb.append(out['series']['emix_B'][-100:].mean())
    eb = np.array(eb); se = eb.std(ddof=1) / np.sqrt(len(eb))
    record(f'two_state_{mode}_EB', abs(eb.mean() + 1) < 4.5 * se, EB_T10_20=float(eb.mean()), se=float(se), exact=-1.0, replicas=8)
# A-only construction (support check off): its infinite-M answer is e_B averaged over q0 = -1.25 at every T (Astra); at T = 0.5 the
# finite-M population still holds both states; at T = 20 A's resampling has removed state 0 and the value is -0.5.  Both are wrong (-1).
eb5 = []; eb200 = []
for r in range(100):
    X0r = np.random.default_rng(300 + r).choice(2, 2000, p=[.5, .5])
    eng = E.PairedFN(g2A, g2B, X0r, X0r, mode='reweight_A', dtau=0.1, seed=400 + r, check_support=False)
    for i in range(200):
        o = eng.block()
        if i == 4: eb5.append(o['emix_B'])
    eb200.append(o['emix_B'])
eb5 = np.array(eb5); se5 = eb5.std(ddof=1) / 10
record('two_state_A_only_unchecked_is_wrong', abs(eb5.mean() + 1.25) < 4.5 * se5 and abs(eb5.mean() + 1) > 8 * se5 and abs(np.mean(eb200) + 1) > 0.1,
       EB_T05=float(eb5.mean()), se=float(se5), Astra_infinite_M=-1.25, EB_T20_finite_M=float(np.mean(eb200)), exact=-1.0)

# ConfigGuide (production TableGuide rows, X-state path) runs in the engine and agrees with the DenseGuide propagator
Xq = X; EAcfg = np.empty((40, NBLK)); EBcfg = np.empty((40, NBLK))
cA = P.table_guide(4, BI, BJ, JB, GU['A']); cB = P.table_guide(4, BI, BJ, JB, GU['both'])
for r in range(40):
    idx = np.random.default_rng(700 + r).choice(n, 2000, p=q0)
    out = E.run(cA, cB, Xq[idx], Xq[idx], mode='coupled', dtau=dtau, measure_blocks=NBLK, pair_seed=800 + r)
    EAcfg[r] = out['series']['emix_A']; EBcfg[r] = out['series']['emix_B']
xA = exact_E(gA, q0, Ts); xB = exact_E(dense(GU['both']), q0, Ts)
zD, _, _ = zcmp(EBcfg - EAcfg, xB - xA); zA, _, _ = zcmp(EAcfg, xA)
record('configguide_production_rows_in_engine', max(np.max(np.abs(zD)), np.max(np.abs(zA))) < 4.5, zD_max=float(np.max(np.abs(zD))), zA_max=float(np.max(np.abs(zA))))

RES['status'] = 'PASS' if not RES['failures'] else 'FAIL'; RES['sec'] = time.time() - T0; dump()
log(f'VERDICT_STAGE1 {RES["status"]} failures={RES["failures"]}')
