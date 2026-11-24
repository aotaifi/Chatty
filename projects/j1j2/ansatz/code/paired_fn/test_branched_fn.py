#!/usr/bin/env python3
"""test_branched_fn.py OUT : CPU validation of branched_fn on a 3x4 open Heisenberg lattice (J1 = 1, Sz = 0, 924 states, DenseGuide providers).
A = Marshall sign x perturbed exact amplitude: exact nodes, so E_FN[A] = E0 (exact).  Exact Delta of any B = lowest eigenvalue of the FN generator
of B (stage-1 convention) minus that of A, from dense linear algebra.
(1) A = A control: Delta == 0 bitwise on every pair, every block (also the in-driver control on every snapshot).
(1b) the control itself fires when B != A (amplitude of B perturbed): ControlFailure.
(2) broken control B = A with the sign of the most frequent configuration class flipped (the states carrying 10% of |psi_A|^2 weight, taken from the
    top of the weight list).  EXPECTATION written before the run: every FN energy is >= E0 and E_FN[A] = E0, so Delta > 0; the exact value is computed
    densely; the branched estimate must have the right sign (> 5 sigma) and lie within 4 sigma + the population-control bias of the exact value.
(3) the A chain is independent of the pairs (snapshots identical with and without pairs).
(4) burn adequacy: pair Delta over blocks 20-70 vs 70-120 agree."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import providers as P, branched_fn as BF

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); RES = dict(checks={})
def rec(name, ok, **kw):
    RES['checks'][name] = dict(passed=bool(ok), **kw); print('PASS' if ok else 'FAIL', name, kw, flush=True)
    json.dump(RES, open(f'{OUT}/test_branched.json', 'w'), indent=1, default=float)

# ---- lattice 3x4 open, J1 only
Lx, Ly = 3, 4; N = Lx * Ly; idx = lambda x, y: x * Ly + y
bonds = [(idx(x, y), idx(x + 1, y)) for x in range(Lx - 1) for y in range(Ly)] + [(idx(x, y), idx(x, y + 1)) for x in range(Lx) for y in range(Ly - 1)]
BI = np.array([b[0] for b in bonds]); BJ = np.array([b[1] for b in bonds]); NB = len(bonds)
states = [s for s in range(1 << N) if bin(s).count('1') == N // 2]; n = len(states); lut = {s: i for i, s in enumerate(states)}
X = np.array([[(s >> i) & 1 for i in range(N)] for s in states]) * 2 - 1
dest = np.tile(np.arange(n)[:, None], (1, NB)); offd = np.zeros((n, NB)); Hd = np.zeros(n)
for i, s in enumerate(states):
    for k in range(NB):
        a, b = BI[k], BJ[k]
        if X[i, a] == X[i, b]: Hd[i] += 0.25
        else: Hd[i] -= 0.25; dest[i, k] = lut[s ^ (1 << a) ^ (1 << b)]; offd[i, k] = 0.5
H = np.diag(Hd)
for i in range(n):
    for k in range(NB):
        if offd[i, k]: H[i, dest[i, k]] += offd[i, k]
w, v = np.linalg.eigh(H); E0 = w[0]; psi0 = np.abs(v[:, 0])
sub = np.array([(x // Ly + x % Ly) % 2 for x in range(N)]); marsh = np.array([(-1.0) ** ((X[i] > 0) & (sub == 0)).sum() for i in range(n)])
rg = np.random.default_rng(7); amp = psi0 ** 0.8 * np.exp(0.15 * rg.standard_normal(n)); amp /= amp.max()
psiA = marsh * amp
def fn_energy(g):
    return float(-np.max(np.linalg.eigvals(g.generator()).real))
gA = P.DenseGuide(Hd, dest, offd, psiA); gA2 = P.DenseGuide(Hd, dest, offd, psiA.copy())
EA_ex = fn_energy(gA)
rec('exact_E_FN_A_equals_E0', abs(EA_ex - E0) < 1e-9, E_FN_A=EA_ex, E0=E0)
# broken B: flip the sign of the heaviest configurations carrying 10% of |psi_A|^2 (indices from the top of the weight list)
wt = psi0 ** 2 / (psi0 ** 2).sum(); order = np.argsort(-wt); cum = np.cumsum(wt[order]); cls = order[:int(np.searchsorted(cum, 0.10)) + 1]
psiB = psiA.copy(); psiB[cls] *= -1; gB = P.DenseGuide(Hd, dest, offd, psiB)
EB_ex = fn_energy(gB); D_ex = (EB_ex - EA_ex) / N
print(f'class: {len(cls)} configurations, weight {wt[cls].sum():.3f}; exact Delta/site = {D_ex:+.5f} (E_FN[B] {EB_ex:.5f}, E0 {E0:.5f})', flush=True)
rec('exact_Delta_positive', D_ex > 0, Delta_site_exact=D_ex, n_class=len(cls), class_weight=float(wt[cls].sum()))

M = 2000; X0 = np.random.default_rng(1).choice(n, M, p=wt); KW = dict(K=20, burn_A=60, spacing=15, burn_pair=20, measure_pair=100, dtau=0.1, seed=11, NS=N, log=lambda m: None)

# (1) A = A: Delta == 0 on every pair and block, plus the in-driver control on every snapshot
pcs = []
def on_pair(k, pr, s): pcs.append(s)
oc = BF.run_branched(gA, gA2, X0, gB_ctl=gA2, control_blocks=5, on_pair=on_pair, **KW)
mx = max(float(np.max(np.abs(s['delta']))) for s in pcs); spmin = min(float(np.min(s['same_post'])) for s in pcs)
rec('control_A_eq_A_zero_every_pair', mx == 0 and spmin == 1 and len(pcs) == KW['K'] and all(p['control']['passed'] for p in oc['pairs']),
    max_abs_delta=mx, same_post_min=spmin, pairs=len(pcs), delta_site_each=[p['delta_site'] for p in oc['pairs']][:3])
# (1b) the control fires on a B != A
gBad = P.DenseGuide(Hd, dest, offd, psiA * np.exp(0.02 * np.random.default_rng(3).standard_normal(n)))
try:
    BF.run_branched(gA, gA2, X0, gB_ctl=gBad, control_blocks=5, **{**KW, 'K': 2}); fired = False
except BF.ControlFailure as e: fired = True
rec('control_fires_on_B_neq_A', fired)
# (2) broken control
ob = BF.run_branched(gA, gB, X0, **KW); sm = BF.summarize(ob['pairs'], NS=N, spacing=KW['spacing'], tau2_A=BF.tau_int_A(ob['A_series'], KW['burn_A']))
z = (sm['delta_site'] - D_ex) / sm['err_scatter']
rec('broken_control_sign_and_value', sm['delta_site'] > 5 * sm['err_scatter'] and abs(z) < 4,
    mc=sm['delta_site'], err=sm['err_scatter'], exact=D_ex, z_vs_exact=float(z), sd_pair=sm['sd_pair'], lag1=sm.get('lag1_corr_pairs'), tau2_A=sm.get('tau2_A_blocks'), spacing=KW['spacing'])
# (3) A chain independent of the pairs: same snapshots with and without pairs (K=3 uses 3 snapshots; compare A_series)
o1 = BF.run_branched(gA, gB, X0, **{**KW, 'K': 4}); o2 = BF.run_branched(gA, gA2, X0, gB_ctl=gA2, control_blocks=3, **{**KW, 'K': 4})
rec('A_chain_independent_of_pairs', np.array_equal(o1['A_series'], o2['A_series']) and np.array_equal(o1['A_series'], ob['A_series'][:len(o1['A_series'])]))
# (4) burn adequacy
w1 = np.array([p['windows']['w1'] for p in ob['pairs']]); w2 = np.array([p['windows']['w2'] for p in ob['pairs']]); dw = w1 - w2
rec('burn_windows_agree', abs(dw.mean()) < 4 * dw.std(ddof=1) / np.sqrt(len(dw)) + 1e-12, w1=float(w1.mean()), w2=float(w2.mean()), diff=float(dw.mean()), err=float(dw.std(ddof=1) / np.sqrt(len(dw))),
    burn_window_mean=float(np.mean([p['windows']['burn_0_20'] for p in ob['pairs']])))
RES['summary'] = sm; RES['pairs'] = ob['pairs']; RES['failures'] = [k for k, c in RES['checks'].items() if not c['passed']]
json.dump(RES, open(f'{OUT}/test_branched.json', 'w'), indent=1, default=float)
print('SUMMARY', 'ALL PASS' if not RES['failures'] else f'FAIL {RES["failures"]}')
