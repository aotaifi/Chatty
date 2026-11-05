"""m5p2x_plateau.py: plateau data of EXTENSION SEGMENT 1 of the M5' rerun (PLAN_pfaffian.md "Extension segment 1"); separate CPU step after m5p2x_read.py.  usage: m5p2x_plateau.py OUTDIR [--selftest-logic]
  Global steps = round-1 steps (1..1500) then ext1 steps + 1500.  ext1 progress.txt logs LOCAL steps (1..600): global 1800 = local 300, 2100 = local 600.
  E_FN of theta(1800) = ext1/tab_th/tab_s1800 (warm theta_step300.npy) and of theta(2100) = the fit's own final-theta table ext1/fit_th/round_1/la_table.npy (bit-identical to the separate table pass of the same theta in the 1500 run: max|diff| 0.0,
  round/fit_th/round_1/la_table.npy vs round/tab_th/tab_s1500; so no third table pass).  Exact solves D0M.efn_vec as m5p_read.py 'bare'.
  Plateau (as amended by the lead 2026-11-05): (B) |e(1800) - e(2100)| < 0.1e-6 AND (A) L_ho window means change < 1.5%.  The window of (A) is ambiguous in the PLAN text: the 1500 verdict (0.12%) was the 4-POINT window (steps 1170..1200 vs 1470..1500;
  the 16-point 1050..1200 vs 1350..1500 gives 2.9%).  BOTH are computed here: A4 = steps 1770..1800 vs 2070..2100 (local 270..300 vs 570..600), A16 = 1650..1800 vs 1950..2100 (local 150..300 vs 450..600).  Single-step L_ho (1800 vs 2100) and the
  E_FN trend 1200, 1500, 1800, 2100 (1200, 1500 from the round's plateau.json) are reported, not gating.  Writes OUT/plateau.json, appends to OUT/readout.log."""
import os, sys, json, re, time, hashlib
import numpy as np
Q = os.environ.get('M5P2X_Q', '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_m5p2'); E = f'{Q}/ext1'; FD = f'{E}/fit_th'; TD = f'{E}/tab_th'
L_A, L_B = 300, 600; OFF = 1500; TOL_LHO = 0.015; TOL_E = 0.1e-6
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
def parse_progress(path):
    d = {}
    for ln in open(path):
        m = re.search(r' step (\d+): held-out ([0-9.eE+-]+)\s+train ([0-9.eE+-]+)', ln)
        if m: d[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return d
def wmean(p, a, b, k=0):
    xs = [p[s][k] for s in range(a, b + 1, 10)]; assert len(xs) == (b - a) // 10 + 1 and len(xs) > 0, (a, b, len(xs)); return float(np.mean(xs))
def decide(la, lb, ea, eb, a4a, a4b, a16a, a16b):
    rel1 = abs(la - lb) / lb; r4 = abs(a4a - a4b) / a4b; r16 = abs(a16a - a16b) / a16b; de = abs(ea - eb)
    cB = bool(de < TOL_E); c4 = bool(r4 < TOL_LHO); c16 = bool(r16 < TOL_LHO)
    return dict(lho_single_rel_change=rel1, lho_win4_rel_change=r4, lho_win16_rel_change=r16, cond_A4=c4, cond_A16=c16, e_abs_diff=de, cond_B=cB, plateau_A4_B=bool(c4 and cB), plateau_A16_B=bool(c16 and cB))
if '--selftest-logic' in sys.argv:
    cases = [((1, 1.01, 3.0e-6, 3.05e-6, 1, 1.001, 1, 1.002), (True, True, True)),    # B ok, A4 ok, A16 ok -> both plateau
             ((1, 1.01, 3.0e-6, 3.05e-6, 1, 1.001, 1, 1.05), (True, True, False)),   # A16 fails only
             ((1, 1.01, 3.0e-6, 3.2e-6, 1, 1.001, 1, 1.002), (False, False, False)), # B fails
             ((1, 1.01, 3.0e-6, 3.05e-6, 1, 1.05, 1, 1.002), (True, False, True))]   # A4 fails only
    for a, want in cases:
        x = decide(*a); got = (x['cond_B'], x['plateau_A4_B'], x['plateau_A16_B']); assert got == want, (a, x, got, want)
    pr = f'/tmp/_plateau_x_{os.getpid()}.txt'; open(pr, 'w').write('x PROGRESS   rbmpp step 10: held-out 6.8104e-03  train 5.4415e-03  rms_e(-11) 0.0 acc 0.5\n'); p = parse_progress(pr); os.remove(pr); assert p == {10: (6.8104e-03, 5.4415e-03)}
    q = {s: (float(s), 1.0) for s in range(10, 101, 10)}; assert abs(wmean(q, 10, 40) - 25.0) < 1e-12 and abs(wmean(q, 70, 100) - 85.0) < 1e-12
    print('selftest-logic OK'); sys.exit(0)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m):
    s = f'[{time.time() - T0:7.0f}s] plateau: {m}'; print(s, flush=True); open(f'{OUT}/readout.log', 'a').write(s + '\n')
PJ = f'{OUT}/plateau.json'; res = json.load(open(PJ)) if os.path.exists(PJ) else {}
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(PJ, 'w'), indent=1, default=float)
prog = parse_progress(f'{FD}/round_1/progress.txt'); res['progress_path'] = f'{FD}/round_1/progress.txt'
miss = [s for s in (L_A, L_B) if s not in prog]; assert not miss, f'local steps {miss} not logged (last {max(prog) if prog else None})'
la, ta = prog[L_A]; lb, tb = prog[L_B]
a4a, a4b = wmean(prog, L_A - 30, L_A), wmean(prog, L_B - 30, L_B); a16a, a16b = wmean(prog, L_A - 150, L_A), wmean(prog, L_B - 150, L_B)
res.update(global_steps=[L_A + OFF, L_B + OFF], local_steps=[L_A, L_B], L_ho_single=[la, lb], L_tr_single=[ta, tb], L_ho_win4_means=[a4a, a4b], L_ho_win16_means=[a16a, a16b], registered=dict(tol_lho=TOL_LHO, tol_e=TOL_E)); dump()
log(f'L_ho single global 1800 {la:.4e}, 2100 {lb:.4e}; win4 means {a4a:.4e} vs {a4b:.4e} (rel {abs(a4a - a4b) / a4b:.4f}); win16 means {a16a:.4e} vs {a16b:.4e} (rel {abs(a16a - a16b) / a16b:.4f})')
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import bond_split as BS
import delta0 as D0M
S1 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy'
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
s1 = np.load(S1).astype(np.float64); assert set(np.unique(s1)) <= {-1.0, 1.0}
K, Vm, D0 = BS.build(n, s1); log('built K,Vm (s_G)')
tabs = res.setdefault('tables', {})
for name, p, thp in (('s1800', f'{TD}/tab_s1800/round_1/la_table.npy', f'{FD}/round_1/theta_step300.npy'), ('s2100', f'{FD}/round_1/la_table.npy', f'{FD}/round_1/theta.npy')):
    if name in tabs and 'e' in tabs[name]: continue
    la_ = np.load(p).astype(np.float64); assert la_.shape == (D,) and np.isfinite(la_).all(), p
    t = time.time(); r, _ = D0M.efn_vec(K, Vm, D0, n, la_)
    tabs[name] = dict(path=p, sha256=sha(p), theta=thp, theta_sha256=sha(thp), e=r['dE_site'], E_FN=r, iters=r['iters'], res=r['res']); dump()
    log(f'{name}: e = {r["dE_site"]:.9e}  it {r["iters"]} res {r["res"]:.1e}  {time.time() - t:.0f}s')
ea, eb = tabs['s1800']['e'], tabs['s2100']['e']
dec = decide(la, lb, ea, eb, a4a, a4b, a16a, a16b); dec.update(e_1800=ea, e_2100=eb, e_diff_signed=eb - ea); res['decision'] = dec
PR = f'{Q}/readout_round/plateau.json'
if os.path.exists(PR):
    pj = json.load(open(PR)); e12, e15 = pj['tables']['s1200']['e'], pj['tables']['s1500']['e']
    res['trend'] = dict(e_1200=e12, e_1500=e15, e_1800=ea, e_2100=eb, per_100_steps=[(e15 - e12) / 3, (ea - e15) / 3, (eb - ea) / 3], note='difference per 100 steps over 1200-1500, 1500-1800, 1800-2100')
RJ = f'{OUT}/readout.json'
if os.path.exists(RJ):
    vn = json.load(open(RJ)).get('verdict_numbers', {})
    if 'e_rel' in vn:
        er = vn['e_rel']; F = 4.367093e-6; res['r5_like'] = dict(e_rel=er, r_1800=(er - ea) / (er - F), r_2100=(er - eb) / (er - F), r_tail_avg=vn.get('r_5_M5p'), e_end_tail_avg=vn.get('e_end'))
res['status'] = 'DONE'; res['done'] = True; dump()
log(f'VERDICT PLATEAU ext1: |e(1800) - e(2100)| = {dec["e_abs_diff"]:.3e} (< {TOL_E:.1e}: {dec["cond_B"]}); A4 {dec["lho_win4_rel_change"]:.4f} ({dec["cond_A4"]}), A16 {dec["lho_win16_rel_change"]:.4f} ({dec["cond_A16"]}); e_1800 {ea:.6e} e_2100 {eb:.6e} -> '
    f'{"PLATEAU" if dec["plateau_A4_B"] else "NOT PLATEAUED"} (A4 and B); with A16 literal: {"PLATEAU" if dec["plateau_A16_B"] else "NOT PLATEAUED"}')
