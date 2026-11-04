"""m5p2_plateau.py: plateau data of the M5' rerun (PLAN_pfaffian.md "M5' rerun", plateau condition).  Separate CPU step AFTER m5p2_read.py final (m5p_read.py is not edited).
  usage: python3 m5p2_plateau.py OUTDIR [--selftest-logic]
  Reads Q/round/fit_th/round_1/progress.txt (held-out loss L_ho at steps 1200 and 1500; train loss too) and the two tables built by m5p2_tab_th.sbatch from the SAME fit:
    Q/round/tab_th/tab_s1200  (warm_theta = fit_th/round_1/theta_step1200.npy)   Q/round/tab_th/tab_s1500  (warm_theta = fit_th/round_1/theta.npy = theta at step 1500)
  and computes the exact E_FN of both (D0M.efn_vec on the sector table, as m5p_read.py 'bare').
  Registered plateau condition (lead): plateau iff (A) |L_ho(1200) - L_ho(1500)| / L_ho(1500) < 0.015 (the fit's own convention: |L(start of window) - L(end)| / L(end))
     AND (B) |e(theta_1200) - e(theta_1500)| < 0.1e-6  (e = dE_site = E_FN per site above E0, units of J1; same difference as in E_FN).
  Also reported, NOT part of the condition: a window-mean variant of (A) (mean of the 4 logged L_ho at steps 1170..1200 vs 1470..1500), L_tr, and r_5-like numbers of both thetas (needs OUT/readout.json).
  Writes OUT/plateau.json (resumable), appends to OUT/readout.log.  PLATEAU_STEPS='1200,1500' env override is for the logic test only."""
import os, sys, json, re, time, hashlib
import numpy as np
Q = os.environ.get('M5P2_Q', '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_m5p2')
FD = f'{Q}/round/fit_th'; TD = f'{Q}/round/tab_th'
S_A, S_B = (int(x) for x in os.environ.get('PLATEAU_STEPS', '1200,1500').split(','))
TOL_LHO = 0.015; TOL_E = 0.1e-6
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()

def parse_progress(path):
    d = {}
    for ln in open(path):
        m = re.search(r' step (\d+): held-out ([0-9.eE+-]+)\s+train ([0-9.eE+-]+)', ln)
        if m: d[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return d

def decide(la, lb, ea, eb, win_a=None, win_b=None):
    """la, lb: held-out loss at S_A, S_B; ea, eb: e at S_A, S_B.  Returns dict with both sub-conditions and the plateau verdict."""
    rel = abs(la - lb) / lb; de = abs(ea - eb)
    cA = bool(rel < TOL_LHO); cB = bool(de < TOL_E)
    out = dict(lho_rel_change=rel, lho_cond_lt_1p5pct=cA, e_abs_diff=de, e_cond_lt_0p1e6=cB, plateau=bool(cA and cB))
    if win_a is not None: out['lho_window_mean_rel_change'] = abs(win_a - win_b) / win_b
    return out

if '--selftest-logic' in sys.argv:           # deliberately checkable: each branch must fire on a synthetic input
    t = [(1.00e-3, 1.01e-3, 3.0e-6, 3.05e-6, True), (1.00e-3, 1.10e-3, 3.0e-6, 3.05e-6, False), (1.00e-3, 1.01e-3, 3.0e-6, 3.2e-6, False), (1.0, 2.0, 1.0, 2.0, False)]
    for la, lb, ea, eb, want in t:
        got = decide(la, lb, ea, eb)['plateau']; assert got == want, (la, lb, ea, eb, got, want)
    pr = f'/tmp/_plateau_{os.getpid()}.txt'
    open(pr, 'w').write('x PROGRESS   rbmpp step 10: held-out 6.8104e-03  train 5.4415e-03  rms_e(-11) 0.0 acc 0.5\nx PROGRESS   rbmpp step 20: held-out 7.0653e-03  train 5.4269e-03  rms_e(-11) 0.0\n')
    p = parse_progress(pr); os.remove(pr); assert p == {10: (6.8104e-03, 5.4415e-03), 20: (7.0653e-03, 5.4269e-03)}, p
    print('selftest-logic OK (4 decide cases incl. both failing sub-conditions and the parse)'); sys.exit(0)

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m):
    s = f'[{time.time() - T0:7.0f}s] plateau: {m}'; print(s, flush=True); open(f'{OUT}/readout.log', 'a').write(s + '\n')
PJ = f'{OUT}/plateau.json'; res = json.load(open(PJ)) if os.path.exists(PJ) else {}
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(PJ, 'w'), indent=1, default=float)
prog = parse_progress(f'{FD}/round_1/progress.txt'); res['progress_path'] = f'{FD}/round_1/progress.txt'
need = [S_A, S_B]; miss = [s for s in need if s not in prog]; assert not miss, f'steps {miss} not logged in progress.txt (last {max(prog) if prog else None})'
la, ta = prog[S_A]; lb, tb = prog[S_B]
wa = [prog[s][0] for s in range(S_A - 30, S_A + 1, 10) if s in prog]; wb = [prog[s][0] for s in range(S_B - 30, S_B + 1, 10) if s in prog]
res.update(steps=[S_A, S_B], L_ho=dict(zip(map(str, need), [la, lb])), L_tr=dict(zip(map(str, need), [ta, tb])), L_ho_window_means=[float(np.mean(wa)), float(np.mean(wb))], n_window=[len(wa), len(wb)],
           registered=dict(tol_lho=TOL_LHO, tol_e=TOL_E)); dump()
log(f'L_ho step {S_A} {la:.4e}, step {S_B} {lb:.4e}: rel change {abs(la - lb) / lb:.4f} (window means {np.mean(wa):.4e} vs {np.mean(wb):.4e}, {len(wa)}+{len(wb)} points)')
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import bond_split as BS
import delta0 as D0M
S1 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy'
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
s1 = np.load(S1).astype(np.float64); assert set(np.unique(s1)) <= {-1.0, 1.0}
K, Vm, D0 = BS.build(n, s1); log('built K,Vm (s_G)')
tabs = res.setdefault('tables', {})
for name, thp in (('s1200', f'{FD}/round_1/theta_step{S_A}.npy'), ('s1500', f'{FD}/round_1/theta.npy')):
    if name in tabs and 'e' in tabs[name]: continue
    p = f'{TD}/tab_{name}/round_1/la_table.npy'; la_ = np.load(p).astype(np.float64); assert la_.shape == (D,) and np.isfinite(la_).all(), p
    t = time.time(); r, _ = D0M.efn_vec(K, Vm, D0, n, la_)
    tabs[name] = dict(path=p, sha256=sha(p), theta=thp, theta_sha256=sha(thp), e=r['dE_site'], E_FN=r, iters=r['iters'], res=r['res']); dump()
    log(f'{name}: e = {r["dE_site"]:.9e}  it {r["iters"]} res {r["res"]:.1e}  {time.time() - t:.0f}s')
ea, eb = tabs['s1200']['e'], tabs['s1500']['e']
dec = decide(la, lb, ea, eb, float(np.mean(wa)), float(np.mean(wb))); dec.update(e_1200=ea, e_1500=eb, e_diff_signed=eb - ea); res['decision'] = dec
RJ = f'{OUT}/readout.json'
if os.path.exists(RJ):
    vn = json.load(open(RJ)).get('verdict_numbers', {})
    if 'e_rel' in vn:
        er = vn['e_rel']; F = 4.367093e-6; res['r5_like'] = dict(e_rel=er, r_1200=(er - ea) / (er - F), r_1500=(er - eb) / (er - F), r_tail_avg=vn.get('r_5_M5p'), e_end_tail_avg=vn.get('e_end'))
res['status'] = 'DONE'; res['done'] = True; dump()
log(f'VERDICT PLATEAU: L_ho change {dec["lho_rel_change"]:.4f} (< {TOL_LHO}: {dec["lho_cond_lt_1p5pct"]}); |e(1200) - e(1500)| = {dec["e_abs_diff"]:.3e} (< {TOL_E:.1e}: {dec["e_cond_lt_0p1e6"]}); e_1200 {ea:.6e} e_1500 {eb:.6e} -> {"PLATEAU" if dec["plateau"] else "NOT PLATEAUED"}')
