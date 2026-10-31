"""r3c_gate.py: mechanical reading of pfaffian_r3c/readout/readout.json against the pre-registered definitions of PLAN_pfaffian.md "R3-cont" (the lead decides; this only applies the stated thresholds).
usage: python3 r3c_gate.py [READOUT_JSON] [RUNDIR]    prints the table, the verdict line, the loss trend (A: L_tr vs L_ho from A_fit/round_1/cx_t1s.json)"""
import sys, json, numpy as np
RC = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3c'
RJ = sys.argv[1] if len(sys.argv) > 1 else f'{RC}/readout/readout.json'; Q = sys.argv[2] if len(sys.argv) > 2 else RC
d = json.load(open(RJ)); assert d.get('done'), 'readout not done'; T = d['tables']
assert d['control']['pass_'], 'control failed'
S = dict(A=1.231092328257688e-05, B=1.313344e-05); order = dict(A=['A_150', 'A_300', 'A_450', 'A_final'], B=['B_150', 'B_300', 'B_450', 'B_600', 'B_750', 'B_final'])
ser = {a: [S[a]] + [T[k]['e'] for k in order[a]] for a in 'AB'}; gl = {a: [T[k]['global_step'] for k in order[a]] for a in 'AB'}
for a in 'AB':
    print(f'ARM {a} start {S[a]:.6e}: ' + '  '.join(f'g{g} {e:.6e}' for g, e in zip(gl[a], ser[a][1:])))
def rise(s): return max(np.diff(s)) > 0.1e-6
A, B = ser['A'], ser['B']
Afalls = (A[-1] - A[0] <= -0.2e-6) and not rise(A); Aflat = abs(A[-1] - A[0]) < 0.2e-6; Arise = rise(A)
Bflat = abs(B[-1] - B[0]) < 0.2e-6; Bslow = (B[0] - B[-1]) <= 0.26e-6 * 3      # drop <= 0.26e-6 per 300 steps over the 900 steps
Brise = rise(B); Bfalls = (B[-1] - B[0] <= -0.2e-6) and not Brise
# equal global step 2100: A final (g2100) vs B final (g2100)
comparable = abs(A[-1] - B[-1]) <= 0.1e-6
print(f'A: falls {Afalls} flat {Aflat} rise>0.1e-6 {Arise} (end-start {(A[-1]-A[0])*1e6:+.3f}e-6) | B: falls {Bfalls} flat {Bflat} slow(<=0.26e-6/300) {Bslow} rise {Brise} (end-start {(B[-1]-B[0])*1e6:+.3f}e-6) | |A-B| at g2100 {abs(A[-1]-B[-1])*1e6:.3f}e-6 comparable {comparable}')
if Arise and Bflat: v = 'A RISES and B flat: the drop was a walk; round 3 counts at e(1200) = 1.313e-5'
elif Afalls and Bfalls and comparable: v = 'BOTH FALL COMPARABLY: just more training; round 3 counts at the arm that meets the E_FN convergence rule'
elif Afalls and (Bflat or Bslow): v = 'A FALLS, B flat/slow: H_u supported; next step = step/stop design on the board first'
else: v = 'UNCLASSIFIED by the pre-registered rules: report the table to the lead'
if Brise or (Arise and not Bflat): v += '  [KILL RULE: a rise > 0.1e-6 at a checkpoint]'
print('VERDICT_R3C ' + v)
for a in 'AB':
    for k in (f'{a}_final',):
        t = T[k]
        if 'scan' in t: print(f'{k}: c* {t["cstar"]} e_aug_min {t["e_aug_min_site"]:.6e}  scan status {t["scan_status"]}')
cj = f'{Q}/A_fit/round_1/cx_t1s.json'
try:
    cv = json.load(open(cj))['fit']['curve']; st = np.array([q['step'] for q in cv]); ho = np.array([q['L_ho'] for q in cv]); tr = np.array([q['L_tr'] for q in cv])
    p_ho = np.polyfit(st, ho, 1)[0] * 100; p_tr = np.polyfit(st, tr, 1)[0] * 100
    h = len(st) // 2
    print(f'A losses: L_tr {tr[0]:.4e} -> mean first half {tr[:h].mean():.4e}, second half {tr[h:].mean():.4e}, slope {p_tr:+.2e}/100 steps | L_ho {ho[0]:.4e} -> {ho[:h].mean():.4e}, {ho[h:].mean():.4e}, slope {p_ho:+.2e}/100 steps')
    print('OVERFIT FLAG (training falls, held-out rises, second half vs first half): ' + str(bool(tr[h:].mean() < tr[:h].mean() and ho[h:].mean() > ho[:h].mean())))
except Exception as ex: print('loss trend unavailable', ex)
