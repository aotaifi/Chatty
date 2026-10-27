"""c1_mech.py: Test C mechanism readout (PLAN_pfaffian.md "Test C"): pair cost of bond_split.cost_pass on the fixed sets
  S_x    = violating pairs with class r < r*/3 in the REAL arm's copy (ln(r/r*) < -ln 3) AND t(x) t(y) = -1, t = sgn(Re A) * s2 (t_real.npy of the real arm),
  S_rest = all other violating pairs.
cost_pass is imported from bond_split.py; since it returns only aggregates, the SAME source text is exec'd with one callback line inserted after
`tot += c.sum(); ...` (per-pair c, rep indices xs/ys, lrs = ln r*).  Checks: patched total == imported cost_pass total (rows slice), and C(S_x)+C(S_rest) == total.
usage: python c1_mech.py OUT.json --sets REAL_DIR0 [--real DIR ...] [--cplx DIR ...] [--target target_la.npy] [--rows i0 i1]
  DIR holds la_table.npy (ln|A| per orbit rep); REAL_DIR0 also t_real.npy and defines the sets (its la_table = "the real arm's copy").
  --real defaults to REAL_DIR0.  cut_X = 1 - mean_cplx C(S_x) / mean_real C(S_x); cut_R likewise on S_rest (means over the dirs given).
Output: per arm C(S_x), C(S_rest), total (all in units of cost_pass's 'total', and per site = /36)."""
import os, sys, json, time, inspect, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bond_split as BS

args = sys.argv[1:]; OUT = args[0]; opt = {'--real': [], '--cplx': [], '--sets': [], '--target': [], '--rows': []}; k = None
for a in args[1:]:
    if a in opt: k = a
    else: opt[k].append(a)
SETS = opt['--sets'][0]; REAL = opt['--real'] or [SETS]; CPLX = opt['--cplx']
TARGET = opt['--target'][0] if opt['--target'] else '/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/o2b_17431886/round_1/target_la.npy'
ROWS = tuple(int(x) for x in opt['--rows']) if opt['--rows'] else None
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s2 = np.load(BS.S2).astype(np.float64)
la_phi = np.load(TARGET); la_set = np.load(f'{SETS}/la_table.npy'); t = np.load(f'{SETS}/t_real.npy').astype(np.int8)
assert t.shape == n.shape == la_set.shape and set(np.unique(t)) <= {-1, 1}, (t.shape, np.unique(t))

src = inspect.getsource(BS.cost_pass)
anchor = "tot += c.sum(); npair += c.size"
assert src.count(anchor) == 1
src = src.replace('def cost_pass(', 'def cost_pass_cb(').replace(anchor, anchor + "\n        _CB(x, y, xs, ys, c, lrs)")
ns = BS.__dict__; ns['_CB'] = None; exec(src, ns); cost_cb = ns['cost_pass_cb']
ACC = np.zeros(2); NPR = np.zeros(2, np.int64)


def cb(x, y, xs, ys, c, lrs):
    q = np.clip(la_set[xs] - la_set[ys], -300, 300) - lrs
    m = (q < -np.log(3.0)) & (t[xs] * t[ys] == -1)
    ACC[0] += c[m].sum(); ACC[1] += c[~m].sum(); NPR[0] += m.sum(); NPR[1] += (~m).sum()


def run(d, rows=ROWS):
    ACC[:] = 0; NPR[:] = 0; ns['_CB'] = cb
    r = cost_cb(n, s2, la_phi, np.load(f'{d}/la_table.npy'), rows=rows)
    return dict(dir=d, C_Sx=float(ACC[0]), C_rest=float(ACC[1]), total=r['total'], n_Sx=int(NPR[0]), n_rest=int(NPR[1]),
                C_Sx_site=float(ACC[0]) / 36, C_rest_site=float(ACC[1]) / 36, sum_check=float(ACC.sum() / r['total'] - 1))


res = dict(sets=SETS, target=TARGET, rows=ROWS, real=[], cplx=[])
# self-check against the imported, unpatched cost_pass on a slice
r0 = BS.cost_pass(n, s2, la_phi, la_set, rows=(0, 131072)); r1 = run(SETS, (0, 131072))
res['check_patched_vs_imported_total_rel'] = r1['total'] / r0['total'] - 1; res['check_sum_rel'] = r1['sum_check']
print('check', res['check_patched_vs_imported_total_rel'], res['check_sum_rel'], flush=True)
assert abs(res['check_patched_vs_imported_total_rel']) < 1e-12 and abs(res['check_sum_rel']) < 1e-9
for d in REAL: res['real'].append(run(d)); print(res['real'][-1], flush=True)
for d in CPLX: res['cplx'].append(run(d)); print(res['cplx'][-1], flush=True)
if CPLX:
    mr = lambda L, k: float(np.mean([q[k] for q in L]))
    res['cut_X'] = 1 - mr(res['cplx'], 'C_Sx') / mr(res['real'], 'C_Sx'); res['cut_R'] = 1 - mr(res['cplx'], 'C_rest') / mr(res['real'], 'C_rest')
    print('cut_X', res['cut_X'], 'cut_R', res['cut_R'])
json.dump(res, open(OUT, 'w'), indent=1)
