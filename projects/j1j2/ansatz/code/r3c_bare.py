"""r3c_bare.py: R3-cont readout (PLAN_pfaffian.md "R3-cont").  K, Vm (sign s_G) are built ONCE; then
  0. control: the round-3 step-1500 table (pfaffian_r3/pdb_fit/round_1/la_table.npy, sha 679fc91a...) must give e = 1.231092e-5 (rel 1e-8), else exit 2 (no other solve is run);
  1. bare e (c = 0, E_FN solve of the fitted table, bond_split.build + delta0.efn_vec as r3_readout.py) for all 10 tables (A: 150/300/450/final, B: 150..750/final);
  2. augmented c scan  e(c) = E_FN[ln a + c f_J[a]]  (c = 0,1,2,3,3.5,4,4.5,5,6,7, parabola refinement around the minimum, as r3_readout.py) for the two FINAL tables only.
Writes OUT/readout.json (resumable: solved points are kept; "done": true at the end), OUT/readout.log.  Table labels carry the global step (A: 1500 + n, B: 1200 + n).
usage: python3 r3c_bare.py OUTDIR [--control-only] [--bare-only]"""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; CONTROL_ONLY = '--control-only' in sys.argv; BARE_ONLY = '--bare-only' in sys.argv
os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
R3 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3'; RC = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3c'
S1 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy'; TGT = f'{R3}/target_T3_sG.npy'
CTRL = (f'{R3}/pdb_fit/round_1/la_table.npy', 1.231092328257688e-05)      # pfaffian_r3/readout/r3_1500/readout.json e_fit_site (full digits; the plan's 1.231092e-5 is its 7-digit rounding)
START = dict(A=1.231092328257688e-05, B=1.313344e-05)           # e at global step 1500 (r3_1500) / 1200 (r3_1200)
TABLES = []                                                     # (name, arm, steps_in_arm, global_step, path)
for arm, base, ns in (('A', 1500, (150, 300, 450)), ('B', 1200, (150, 300, 450, 600, 750))):
    for k in ns: TABLES.append((f'{arm}_{k}', arm, k, base + k, f'{RC}/{arm}_tabs/tab_{k}/round_1/la_table.npy'))
    TABLES.append((f'{arm}_final', arm, 600 if arm == 'A' else 900, base + (600 if arm == 'A' else 900), f'{RC}/{arm}_fit/round_1/la_table.npy'))
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RJ = f'{OUT}/readout.json'; res = json.load(open(RJ)) if os.path.exists(RJ) else {}
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
res.update(host=os.uname().nodename, s1=S1, s1_sha256=sha(S1), E0=BS.E0, code_sha256={f: sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in ('r3c_bare.py', 'bond_split.py', 'delta0.py')})
fitsha = {}
for arm in 'AB':
    sp = f'{RC}/{arm}_fit/spec_r1.json'
    if os.path.exists(sp): fitsha[arm] = json.load(open(sp)).get('code_sha256', {}).get('cx_r1fit.py')
res['fit_cx_r1fit_sha256'] = fitsha
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
s1 = np.load(S1).astype(np.float64); assert set(np.unique(s1)) <= {-1.0, 1.0}
K, Vm, D0 = BS.build(n, s1); log('built K,Vm (s_G) once')
def load(p):
    la = np.load(p).astype(np.float64); assert la.shape == (D,) and np.isfinite(la).all(), p; return la
tabs = res.setdefault('tables', {})
def bare(name, path):
    if name in tabs and 'E_FN' in tabs[name]: return load(path), tabs[name]['E_FN']
    la = load(path); t = time.time(); r, _ = D0M.efn_vec(K, Vm, D0, n, la)
    tabs.setdefault(name, {}).update(path=path, sha256=sha(path), e=r['dE_site'], E_FN=r); dump()
    log(f'{name}: e = {r["dE_site"]:.6e}  it {r["iters"]} res {r["res"]:.1e}  {time.time() - t:.0f}s'); return la, r
# 0. control
la, r = bare('control_r3_1500', CTRL[0]); rel = (tabs['control_r3_1500']['e'] - CTRL[1]) / CTRL[1]
res['control'] = dict(path=CTRL[0], e=tabs['control_r3_1500']['e'], expect=CTRL[1], rel=rel, pass_=bool(abs(rel) <= 1e-8)); dump()
log(f'control: e {res["control"]["e"]:.9e} expect {CTRL[1]:.6e} rel {rel:.2e} pass {res["control"]["pass_"]}')
if not res['control']['pass_']: res['status'] = 'CONTROL FAIL'; dump(); log('CONTROL FAIL: STOP'); sys.exit(2)
if CONTROL_ONLY: res['status'] = 'CONTROL ONLY'; dump(); sys.exit(0)
# 1. bare e at all tables
missing = [t[0] for t in TABLES if not os.path.exists(t[4])]
if missing: res['status'] = f'MISSING TABLES {missing}'; dump(); log(res['status']); sys.exit(3)
for name, arm, k, g, path in TABLES:
    bare(name, path); tabs[name].update(arm=arm, steps_in_arm=k, global_step=g)
dump()
for arm in 'AB':
    rows = [(t[3], tabs[t[0]]['e']) for t in TABLES if t[1] == arm]
    log(f'ARM {arm} (start {START[arm]:.6e}): ' + '  '.join(f'g{g}: {e:.6e} ({(e - START[arm]) * 1e6:+.3f}e-6)' for g, e in rows))
res['bare_done'] = True; dump()
if BARE_ONLY: res['status'] = 'BARE ONLY'; res['done'] = True; dump(); sys.exit(0)
# 2. augmented scan, final tables only (f_J as r3_readout.py: E' = E_FN[fit], s1 blocked bonds)
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
def fJ_of(lF, EP):
    fJ = np.empty(D); nbad = 0; dmin = np.inf
    for i0 in range(0, D, 65536):
        i1 = min(D, i0 + 65536); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        ss = s1[i0:i1, None] * s1[tt]; blk = valid & (ss > 0); kpt = valid & (ss < 0)
        ratio = np.exp(np.clip((lF - lF.max())[tt] - (lF - lF.max())[i0:i1, None], -700, 700))
        VA = (blk * BS.HW[None, :] * ratio).sum(1); SK = (kpt * BS.HW[None, :] * ratio).sum(1)
        den = 27.0 - 0.5 * (valid * BS.JW[None, :]).sum(1) + VA - EP; dmin = min(dmin, float(den.min())); nbad += int((den <= 0).sum() + (SK <= 0).sum())
        with np.errstate(divide='ignore', invalid='ignore'): fJ[i0:i1] = np.log(SK) - np.log(den)
    bad = np.nonzero(~np.isfinite(fJ))[0]; info = dict(den_min=dmin, n_bad=nbad, bad_rows=[int(i) for i in bad]); assert len(bad) <= 3 and dmin > 0
    fJ[bad] = 0.0; return fJ, info
for arm in 'AB':
    name = f'{arm}_final'; lF, r = bare(name, dict((t[0], t[4]) for t in TABLES)[name]); EP = r['E']
    fJ, info = fJ_of(lF, EP); tabs[name]['fJ'] = info; log(f'{name} fJ {info}')
    cache = {float(k): v for k, v in tabs[name].get('scan', {}).items()}; cache[0.0] = tabs[name]['e']; tabs[name]['scan'] = {str(k): v for k, v in cache.items()}
    def solve(c):
        c = round(float(c), 6)
        if c in cache: return cache[c]
        rr, _ = D0M.efn_vec(K, Vm, D0, n, lF + c * fJ); cache[c] = rr['dE_site']; tabs[name]['scan'][str(c)] = rr['dE_site']; dump()
        log(f'{name} c={c:.3f} e {rr["dE_site"]:.6e} it {rr["iters"]} res {rr["res"]:.1e}'); return rr['dE_site']
    for c in (1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0): solve(c)
    status = None
    for it in range(8):
        b = min(cache, key=cache.get)
        if b >= 7.0: status = 'MIN AT UPPER EDGE'; break
        if b <= 0.0: status = 'MIN AT c=0'; break
        cs = sorted(cache); i = cs.index(b); trio = cs[i - 1:i + 2]
        p = np.polyfit(trio, [cache[c] for c in trio], 2)
        if p[0] <= 0: status = 'NOT CONVEX'; break
        cv = round(float(np.clip(-p[1] / (2 * p[0]), 0, 7)) / 0.1) * 0.1; log(f'{name} iter {it}: trio {trio} vertex-> {cv:.2f}')
        if round(cv, 6) in cache: break
        solve(cv)
    cst = min(cache, key=cache.get); tabs[name].update(cstar=cst, e_aug_min_site=cache[cst], scan_status=status or 'DONE'); dump()
    log(f'{name}: c* {cst} e_aug {cache[cst]:.6e}'); del fJ, lF
res['status'] = 'DONE'; res['done'] = True; dump(); log('DONE')
