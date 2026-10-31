"""r3avg_bare.py: R3-avg readout (PLAN_pfaffian.md STEP BACK).  K, Vm (sign s_G) built ONCE; then
  (a) control: B final table (pfaffian_r3c/B_fit/round_1/la_table.npy) must give e = 1.2884743871942482e-05 (rel 1e-8), else exit 2;
  (c) round-3 step-1200 table (pfaffian_r3/pdb_tabs/tab_1200, sha 193312fe): bare e must reproduce 1.313343631083165e-05 (rel 1e-8), then the augmented c scan;
  (b) wait (<= WAITMAX s) for theta_bar's table pfaffian_r3avg/tabs/tab_bar/round_1/la_table.npy; bare e plus the augmented c scan.
Scan: c = 0,1,2,3,3.5,4,4.5,5,6,7 + parabola refinement (as r3c_bare.py).  Writes OUT/readout.json (resumable), readout.log.
usage: python3 r3avg_bare.py OUTDIR"""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; WAITMAX = 11700.0
os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
R3 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3'; RC = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3c'; RA = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3avg'
S1 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy'
CTRL_B = (f'{RC}/B_fit/round_1/la_table.npy', 1.2884743871942482e-05)
T1200 = (f'{R3}/pdb_tabs/tab_1200/round_1/la_table.npy', 1.313343631083165e-05, '193312fe')
TBAR = f'{RA}/tabs/tab_bar/round_1/la_table.npy'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RJ = f'{OUT}/readout.json'; res = json.load(open(RJ)) if os.path.exists(RJ) else {}
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
res.update(host=os.uname().nodename, s1=S1, s1_sha256=sha(S1), E0=BS.E0, code_sha256={f: sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in ('r3avg_bare.py', 'bond_split.py', 'delta0.py')})
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
    log(f'{name}: e = {r["dE_site"]:.9e}  it {r["iters"]} res {r["res"]:.1e}  {time.time() - t:.0f}s'); return la, r
def control(key, name, path, expect):
    la, r = bare(name, path); rel = (tabs[name]['e'] - expect) / expect
    res.setdefault('controls', {})[key] = dict(path=path, e=tabs[name]['e'], expect=expect, rel=rel, pass_=bool(abs(rel) <= 1e-8)); dump()
    log(f'control {key}: e {tabs[name]["e"]:.9e} expect {expect:.9e} rel {rel:.2e} pass {abs(rel) <= 1e-8}')
    if abs(rel) > 1e-8: res['status'] = f'CONTROL FAIL {key}'; dump(); log('CONTROL FAIL: STOP'); sys.exit(2)
    return la, r
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

def aug(name, lF, r):
    EP = r['E']; fJ, info = fJ_of(lF, EP); tabs[name]['fJ'] = info; log(f'{name} fJ {info}')
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
    log(f'{name}: c* {cst} e_aug {cache[cst]:.6e} ({tabs[name]["scan_status"]})')
# (a) control
control('B_final', 'B_final', *CTRL_B)
# (c) theta(1200)
assert sha(T1200[0]).startswith(T1200[2]), 'tab_1200 sha'
la, r = control('r3_1200', 'r3_1200', T1200[0], T1200[1]); aug('r3_1200', la, r); del la
res['part_c_done'] = True; dump()
# (b) theta_bar
t = time.time()
while not (os.path.exists(TBAR) and os.path.getsize(TBAR) > 0):
    if time.time() - T0 > WAITMAX: res['status'] = 'TBAR TABLE MISSING (timeout)'; dump(); log(res['status']); sys.exit(3)
    log('waiting for theta_bar table'); time.sleep(300)
time.sleep(30)
la, r = bare('theta_bar', TBAR); aug('theta_bar', la, r)
res['status'] = 'DONE'; res['done'] = True; dump(); log('DONE')
