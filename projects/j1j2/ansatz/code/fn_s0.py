"""fn_s0.py: exact-sign (s0 = sign of the exact ground state psi0 per orbit rep) additions to the s_G floor step (PLAN_pfaffian.md "s_G floor", Additions A-D).
Table: /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz (reps sorted uint64 = norb.npy order, 'amp' float64 per rep = per-configuration psi0, E0 key);
s0 = -1 where amp < 0 else +1 (no zero amplitudes), |psi0| = |amp|, la0 = log|amp|.  Convention of bond_split.build: blocked/violating = s_x s_y > 0, kept < 0.
 (A) E_FN[|psi0|, s0] per site must be < 1e-10.  (B) blocked-bond counts, J1/J2.  (C) theta_4 amplitude (pfaffian_r4/pdb_tabs/tab_t4, sha 8e358ecb) on s0: bare, f_J with s0 walls
 (E' = E_FN[theta_4, s0]), augmented c scan 0..7 + refinement (same code as r4_bare.py aug).  (D) 10 plain-iteration solves from |psi0| at s0 (fn_floor.py update rule).
usage: python3 fn_s0.py OUTDIR   -> OUTDIR/floor.json (done: true), floor.log"""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
TABLE = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz'
TH4 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4/pdb_tabs/tab_t4/round_1/la_table.npy'
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RJ = f'{OUT}/floor.json'; res = json.load(open(RJ)) if os.path.exists(RJ) else {}
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ + '.tmp', 'w'), indent=1, default=float); os.replace(RJ + '.tmp', RJ)
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n)
z = np.load(TABLE); amp = z['amp'].astype(np.float64); E0t = float(z['E0']); reps = z['reps']; del z
assert amp.shape == (D,) and (amp != 0).all() and abs(E0t - BS.E0) < 1e-12
s0 = np.where(amp < 0, -1.0, 1.0); la0 = np.log(np.abs(amp)); np.save(f'{OUT}/s0.npy', s0.astype(np.int8))
res.update(table=TABLE, table_sha256=sha(TABLE), E0_table=E0t, E0_bs=BS.E0, n_neg=int((amp < 0).sum()), host=os.uname().nodename, th4=TH4, th4_sha256=sha(TH4),
           code_sha256={f: sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in ('fn_s0.py', 'bond_split.py', 'delta0.py')})
assert res['th4_sha256'].startswith('8e358ecb1d4c')
K, Vm, D0 = BS.build(n, s0); log('built K, Vm (s0)')
def norm_la(la):
    a = np.exp(la - la.max()); return la - la.max() - 0.5 * np.log(np.sum(n * a * a))
# (A)
if 'A' not in res:
    r, v = D0M.efn_vec(K, Vm, D0, n, la0); res['A'] = dict(e_site=r['dE_site'], E=r['E'], iters=r['iters'], res=r['res'], pass_=bool(abs(r['dE_site']) < 1e-10)); dump()
log(f'(A) e = {res["A"]["e_site"]:.3e} it {res["A"]["iters"]} res {res["A"]["res"]:.1e} pass {res["A"]["pass_"]}')
if not res['A']['pass_']: res['status'] = 'CONTROL A FAIL'; dump(); log('CONTROL FAIL: STOP'); sys.exit(2)
# (B)
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
if 'B' not in res:
    acc = np.zeros((2, 6)); R = 65536                       # per class: [valid entries, blocked entries, n-weighted valid, n-weighted blocked, |H|a a valid, |H|a a blocked]
    for i0 in range(0, D, R):
        i1 = min(D, i0 + R); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        blk = valid & (s0[i0:i1, None] * s0[tt] > 0); w = (BS.HW[None, :] * np.abs(amp[i0:i1, None]) * np.abs(amp[tt]) * n[i0:i1, None])
        nx = n[i0:i1, None] * np.ones_like(w)
        for c, sl in ((0, slice(0, 72)), (1, slice(72, 144))):
            v_, b_ = valid[:, sl], blk[:, sl]
            acc[c] += [v_.sum(), b_.sum(), (nx[:, sl] * v_).sum(), (nx[:, sl] * b_).sum(), (w[:, sl] * v_).sum(), (w[:, sl] * b_).sum()]
    B = {}
    for c, nm in ((0, 'J1'), (1, 'J2')):
        B[nm] = dict(valid_rep_entries=acc[c, 0], blocked_rep_entries=acc[c, 1], share_rep_entries=acc[c, 1] / acc[c, 0], valid_n_weighted=acc[c, 2], blocked_n_weighted=acc[c, 3],
                     share_n_weighted=acc[c, 3] / acc[c, 2], weighted_valid_Haa=acc[c, 4], weighted_blocked_Haa=acc[c, 5], share_Haa=acc[c, 5] / acc[c, 4])
    a_ = acc.sum(0); B['all'] = dict(share_n_weighted=a_[3] / a_[2], share_rep_entries=a_[1] / a_[0], share_Haa=a_[5] / a_[4], blocked_n_weighted=a_[3], valid_n_weighted=a_[2])
    res['B'] = B; dump()
log(f'(B) {json.dumps(res["B"], default=float)}')
# (D) before (C): cheap
def step(la):
    r, v = D0M.efn_vec(K, Vm, D0, n, la); u = np.abs(v); u = u / np.linalg.norm(u); return r, norm_la(np.log(np.maximum(u, 1e-300) / np.sqrt(n)))
if 'D' not in res:
    la = la0.copy(); h = []; dl = []
    for k in range(11):
        r, la = step(la); h.append(r['dE_site']); dl.append(float(np.abs(la - norm_la(la0)).max())); log(f'(D) solve {k}: e = {r["dE_site"]:.3e} it {r["iters"]} res {r["res"]:.1e} max|dla| {dl[-1]:.2e}')
    res['D'] = dict(hist=h, max_abs_dla=dl); dump()
# (C)
tabs = res.setdefault('C', {}); la4 = np.load(TH4).astype(np.float64); assert la4.shape == (D,) and np.isfinite(la4).all()
def fJ_of(lF, EP):                                                                          # r4_bare.py fJ_of with s = s0
    fJ = np.empty(D); nbad = 0; dmin = np.inf
    for i0 in range(0, D, 65536):
        i1 = min(D, i0 + 65536); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        ss = s0[i0:i1, None] * s0[tt]; blk = valid & (ss > 0); kpt = valid & (ss < 0)
        ratio = np.exp(np.clip((lF - lF.max())[tt] - (lF - lF.max())[i0:i1, None], -700, 700))
        VA = (blk * BS.HW[None, :] * ratio).sum(1); SK = (kpt * BS.HW[None, :] * ratio).sum(1)
        den = 27.0 - 0.5 * (valid * BS.JW[None, :]).sum(1) + VA - EP; dmin = min(dmin, float(den.min())); nbad += int((den <= 0).sum() + (SK <= 0).sum())
        with np.errstate(divide='ignore', invalid='ignore'): fJ[i0:i1] = np.log(SK) - np.log(den)
    bad = np.nonzero(~np.isfinite(fJ))[0]; info = dict(den_min=dmin, n_bad=nbad, n_nonfinite=len(bad), bad_rows=[int(i) for i in bad[:20]])
    fJ[bad] = 0.0; return fJ, info
if 'e' not in tabs:
    r, _ = D0M.efn_vec(K, Vm, D0, n, la4); tabs.update(e=r['dE_site'], E_FN=r); dump()
log(f'(C) bare e(theta_4, s0) = {tabs["e"]:.6e}')
fJ, info = fJ_of(la4, tabs['E_FN']['E']); tabs['fJ'] = info; log(f'(C) fJ {info}')
cache = {float(k): v for k, v in tabs.get('scan', {}).items()}; cache[0.0] = tabs['e']; tabs['scan'] = {str(k): v for k, v in cache.items()}
def solve(c):
    c = round(float(c), 6)
    if c in cache: return cache[c]
    rr, _ = D0M.efn_vec(K, Vm, D0, n, la4 + c * fJ); cache[c] = rr['dE_site']; tabs['scan'][str(c)] = rr['dE_site']; dump()
    log(f'(C) c={c:.3f} e {rr["dE_site"]:.6e} it {rr["iters"]} res {rr["res"]:.1e}'); return rr['dE_site']
for c in (1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0): solve(c)
status = None
for it in range(8):
    b = min(cache, key=cache.get)
    if b >= 7.0: status = 'MIN AT UPPER EDGE'; break
    if b <= 0.0: status = 'MIN AT c=0'; break
    cs = sorted(cache); i = cs.index(b); trio = cs[i - 1:i + 2]; p = np.polyfit(trio, [cache[c] for c in trio], 2)
    if p[0] <= 0: status = 'NOT CONVEX'; break
    cv = round(float(np.clip(-p[1] / (2 * p[0]), 0, 7)) / 0.1) * 0.1
    if round(cv, 6) in cache: break
    solve(cv)
cst = min(cache, key=cache.get); tabs.update(cstar=cst, e_aug_min_site=cache[cst], scan_status=status or 'DONE')
res['done'] = True; dump(); log(f'(C) c* {cst} e_aug {cache[cst]:.6e} ({tabs["scan_status"]}) DONE')
