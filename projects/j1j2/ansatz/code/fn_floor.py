"""fn_floor.py: self-consistent fixed-node floor of one sign (PLAN_pfaffian.md "s_G floor"; pre-registered 2026-11-02).
Floor(s) = lim_k E_FN[phi_k, s], phi_{k+1} = ground state of H_FN[phi_k, s] (plain iteration).
Method = the records' (experiments/compress_6x6/cx_veto.py fp loop; cx_s2ceil.py ceiling()):
  start  la_P = lP0 of /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/sym_tables.npz  (the symmetrised ViT ratios, NOT |psi0|),
  hist[0] = E_FN[la_P, s]; hist[k] = E_FN[phi_{k-1}, s], k = 1..40 (41 solves; "solve 40" = hist[40], as cx_veto fp[40], cx_s2ceil hist[40]),
  update la <- log(max(u, 1e-300)/sqrt(n)), u = |Perron vector| normalised (sec.fn_solve), amplitude floor 1e-15 on the normalised guide a (as sec.fn_solve and bond_split.guide),
  Aitken on hist[-3:]: c - (c-b)^2/(c-2b+a), last ratio (c-b)/(b-a).
Differences from the old code (numpy LOBPCG of delta0.efn_vec instead of jax Sector.fn_solve): same operator, start vector, preconditioner; tol here 1e-8 absolute residual
(old: 1e-10 relative to |theta| = 1.8e-9 absolute), maxit 150.
usage: python3 fn_floor.py OUTDIR SIGN.npy [--solves 40]
Writes OUTDIR/floor.json (resumable: la_k.npy of the last solve + history), floor.log."""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; SIGN = sys.argv[2]; NSOLVE = int(sys.argv[sys.argv.index('--solves') + 1]) if '--solves' in sys.argv else 40
os.makedirs(OUT, exist_ok=True); T0 = time.time()
SYM = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/sym_tables.npz'
TOL = 1e-8; MAXIT = 150
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RJ = f'{OUT}/floor.json'
res = json.load(open(RJ)) if os.path.exists(RJ) else {}
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ + '.tmp', 'w'), indent=1, default=float); os.replace(RJ + '.tmp', RJ)
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
s = np.load(SIGN).astype(np.float64); assert s.shape == (D,) and set(np.unique(s)) <= {-1.0, 1.0}
res.update(sign=SIGN, sign_sha256=sha(SIGN), host=os.uname().nodename, E0=BS.E0, nsolve=NSOLVE, tol=TOL, maxit=MAXIT,
           code_sha256={f: sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in ('fn_floor.py', 'bond_split.py', 'delta0.py')})
K, Vm, D0 = BS.build(n, s); log('built K, Vm once')
def norm_la(la):
    a = np.exp(la - la.max()); return la - la.max() - 0.5 * np.log(np.sum(n * a * a))
def step(la):
    r, v = D0M.efn_vec(K, Vm, D0, n, la, tol=TOL, maxit=MAXIT)
    u = np.abs(v); u = u / np.linalg.norm(u)
    return r, norm_la(np.log(np.maximum(u, 1e-300) / np.sqrt(n)))
hist = res.setdefault('hist', []); LK = f'{OUT}/la_last.npy'
if hist and os.path.exists(LK): la = np.load(LK); log(f'resume after {len(hist)} solves')
else:
    z = np.load(SYM); la = z['lP'].astype(np.float64); del z; assert la.shape == (D,); res['hist'] = hist = []; res['info'] = []
for k in range(len(hist), NSOLVE + 1):
    t = time.time(); r, la = step(la); hist.append(r['dE_site']); res['info'].append(dict(iters=r['iters'], res=r['res']))
    np.save(LK + '.tmp.npy', la); os.replace(LK + '.tmp.npy', LK); dump()
    log(f'solve {k}: e = {r["dE_site"]:.9e}  it {r["iters"]} res {r["res"]:.1e}  {time.time() - t:.0f}s')
h = hist; a_, b_, c_ = h[-3:]
res['summary'] = dict(at={str(k): h[k] for k in (10, 20, 30, 40) if k < len(h)}, last=h[-1], last_ratio=(c_ - b_) / (b_ - a_),
                      aitken=c_ - (c_ - b_) ** 2 / (c_ - 2 * b_ + a_), max_res=max(i['res'] for i in res['info']))
res['done'] = True; dump(); log(f'DONE {res["summary"]}')
