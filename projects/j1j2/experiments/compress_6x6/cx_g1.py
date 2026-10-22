"""Test G1 (README "Test G1"): one hop (T0 part-(c) / T0-phys part-(1) construction, reused verbatim) on arbitrary bases:
the tail-trained tables (t1s2_*), the T1s ones, and a control = ViT VMC table (lab.lP0), which must reproduce T0-phys's ViT one-hop.
Exact 6x6, no training.   python cx_g1.py OUT SPEC.json
  SPEC: {"refresh": run dir (s2.npy), "bases": {name: la_table.npy}, "control_ref": cx_t0phys.json, "code_sha256": {...}}
"""
import os, sys, time, hashlib
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_g1.json'); T00 = time.time()
res = dict(spec=SPEC, done=False)
def progress(msg):
    log(msg); open(os.path.join(OUT, 'progress.txt'), 'a').write(f'PROGRESS {time.strftime("%H:%M:%S")} {msg}\n')
# code drift guard (as cx_t1s.py)
HERE_ = os.path.dirname(os.path.abspath(__file__)); CODE_FILES = ('cx_g1.py', 'cx_lib.py')
SHA_RUN = {fn: (hashlib.sha256(open(os.path.join(HERE_, fn), 'rb').read()).hexdigest() if os.path.exists(os.path.join(HERE_, fn)) else None) for fn in CODE_FILES}
SHA_EXP = SPEC.get('code_sha256'); res['code_sha256'] = dict(expected=SHA_EXP, running=SHA_RUN, dir=HERE_)
bad = [fn for fn in CODE_FILES if (SHA_EXP or {}).get(fn) != SHA_RUN[fn]]
res['code_sha256']['match'] = not bad
if bad:
    res['code_sha256']['mismatch'] = bad; dump(F, res); progress(f'CODE DRIFT: sha256 mismatch for {bad}; exit 3'); sys.exit(3)
dump(F, res)
lab = Lab(); sec = lab.sec
def norm_la(la):
    a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))
def efn(la, s):
    _, u, info = sec.fn_solve(la, s); return info['dE_FN_site']
RF = SPEC['refresh']
ls2 = norm_la(jnp.asarray(np.load(os.path.join(RF, 'la_star_s2.npy'))))
s2 = jnp.asarray(np.load(os.path.join(RF, 's2.npy')).astype(np.float32))
def onehop(nm, lb):
    """T0 (c) / T0-phys (1): a = |base| |1 + alpha (E_L - E)|, E_L of (|base|, s2); bounded alpha search, 12 evaluations."""
    t0 = time.time()
    rec = dict(EFN_zero=efn(lb, s2), EFN_zero_veto=efn(jnp.where(jnp.exp(ls2 - lb) < 0.1, lb - 40.0, lb), s2))
    _, V, W = lab.VW(lb, s2); v = sec.vec(lb, s2); Eb = float(v @ sec.H(v)); del v
    ELb = lab.D0 + V - W; del V, W
    sdE = float(jnp.sqrt(jnp.sum(sec.vec(lb, s2) ** 2 * (ELb - Eb) ** 2)))
    def la_c(a): return lb + jnp.log(jnp.maximum(jnp.abs(1.0 + a * (ELb - Eb)), 1e-30))
    hist = []
    def obj(a):
        e_ = efn(la_c(a), s2); hist.append((float(a), e_)); progress(f'  {nm} alpha {a:.5f}: E_FN {e_:.4e}'); return e_
    r_ = minimize_scalar(obj, bounds=(-3.0 / sdE, 3.0 / sdE), method='bounded', options=dict(maxiter=12, xatol=1e-3 / sdE))
    ac = float(r_.x); lc = norm_la(la_c(ac)); Xc = jnp.exp(ls2 - lc) < 0.1
    rec.update(alpha=ac, sdE=sdE, E_base=Eb, EFN=min(h[1] for h in hist), alpha_of_min=min(hist, key=lambda h: h[1])[0],
               EFN_veto=efn(jnp.where(Xc, lc - 40.0, lc), s2), X_frac=float(jnp.sum(sec.n * Xc) / jnp.sum(sec.n)), scan=hist, sec=time.time() - t0)
    progress(f'[{nm}] zero {rec["EFN_zero"]:.4e} (veto {rec["EFN_zero_veto"]:.4e}); one-hop alpha {ac:.5f} E_FN {rec["EFN"]:.4e} veto {rec["EFN_veto"]:.4e}')
    del ELb
    return rec
# ---------------------------------------------------------------- control first: ViT VMC table, must reproduce T0-phys
ref = json.load(open(SPEC['control_ref']))['c']['vit']
rc = onehop('control_vit', norm_la(lab.lP0))
rel = abs(rc['EFN'] - ref['EFN']) / ref['EFN']; relv = abs(rc['EFN_veto'] - ref['EFN_veto']) / ref['EFN_veto']
res['control'] = dict(rec=rc, ref_EFN=ref['EFN'], ref_EFN_veto=ref['EFN_veto'], ref_alpha=ref['alpha'], rel_EFN=rel, rel_EFN_veto=relv,
                      ok=bool(rel < 1e-3 and relv < 1e-3))
dump(F, res)
progress(f'CONTROL vit: E_FN {rc["EFN"]:.4e} vs T0-phys {ref["EFN"]:.4e} (rel {rel:.2e}); veto rel {relv:.2e}; ok={res["control"]["ok"]}')
if not res['control']['ok']:
    res['sec'] = time.time() - T00; dump(F, res); progress('CONTROL FAILED: stop before the bases'); sys.exit(2)
# ---------------------------------------------------------------- bases
res['bases'] = {}
for nm, path in SPEC['bases'].items():
    lb = norm_la(jnp.asarray(np.load(path), jnp.float64))
    res['bases'][nm] = dict(path=path, **onehop(nm, lb)); dump(F, res)
res['sec'] = time.time() - T00; res['done'] = True; dump(F, res); progress(f'DONE {res["sec"]:.0f}')
