"""Test O1 (README "De-oracling the G1 guide", O1): alpha_V = argmin exact <H> of the one-hop state; one hop (T0 part-(c) / T0-phys part-(1) construction, reused verbatim) on arbitrary bases:
the tail-trained tables t1s2_* (no control vit here; control = ppbf at the G1 alpha reproduces 7.056e-6).
Exact 6x6, no training.   python cx_o1.py OUT SPEC.json
  SPEC: {"refresh", "bases", "g1_json": cx_g1.json of the G1 run, "code_sha256"}
"""
import os, sys, time, hashlib
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_o1.json'); T00 = time.time()
res = dict(spec=SPEC, done=False)
def progress(msg):
    log(msg); open(os.path.join(OUT, 'progress.txt'), 'a').write(f'PROGRESS {time.strftime("%H:%M:%S")} {msg}\n')
# code drift guard (as cx_t1s.py)
HERE_ = os.path.dirname(os.path.abspath(__file__)); CODE_FILES = ('cx_o1.py', 'cx_lib.py')
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
def onehop(nm, lb, g1):
    """O1: state a(alpha) = |base| |1 + alpha (E_L - E)| with s2, E_L/E/sdE exactly as cx_g1.onehop. alpha_V = bounded argmin of the exact
    variational energy H_site(la_c(alpha), s2) (sector-weighted vec/energy, no FN solve); then one E_FN solve at alpha_V and one at the G1 alpha."""
    t0 = time.time()
    _, V, W = lab.VW(lb, s2); v = sec.vec(lb, s2); Eb = float(v @ sec.H(v)); del v
    ELb = lab.D0 + V - W; del V, W
    sdE = float(jnp.sqrt(jnp.sum(sec.vec(lb, s2) ** 2 * (ELb - Eb) ** 2)))
    def la_c(a): return lb + jnp.log(jnp.maximum(jnp.abs(1.0 + a * (ELb - Eb)), 1e-30))
    hist = []
    def hv(a):
        h = float(lab.H_site(la_c(a), s2)); hist.append((float(a), h)); return h
    lo, hi = -3.0 / sdE, 3.0 / sdE
    r_ = minimize_scalar(hv, bounds=(lo, hi), method='bounded', options=dict(maxiter=100, xatol=1e-4 / sdE))
    aV = float(r_.x); HV = float(r_.fun)
    grid = [(float(a), float(lab.H_site(la_c(a), s2))) for a in np.linspace(lo, hi, 41)]   # diagnostic only (unimodality / boundary check)
    gmin = min(grid, key=lambda g: g[1])
    progress(f'[{nm}] alpha_V {aV:.5f} <H>_site-E0 {HV:.4e} ({r_.nfev} evals; grid min {gmin[1]:.4e} at {gmin[0]:.4f}; bounds +-{hi:.4f})')
    eV = efn(la_c(aV), s2); progress(f'[{nm}] E_FN(alpha_V) {eV:.4e}')
    aG = g1['alpha']; HG = float(lab.H_site(la_c(aG), s2)); eG = efn(la_c(aG), s2)
    progress(f'[{nm}] G1 alpha {aG:.5f}: <H> {HG:.4e} E_FN {eG:.4e} (G1 file {g1["EFN"]:.4e})')
    return dict(alpha_V=aV, H_V=HV, EFN_V=eV, alpha_G1=aG, H_G1=HG, EFN_G1_recomputed=eG, EFN_G1_file=g1['EFN'], sdE=sdE, E_base=Eb,
                bound=hi, at_bound=bool(abs(abs(aV) - hi) < 1e-3 * hi), nfev=r_.nfev, grid=grid, grid_min=gmin, sec=time.time() - t0)
res['g1_json'] = SPEC['g1_json']; G1 = json.load(open(SPEC['g1_json']))['bases']
res['bases'] = {}
for k, (nm, path) in enumerate(SPEC['bases'].items()):
    lb = norm_la(jnp.asarray(np.load(path), jnp.float64))
    r = onehop(nm, lb, G1[nm]); res['bases'][nm] = dict(path=path, **r)
    if nm == 't1s2_ppbf':   # control: G1 alpha must reproduce 7.056e-6 to 3 digits
        rel = abs(r['EFN_G1_recomputed'] - 7.056e-6) / 7.056e-6
        res['control'] = dict(EFN=r['EFN_G1_recomputed'], ref=7.056e-6, rel=rel, ok=bool(rel < 5e-4)); dump(F, res)
        progress(f'CONTROL ppbf G1 alpha: E_FN {r["EFN_G1_recomputed"]:.4e} vs 7.056e-6 (rel {rel:.2e}); ok={res["control"]["ok"]}')
        if not res['control']['ok']:
            res['sec'] = time.time() - T00; dump(F, res); progress('CONTROL FAILED: stop'); sys.exit(2)
    dump(F, res)
res['sec'] = time.time() - T00; res['done'] = True; dump(F, res); progress(f'DONE {res["sec"]:.0f}')
