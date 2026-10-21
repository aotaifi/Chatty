"""Amendment AR-1 add-ons to the amplitude refresh: second seed, decade splices, flip labels from the fitted network.
  python cx_refresh2.py OUT SPEC.json   SPEC: {"base": run dir of cx_refresh (17409419), "fit_sec": 900}
"""
import os, sys, time, json, subprocess
import numpy as np
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = SPEC['base']; T00 = time.time()
F = os.path.join(OUT, 'cx_refresh2.json'); res = dict(spec=SPEC)
VD1 = os.path.join(OUT, 'vit_s1'); os.makedirs(VD1, exist_ok=True)
if not os.path.exists(os.path.join(VD1, 'la_table.npy')):
    sp = os.path.join(OUT, 'spec_s1.json')
    json.dump(dict(k=1, seed=1, opt='minsr', target_la=os.path.join(BASE, 'la_star_s2.npy'), target_s=os.path.join(BASE, 's2.npy'),
                   edges='all', fit_only=True, fit_sec=SPEC.get('fit_sec', 900)), open(sp, 'w'))
    t0 = time.time(); assert subprocess.run([sys.executable, os.path.join(HERE, 'cx_reanchor.py'), VD1, sp]).returncode == 0
    res['fit_s1_sec'] = time.time() - t0
from cx_lib import *
lab = Lab(); sec = lab.sec
def norm_la(la):
    a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))
def efn(la, s):
    _, u, info = sec.fn_solve(la, s)
    return info['dE_FN_site'], jnp.log(jnp.maximum(u, 1e-300) / sec.sqrt_n)
z = np.load(TABLE); amp = np.asarray(z['amp'], np.float64); del z
s0 = jnp.where(jnp.asarray(amp) >= 0, 1.0, -1.0).astype(jnp.float32)
la0 = jnp.log(jnp.maximum(jnp.abs(jnp.asarray(amp)), 1e-300)); del amp
P0 = sec.vec(la0, s0) ** 2; nz0 = P0 > 0
ls2 = norm_la(jnp.asarray(np.load(os.path.join(BASE, 'la_star_s2.npy'))))
s2 = jnp.asarray(np.load(os.path.join(BASE, 's2.npy')).astype(np.float32))
s0 = s0 * (1.0 if float(jnp.sum(P0 * s0 * s2)) >= 0 else -1.0)
W2 = (s2 != s0) & nz0
res['W2_weight'] = float(jnp.sum(P0 * W2))
laPn = norm_la(lab.lP0)
dec, _ = lab.decade_of(ls2)
def mu_of(lphi, s):
    ph = jnp.exp(lphi); return s * sec.H(s * ph * sec.sqrt_n) / sec.sqrt_n
def pr(A, B):
    a = float(jnp.sum(P0 * (A & B))); return dict(precision=a / max(float(jnp.sum(P0 * A)), 1e-300), recall=a / max(float(jnp.sum(P0 * B)), 1e-300))
Xe = jnp.exp(ls2 - laPn) < 0.1; Ke = Xe & (mu_of(ls2, s2) > 0)
res['labels_exact'] = dict(X=pr(Xe, W2), KKT=pr(Ke, W2), X_frac=float(jnp.sum(sec.n * Xe) / jnp.sum(sec.n)))
log('exact labels', res['labels_exact']); dump(F, res)
res['seeds'] = {}
for sd_, vd in (('0', os.path.join(BASE, 'vit')), ('1', VD1)):
    lf = norm_la(jnp.asarray(np.load(os.path.join(vd, 'la_table.npy'))))
    rec = {}
    e1, lph = efn(lf, s2); rec['EFN_1solve'] = e1
    if sd_ == '1':
        own = [e1]; lp = lph
        for _ in range(3):
            e_, lp = efn(lp, s2); own.append(e_)
        rec['EFN_own'] = own
    # decade splices: fitted ratios where dec >= t, exact phi* ratios below (offsets aligned in the phi* metric)
    PT = sec.vec(ls2, s2) ** 2
    lfa = lf - jnp.sum(PT * (lf - ls2))
    rec['splice'] = {}
    for t in (-8, -9, -10, -11):
        e_, _ = efn(jnp.where(dec >= t, lfa, ls2), s2); rec['splice'][str(t)] = e_
    # labels from the fitted network's FN solution
    Xf = jnp.exp(norm_la(lph) - lf) < 0.1; Kf = Xf & (mu_of(norm_la(lph), s2) > 0)
    rec['labels_fit'] = dict(X=pr(Xf, W2), KKT=pr(Kf, W2), X_frac=float(jnp.sum(sec.n * Xf) / jnp.sum(sec.n)),
                             overlap_X_with_exact=pr(Xf, Xe), overlap_KKT_with_exact=pr(Kf, Ke))
    log(f'seed {sd_}:', json.dumps(rec)); res['seeds'][sd_] = rec; dump(F, res)
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
