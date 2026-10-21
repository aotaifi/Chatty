"""Amplitude refresh after the flip (README "Amplitude refresh after the flip", pre-registered before any run).
  python cx_refresh.py OUT SPEC.json   SPEC: {"la_star_s1": path, "tau": 0.1, "solves": 40, "fit_sec": 900}
Stages (one job): (A) phi*(s_2) by 40 self-consistent solves (subprocess); (B) warm-ViT minSR fine-tune to log phi* with
all-bond edge loss (cx_reanchor.py, subprocess, fit_only); (C) E_FN(s_2, fitted ViT ratios): 1 solve + 3 own solves,
oracle-veto variant, majoriser-gap decomposition (bulk / near-node / into X), cost.
"""
import os, sys, time, json, subprocess
import numpy as np
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); STAGE = SPEC.get('stage', 'all')
F = os.path.join(OUT, f'cx_refresh_{STAGE}.json'); T00 = time.time()
if STAGE == 'all':
    res = dict(spec=SPEC, sec={})
    if not os.path.exists(os.path.join(OUT, 'la_star_s2.npy')):
        sp = os.path.join(OUT, 'specA.json'); json.dump({**SPEC, 'stage': 'A'}, open(sp, 'w')); t0 = time.time()
        assert subprocess.run([sys.executable, os.path.join(HERE, 'cx_refresh.py'), OUT, sp]).returncode == 0
        res['sec']['A'] = time.time() - t0
    VD = os.path.join(OUT, 'vit'); os.makedirs(VD, exist_ok=True)
    if not os.path.exists(os.path.join(VD, 'la_table.npy')):
        sp = os.path.join(OUT, 'specB.json')
        json.dump(dict(k=1, seed=0, opt='minsr', target_la=os.path.join(OUT, 'la_star_s2.npy'),
                       target_s=os.path.join(OUT, 's2.npy'), edges='all', fit_only=True,
                       fit_sec=SPEC.get('fit_sec', 900)), open(sp, 'w')); t0 = time.time()
        assert subprocess.run([sys.executable, os.path.join(HERE, 'cx_reanchor.py'), VD, sp]).returncode == 0
        res['sec']['B'] = time.time() - t0
    json.dump(res, open(F, 'w'), indent=1)
    STAGE = 'C'
from cx_lib import *
from scipy.optimize import minimize_scalar
lab = Lab(); sec = lab.sec; EPS = 0.3
la_P, s_P = lab.lP0, lab.sP0
_, V, W = lab.VW(la_P, s_P); v = sec.vec(la_P, s_P); E = float(v @ sec.H(v)); del v
EL = lab.D0 + V - W; del V, W
def sg(a): return s_P * jnp.where(1.0 + a * (EL - E) >= 0, 1.0, -1.0).astype(s_P.dtype)
def lclip(a): return la_P + jnp.log(jnp.maximum(jnp.abs(1.0 + a * (EL - E)), EPS))
sd = float(jnp.sqrt(jnp.sum(sec.vec(la_P, s_P) ** 2 * (EL - E) ** 2)))
r_ = minimize_scalar(lambda a: lab.H_site(lclip(a), sg(a)), bounds=(-3.0 / sd, 0.0), method='bounded', options=dict(xatol=1e-4 / sd))
s1 = sg(float(r_.x)); del EL
def norm_la(la):
    a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))
def efn(la, s):
    _, u, info = sec.fn_solve(la, s)
    return info['dE_FN_site'], jnp.log(jnp.maximum(u, 1e-300) / sec.sqrt_n)
laPn = norm_la(la_P)
if STAGE == 'A':
    ls1 = norm_la(jnp.asarray(np.load(SPEC['la_star_s1'])))
    X = jnp.exp(ls1 - laPn) < SPEC.get('tau', 0.1)
    s2 = jnp.where(X, -s1, s1)
    e, lp = efn(la_P, s2); hist = [e]
    for k in range(SPEC.get('solves', 40)):
        e, lp = efn(lp, s2); hist.append(e)
        if k % 10 == 9: log(f'  phi*(s2) solve {k + 1}: {e:.6e}')
    np.save(os.path.join(OUT, 'la_star_s2.npy'), np.asarray(norm_la(lp))); np.save(os.path.join(OUT, 's2.npy'), np.asarray(s2, np.int8))
    json.dump(dict(history=hist, flipped=float(jnp.sum(sec.n * X) / jnp.sum(sec.n))), open(F, 'w'), indent=1)
    log('stage A done', hist[-1]); sys.exit(0)
# ---------------------------------------------------------------- stage C: evaluation
res = json.load(open(os.path.join(OUT, 'cx_refresh_all.json'))) if os.path.exists(os.path.join(OUT, 'cx_refresh_all.json')) else dict(spec=SPEC)
F = os.path.join(OUT, 'cx_refresh_all.json')
ra = json.load(open(os.path.join(OUT, 'cx_refresh_A.json'))) if os.path.exists(os.path.join(OUT, 'cx_refresh_A.json')) else {}
res['ceiling_s2'] = ra.get('history', [None])[-1]
ls2 = norm_la(jnp.asarray(np.load(os.path.join(OUT, 'la_star_s2.npy'))))
s2 = jnp.asarray(np.load(os.path.join(OUT, 's2.npy')).astype(np.float32))
la_fit = norm_la(jnp.asarray(np.load(os.path.join(OUT, 'vit', 'la_table.npy'))))
rv = json.load(open(os.path.join(OUT, 'vit', 'cx_reanchor.json')))
res['fit'] = rv['iter'].get('fit'); res['fit_cost_sec'] = rv['iter'].get('fit', {}).get('sec')
e_vit, _ = efn(la_P, s2); res['EFN_s2_vit_ratios'] = e_vit
e1, lp = efn(la_fit, s2); own = [e1]
for _ in range(3):
    e_, lp = efn(lp, s2); own.append(e_)
res['EFN_s2_fit_1solve'] = e1; res['EFN_s2_fit_own'] = own
qf = jnp.exp(ls2 - la_fit); Xf = qf < 0.1
res['X_fit_frac'] = float(jnp.sum(sec.n * Xf) / jnp.sum(sec.n))
res['EFN_s2_fit_oracle_veto'] = efn(jnp.where(Xf, la_fit - 40.0, la_fit), s2)[0]
log('E_FN s2: ViT', e_vit, ' fitted 1 solve', e1, ' own', own, ' oracle veto', res['EFN_s2_fit_oracle_veto']); dump(F, res)
CHW = 2048
@jax.jit
def gap_chunk(i0, la_b, ls):
    ii = jnp.minimum(i0 + jnp.arange(CHW), sec.D - 1); real = (i0 + jnp.arange(CHW)) < sec.D
    xx = sec.reps[ii]; vv = valid_bonds(xx)
    yy = jnp.where(vv, xx[:, None] ^ MASKS[None, :], xx[:, None])
    jj = SS.canon(sec.T, sec.reps, yy.reshape(-1)).reshape(CHW, NB)
    bd = vv & (s2[ii][:, None] * s2[jj] > 0) & real[:, None]
    phx = jnp.exp(ls[ii])[:, None]; phy = jnp.exp(ls[jj])
    r = jnp.exp(jnp.clip(la_b[jj] - la_b[ii][:, None], -60, 60))
    term = 0.5 * JB[None, :] * (jnp.sqrt(r) * phx - phy / jnp.sqrt(r)) ** 2 * sec.n[ii][:, None] * 0.5
    q = jnp.exp(ls - la_b); qx = q[ii][:, None]; qy = q[jj]
    ex = (qx < 0.1) | (qy < 0.1); nr = ~ex & ((qx < 0.3) | (qy < 0.3)); bk = ~ex & ~nr
    return jnp.array([jnp.sum(jnp.where(bd & bk, term, 0)), jnp.sum(jnp.where(bd & nr, term, 0)), jnp.sum(jnp.where(bd & ex, term, 0))])
for nm, lb_ in (('fitted', la_fit), ('vit', laPn)):
    g = sum(gap_chunk(i0, lb_, ls2) for i0 in range(0, sec.D, CHW)) / N
    res[f'gap_decomp_{nm}'] = dict(bulk=float(g[0]), near_node=float(g[1]), into_X=float(g[2]))
    log('gap', nm, res[f'gap_decomp_{nm}']); dump(F, res)
res['sec_C'] = time.time() - T00; dump(F, res); log('DONE', res['sec_C'])
