"""Test T0b (README "Test T0b"; cx_t0 plus bases B2, B1, no part (c)): determinant/Pfaffian bases as the bad-bond field for s_2. Exact 6x6, no training.
  python cx_t0b.py OUT SPEC.json   SPEC: {"refresh": run dir, "pp": path, "A": path, "B2": path, "B1": path}
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_t0b.json'); T00 = time.time()
lab = Lab(); sec = lab.sec
def norm_la(la):
    a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))
def efn(la, s):
    _, u, info = sec.fn_solve(la, s); return info['dE_FN_site']
RF = SPEC['refresh']
ls2 = norm_la(jnp.asarray(np.load(os.path.join(RF, 'la_star_s2.npy'))))
s2 = jnp.asarray(np.load(os.path.join(RF, 's2.npy')).astype(np.float32))
bases = {'vit': norm_la(lab.lP0), 'vit_refreshed': norm_la(jnp.asarray(np.load(os.path.join(RF, 'vit', 'la_table.npy'))))}
for nm, key in (('pp', 'pp'), ('rbmpp', 'A'), ('B2', 'B2'), ('B1', 'B1')):
    z = np.load(SPEC[key]); bases[nm] = norm_la(jnp.asarray(z['lP'], jnp.float64)); del z
dec, _ = lab.decade_of(ls2)
PT = sec.vec(ls2, s2) ** 2
CHW = 2048
@jax.jit
def bond_chunk(i0, lb):
    ii = jnp.minimum(i0 + jnp.arange(CHW), sec.D - 1); real = (i0 + jnp.arange(CHW)) < sec.D
    xx = sec.reps[ii]; vv = valid_bonds(xx)
    yy = jnp.where(vv, xx[:, None] ^ MASKS[None, :], xx[:, None])
    jj = SS.canon(sec.T, sec.reps, yy.reshape(-1)).reshape(CHW, NB)
    bd = vv & (s2[ii][:, None] * s2[jj] > 0) & real[:, None]
    w = jnp.where(bd, 0.5 * JB[None, :] * jnp.exp(ls2[ii][:, None] + ls2[jj]) * sec.n[ii][:, None], 0.0)
    dl = (lb[jj] - lb[ii][:, None]) - (ls2[jj] - ls2[ii][:, None])
    dx = dec[ii]
    nn = (jnp.arange(NB) < len(SS.NN))[None, :]
    out = []
    for q in range(-4, -16, -1):
        m = (dx == q)[:, None]
        out.append(jnp.array([jnp.sum(jnp.where(m, w, 0)), jnp.sum(jnp.where(m, w * dl ** 2, 0)),
                              jnp.sum(jnp.where(m & nn, w, 0)), jnp.sum(jnp.where(m & nn, w * dl ** 2, 0)),
                              jnp.sum(jnp.where(m & ~nn, w, 0)), jnp.sum(jnp.where(m & ~nn, w * dl ** 2, 0))]))
    return jnp.stack(out)
def bond_stats(lb):
    acc = sum(bond_chunk(i0, lb) for i0 in range(0, sec.D, CHW))
    acc = np.asarray(acc); tot = acc.sum(0)
    r = lambda a, b: float(np.sqrt(a / b)) if b > 0 else None
    return dict(total=dict(all=r(tot[1], tot[0]), NN=r(tot[3], tot[2]), NNN=r(tot[5], tot[4])),
                by_decade={str(q): dict(all=r(acc[k, 1], acc[k, 0]), NN=r(acc[k, 3], acc[k, 2]), NNN=r(acc[k, 5], acc[k, 4]),
                                        weight=float(acc[k, 0])) for k, q in enumerate(range(-4, -16, -1))})
def eps_stats(lb):
    e = lb - ls2; e = e - jnp.sum(PT * e)
    out = {}
    for q in range(-4, -16, -1):
        m = dec == q; w = float(jnp.sum(jnp.where(m, PT, 0)))
        if w <= 0: continue
        rms = float(jnp.sqrt(jnp.sum(jnp.where(m, PT * e ** 2, 0)) / w))
        out[str(q)] = dict(rms_e=rms, eps=rms * 10 ** (q / 2))
    tail = [v['eps'] for k, v in out.items() if -13 <= int(k) <= -9]
    return out, float(np.median(tail)) if tail else None
DE_VMC = dict(vit=1.32e-4, pp=6.3e-5, rbmpp=4.74e-5)  # B2/B1: guide.dE_site of table_B2.json / table_B1.json (rpTabB2_17166873, rpTabB1_17166872)
for nm in ('B2', 'B1'):
    DE_VMC[nm] = json.load(open(os.path.join(os.path.dirname(SPEC[nm]), f'table_{nm}.json')))['guide']['dE_site']
res = dict(spec=SPEC, bases={})
for nm, lb in bases.items():
    t0 = time.time()
    Xb = jnp.exp(ls2 - lb) < 0.1
    rec = dict(dE_VMC=DE_VMC.get(nm), EFN=efn(lb, s2), EFN_veto=efn(jnp.where(Xb, lb - 40.0, lb), s2), X_frac=float(jnp.sum(sec.n * Xb) / jnp.sum(sec.n)))
    rec['bond_error'] = bond_stats(lb)
    rec['eps_by_decade'], rec['eps_median_tail'] = eps_stats(lb)
    rec['sec'] = time.time() - t0
    log(f'[{nm}] E_FN {rec["EFN"]:.4e}  veto {rec["EFN_veto"]:.4e}  delta rms {rec["bond_error"]["total"]}  eps(tail median) {rec["eps_median_tail"]}')
    res['bases'][nm] = rec; dump(F, res)
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
