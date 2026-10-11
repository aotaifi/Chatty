"""Diagnostic: held-out vs training edge loss of a trained FN arm, symmetric f vs single images, top contributors.
  python l8_diag.py POOL PARAMS.npz STEP [n]
"""
import sys, numpy as np
import l8_core as C
import jax, jax.numpy as jnp
from l8_net import Corr, feats

pool, pfile, step = sys.argv[1], sys.argv[2], sys.argv[3]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 800
Z = np.load(pool); P_ = np.load(pfile); stats = [tuple(s) for s in P_['stats']]
st = int(P_['best']) if step == 'best' else int(step)
flat = jnp.asarray(P_[f's{st}'])
X, Y, ix, iy = Z['X'], Z['Y'], Z['ix'], Z['iy']; K = Y.shape[1]
U = {k: Z['U_' + k] for k in ('la', 's', 'V', 'W', 'T')}
bond = Z['bond'].astype(int); nval = Z['nval']
r = 0.5 * C.JB[bond] * np.exp(np.clip(U['la'][iy] - U['la'][ix][:, None], -60, 60))
wgt = np.where(U['s'][ix][:, None] * U['s'][iy] < 0, r, 0) * nval[:, None] / K
rg = np.random.default_rng(3); uc = np.unique(Z['chain'])
ho_c = rg.choice(uc, int(round(0.2 * len(uc))), replace=False); HO = np.isin(Z['chain'], ho_c)
m = Corr(32, 4); fs = jax.jit(m.f); fa = jax.jit(m.f_aug)
for name, mask in (('train', ~HO), ('held-out', HO)):
    ii = np.nonzero(mask)[0][:n]
    sx = jnp.asarray(C.bits_to_spins_np(X[ii])); sy = jnp.asarray(C.bits_to_spins_np(Y[ii].reshape(-1)))
    tx = jnp.asarray(feats(U['la'][ix[ii]], U['V'][ix[ii]], U['W'][ix[ii]], stats))
    ty = jnp.asarray(feats(U['la'][iy[ii]].reshape(-1), U['V'][iy[ii]].reshape(-1), U['W'][iy[ii]].reshape(-1), stats))
    w = np.exp(Z['lw'][ii]); Tx = U['T'][ix[ii]]; Ty = U['T'][iy[ii]]
    def loss(fx, fy):
        c = w[:, None] * wgt[ii] * ((fx - Tx)[:, None] - (fy.reshape(-1, K) - Ty)) ** 2
        return 0.5 * c.sum() / w.sum() / 64, c.sum(1)
    l0, _ = loss(np.zeros(len(ii)), np.zeros(len(ii) * K))
    fx = np.asarray(fs(flat, sx, tx), float); fy = np.asarray(fs(flat, sy, ty), float)
    ls, per = loss(fx, fy)
    out = [f'{name}: f=0 {l0:.3e}  sym {ls:.3e}']
    for g in (0, 3, 9):
        G = jnp.full(len(ii), g); Gy = jnp.full(len(ii) * K, g)
        out.append(f'img{g} {loss(np.asarray(fa(flat, sx, tx, G), float), np.asarray(fa(flat, sy, ty, Gy), float))[0]:.3e}')
    print('  '.join(out))
    top = np.argsort(-per)[:5]
    print('  top contributors share', per[top].sum() / per.sum(), 'f_x', fx[top], 'T_x', Tx[top], 'max wgt', wgt[ii][top].max(1),
          'la_x', U['la'][ix[ii]][top], 'beta', Z['beta'][ii][top])
    print('  f sd', fx.std(), 'T sd', Tx.std(), 'corr(f,T)', np.corrcoef(fx, Tx)[0, 1])
