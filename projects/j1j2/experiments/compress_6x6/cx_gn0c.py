"""Gate G-N0c (README compress_6x6 "M1 + gate for the loop"): table maker.  A smooth perturbation of phi_1 unrelated to the fit:
  la_pert = norm_la(log phi_1) + eps * f(x),  f = fixed random combination of nearest-neighbour z_i z_j correlators, symmetrised (group average over the 288
  space-group images; spin flip leaves z_i z_j invariant).  Two versions:
  lin : f = sum_b w_b C_b with w_b ~ N(0,1) (seed 11); the group average of any linear combination is mean(w) * sum_b C_b (the group is transitive on the
        72 NN bonds): f is the NN energy, the only invariant linear one (checked numerically: wsym uniform).
  quad: f = sum_{b!=b'} W_bb' C_b C_b' with W symmetric Gaussian (seed 12), group-averaged (bond-pair classes), a richer invariant smooth function.
eps is set so that the |phi_1|^2-weighted rms (about the weighted mean) of eps f over the BULK decades (>= -8, decade of |phi_1|^2) equals that of
r = norm_la(a_2) - norm_la(phi_1) over the same set (O2b round 1).  Writes OUT/pert_{lin,quad}_la.npy (float64) and OUT/cx_gn0c.json.
Then cx_n0.py is run with a2_table = this table (same sampler/seed as N0: the bond sample and c do not depend on a2).
  python cx_gn0c.py OUT SPEC.json   SPEC: refresh, o2b_table, target_la
"""
import os, sys, time
import numpy as np
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_gn0c.json'); res = dict(spec=SPEC, done=False)
def progress(msg):
    log(msg); open(os.path.join(OUT, 'progress.txt'), 'a').write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}\n')
lab = Lab(); sec = lab.sec
def norm_la(la):
    a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))
lphi = norm_la(jnp.asarray(np.load(SPEC['target_la']), jnp.float64)); la2 = norm_la(jnp.asarray(np.load(SPEC['o2b_table']), jnp.float64))
r = np.asarray(la2 - lphi)
dec, lp = lab.decade_of(lphi); dec_h = np.asarray(dec); p = np.asarray(sec.n * jnp.exp(lp)); p = p / p.sum()
bulk = dec_h >= -8; tail = ~bulk
NNB = len(SS.NN); perms = SS.space_group()
bond_id = {tuple(sorted(b)): k for k, b in enumerate(SS.NN)}
pb = np.array([[bond_id[tuple(sorted((int(g[i]), int(g[j]))))] for (i, j) in SS.NN] for g in perms])      # (288, 72) bond map of each group element
assert all(sorted(row) == list(range(NNB)) for row in pb), 'group does not permute the NN bonds'
rng = np.random.default_rng(11); w = rng.normal(size=NNB); wsym = np.zeros(NNB)
for g in range(perms.shape[0]): wsym[pb[g]] += w / perms.shape[0]
res['lin_wsym_spread'] = float(wsym.max() - wsym.min()); res['lin_wsym_mean_vs_mean_w'] = [float(wsym.mean()), float(w.mean())]
rng = np.random.default_rng(12); W = rng.normal(size=(NNB, NNB)); W = 0.5 * (W + W.T); np.fill_diagonal(W, 0.0); Wsym = np.zeros_like(W)
for g in range(perms.shape[0]): Wsym[np.ix_(pb[g], pb[g])] += W / perms.shape[0]
res['quad_Wsym_distinct_values_offdiag'] = int(len(np.unique(np.round(Wsym[~np.eye(NNB, dtype=bool)], 10))))
# invariance check of Wsym: permuting by any group element leaves it unchanged
res['quad_Wsym_invariance_maxdev'] = float(max(np.abs(Wsym[np.ix_(pb[g], pb[g])] - Wsym).max() for g in (1, 5, 77, 200)))
BI_ = jnp.asarray(np.array([b[0] for b in SS.NN], np.uint64)); BJ_ = jnp.asarray(np.array([b[1] for b in SS.NN], np.uint64))
wsym_j = jnp.asarray(wsym); Wsym_j = jnp.asarray(Wsym)
@jax.jit
def feats(x):
    C = (1.0 - 2.0 * (((x[:, None] >> BI_[None, :]) ^ (x[:, None] >> BJ_[None, :])) & jnp.uint64(1)).astype(jnp.float64))   # z_i z_j
    return C @ wsym_j, jnp.einsum('nb,bc,nc->n', C, Wsym_j, C)
fl, fq = [], []
for i in range(0, sec.D, 1 << 18):
    a, b = feats(sec.reps[i:i + (1 << 18)]); fl.append(np.asarray(a)); fq.append(np.asarray(b))
fl = np.concatenate(fl); fq = np.concatenate(fq)
# invariance check on random configs: f(x) equals f(image of x under a random group element)
rr = np.random.default_rng(3); chk = np.asarray(sec.reps)[rr.choice(sec.D, 200, replace=False)]
bit = ((chk[:, None] >> np.arange(36, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.int64)
def f_of_bits(b):
    z = 1 - 2 * b; C = np.array([z[:, i] * z[:, j] for (i, j) in SS.NN]).T; return C @ wsym, np.einsum('nb,bc,nc->n', C, Wsym, C)
gi = rr.integers(0, 288, 200); img = np.zeros_like(bit)
for k in range(200): img[k, perms[gi[k]]] = bit[k]
a1, b1 = f_of_bits(bit); a2_, b2_ = f_of_bits(img)
res['invariance_maxdev'] = dict(lin=float(np.abs(a1 - a2_).max()), quad=float(np.abs(b1 - b2_).max()), scale_lin=float(np.abs(a1).max()), scale_quad=float(np.abs(b1).max()))
ja = np.asarray(feats(sec.reps[:200])[0]); res['gpu_vs_numpy_lin_maxdev'] = float(np.abs(ja - f_of_bits(((np.asarray(sec.reps[:200])[:, None] >> np.arange(36, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.int64))[0]).max())
def wstat(v, m):
    q = p[m]; Q = q.sum(); mu = (q * v[m]).sum() / Q; return float(np.sqrt((q * (v[m] - mu) ** 2).sum() / Q))
rb = wstat(r, bulk); rt = wstat(r, tail)
res['r'] = dict(bulk_rms=rb, tail_rms=rt, bulk_prob=float(p[bulk].sum()))
for nm, f in (('lin', fl), ('quad', fq)):
    eps = rb / wstat(f, bulk); la = np.asarray(lphi) + eps * f
    np.save(os.path.join(OUT, f'pert_{nm}_la.npy'), la)
    res[nm] = dict(eps=float(eps), bulk_rms=wstat(eps * f, bulk), tail_rms=wstat(eps * f, tail), finite=bool(np.isfinite(la).all()),
                   corr_with_r_bulk=float(np.corrcoef(f[bulk], r[bulk])[0, 1]), max_abs=float(np.abs(eps * f).max()))
    progress(f'{nm}: {json.dumps(res[nm])}')
res['done'] = True; dump(F, res); progress('DONE maker')
