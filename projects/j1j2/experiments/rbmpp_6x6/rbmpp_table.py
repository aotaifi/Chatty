"""Exact evaluation of an RBM+PP guide on all 15,804,956 orbit representatives of the 6x6 sector (k=0, A1, flip+).

  python rbmpp_table.py OUT SPEC.json      SPEC: params (npz), tag, [batch 512], [lanczos_p 3], [fn true]
The guide is fully projected (space group 288 x spin flip = 576 images), so it lives in the sector; psi_P(rep) is
the per-configuration amplitude (complex: the RBM is complex). The sector vector is v_r = psi_P(rep) sqrt(n_r).
Real guide for the loop: the best real combination of Re v and Im v (2x2 generalised eigenproblem; equivalently
Re(e^{-i theta} v) for the optimal global phase). Written as OUT/sym_tables_<tag>.npz with lP (float64 log|psi|
per configuration) and sP (int8 sign), the format of chatty_stall6/data/sym_tables.npz (ST6_SYM).
Exact numbers (per site, relative to E0): <H> of the complex state, of Re v, of the real guide; w_s (wrong-sign
weight vs the exact ground state), amplitude infidelity; FN energy of the real guide; guide + p = 1..3 exact
Lanczos steps (E, variance) and the FN energy of the p = 1 Lanczos state.
"""
import json, os, sys, time
import numpy as np
import jax
import jax.numpy as jnp
import scipy.linalg
import st6_sector as SS
from lanczos_lib import lanczos_ritz
from rbmpp_lib import Lattice, make_fns
jax.config.update("jax_enable_x64", True)

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
ROOT = '/project/theorie/a/A.Otaifi/chatty_stall6'
CSR = os.environ.get('ST6_CSR', ROOT + '/csr'); TABLE = os.environ.get('ST6_TABLE', ROOT + '/data/psi0_6x6_table.npz')
N = 36; log = SS.log; T00 = time.time()
tag = SPEC.get('tag', 'guide')
F = os.path.join(OUT, f'table_{tag}.json')
res = dict(spec=SPEC)


def dump():
    json.dump(res, open(F, 'w'), indent=1, default=float)


z = np.load(SPEC['params'])
use_rbm = bool(z['rbm']) if 'rbm' in z.files else True
p = {k: jnp.asarray(z[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}
lat = Lattice(6, 0.5, 'full')
fns = make_fns(lat, jnp.float64, use_rbm)
AR = jnp.arange(N, dtype=jnp.uint64)
lp_batch = jax.jit(jax.vmap(lambda x: fns['logpsi'](p, (((x >> AR) & jnp.uint64(1)).astype(jnp.int32) * 2 - 1)
                                                    .astype(jnp.float64))))

sec = SS.Sector(CSR, TABLE)
sec.offload()
B = SPEC.get('batch', 512)
D = sec.D; reps = sec.reps
t0 = time.time()
out = np.empty(D, np.complex128)
for i in range(0, D, B):
    x = reps[i:i + B]; m = x.shape[0]
    if m < B: x = jnp.concatenate([x, jnp.repeat(x[:1], B - m)])
    out[i:i + m] = np.asarray(lp_batch(x))[:m]
    if i == 0 or (i // B) % 2000 == 0:
        log(f'  table {i}/{D}  {time.time() - t0:.0f}s')
res['table_sec'] = time.time() - t0
log(f'table done {res["table_sec"]:.0f}s')
sec.reload()
l = jnp.asarray(out); del out
v = jnp.exp(l - jnp.max(jnp.real(l))) * sec.sqrt_n
R = jnp.real(v); I = jnp.imag(v); del v
HR, HI = sec.H2(R, I)
A = np.array([[float(R @ HR), float(R @ HI)], [float(I @ HR), float(I @ HI)]]); A = 0.5 * (A + A.T)
Bm = np.array([[float(R @ R), float(R @ I)], [float(I @ R), float(I @ I)]])
E0 = sec.E0
res['E_complex'] = ((A[0, 0] + A[1, 1]) / (Bm[0, 0] + Bm[1, 1]) - E0) / N
res['E_re'] = (A[0, 0] / Bm[0, 0] - E0) / N
res['im_weight'] = Bm[1, 1] / (Bm[0, 0] + Bm[1, 1])
ev, ec = scipy.linalg.eigh(A, Bm)
c = ec[:, 0]
g = c[0] * R + c[1] * I
del R, I, HR, HI
g = g / jnp.linalg.norm(g)
la = jnp.log(jnp.maximum(jnp.abs(g) / sec.sqrt_n, 1e-300))
s = jnp.where(g >= 0, 1.0, -1.0).astype(jnp.float32)
if float(s[0]) < 0: s = -s
np.savez(os.path.join(OUT, f'sym_tables_{tag}.npz'), lP=np.asarray(la), sP=np.asarray(s, np.int8))
sc = sec.score(la, s)
res['guide'] = dict(dE_site=sc['dE_site'], E_site=sc['E_site'], w_s=sc['w_s'], amp_infid=sc['amp_infid'],
                    rel_err=sc['dE_site'] / abs(E0 / N), c=c.tolist())
log('guide', res['guide'], 'E_complex', res['E_complex'], 'E_re', res['E_re'], 'im_weight', res['im_weight'])
dump()
if SPEC.get('fn', True):
    _, _, info = sec.fn_solve(la, s)
    res['guide']['dE_FN_site'] = info['dE_FN_site']; log('FN', info); dump()
pl = SPEC.get('lanczos_p', 3)
if pl:
    vg = sec.vec(la, s)
    rl = lanczos_ritz(lambda x: sec.Hm(x), vg, pl)
    res['lanczos'] = [dict(p=r['p'], dE_site=(r['E'] - E0) / N, var_site=r['var'] / N) for r in rl]
    log('lanczos', res['lanczos']); dump()
    if SPEC.get('fn', True):
        psi = rl[1]['psi']
        la1 = jnp.log(jnp.maximum(jnp.abs(psi) / sec.sqrt_n, 1e-300)); s1 = jnp.where(psi >= 0, 1.0, -1.0).astype(jnp.float32)
        _, _, i1 = sec.fn_solve(la1, s1); res['lanczos'][1]['dE_FN_site'] = i1['dE_FN_site']
        log('FN of Lanczos p1', i1['dE_FN_site'])
res['sec'] = time.time() - T00; dump(); log('DONE', res['sec'])
