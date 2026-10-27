"""c1_check.py: CPU checks a-d, g of Test C. Executes the code between '# C1 MODEL BEGIN/END' of cx_c1fit.py (the production model), real and complex.
usage: JAX_PLATFORMS=cpu PYTHONPATH=<compress6/code> python c1_check.py OUTJSON"""
import os, sys, json, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cx_lib import *
from rbmpp_lib import Lattice
src = open(os.path.join(HERE, 'cx_c1fit.py')).read()
blk = src[src.index('# C1 MODEL BEGIN'):src.index('# C1 MODEL END')]
lat = Lattice(6, 0.5, 'full'); IMG = lat.IMG; FLP = lat.FL.astype(jnp.float64); NH = N // 2
xy = np.array([(i % 6, i // 6) for i in range(N)])
DISP = jnp.asarray(np.array([[((xy[j, 0] - xy[i, 0]) % 6) + 6 * ((xy[j, 1] - xy[i, 1]) % 6) for j in range(N)] for i in range(N)], np.int32))
BF = True; BFW = 3; ND9 = 9; HW = 1; NBRW = NBR; NP_REAL = N * N + 2 * ND9 * N


def make(cplx):
    ns = dict(globals()); ns['COMPLEX'] = cplx
    exec('if True:\n' + blk.replace('# C1 MODEL BEGIN', ''), ns)
    return ns['logA_one'], ns['A_one']


lr, _ = make(False); lc, _ = make(True)
TH = '/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/f1_A/round_1/theta.npy'
th = jnp.asarray(np.load(TH)); assert th.size == NP_REAL
rms = float(jnp.sqrt(jnp.mean(th ** 2)))
reps = np.load('/project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz')['reps']
rng = np.random.default_rng(7)
S = jnp.asarray(reps[np.sort(rng.choice(len(reps), 1000, replace=False))].astype(np.uint64))
X = bits(S).astype(jnp.float64)
def chunked(f, ch=25):
    jf = jax.jit(jax.vmap(f, (None, 0)))
    return lambda t, Xa: jnp.concatenate([jf(t, Xa[i:i + ch]) for i in range(0, Xa.shape[0], ch)])
vr = chunked(lr); vc = chunked(lc); jr = chunked(jax.grad(lr), 10); jc = chunked(jax.grad(lc), 10)
out = {}
imn = lambda s_, seed: jnp.asarray(np.random.default_rng(seed).standard_normal(NP_REAL)) * s_ * rms
def cth(s_, seed=1): return jnp.concatenate([th, imn(s_, seed)])
a_r = np.asarray(vr(th, X))
a0 = np.asarray(vc(cth(0.0), X)); out['a_max_dev_cplx0_vs_real'] = float(np.max(np.abs(a0 - a_r)))
# broken deliberately: sign convention flipped in the complex arm must be detected
a1 = np.asarray(vc(cth(1.0), X)); out['b_im_sigma1_dev_median'] = float(np.median(np.abs(a1 - a_r))); out['b_im_sigma1_dev_max'] = float(np.max(np.abs(a1 - a_r)))
out['b_im_sigma1_dev_rms'] = float(np.sqrt(np.mean((a1 - a_r) ** 2)))
a2 = np.asarray(vc(cth(0.01), X)); out['b_im_sigma0.01_dev_median'] = float(np.median(np.abs(a2 - a_r)))
# la_table of F1 A absent locally: compare with the real code (done above). Also against the o2b-style table? not available for f1_A.
# c: zeros
z = np.load('/scratch/a/A.Otaifi/ansatz_runs/pfaffian_q/sample.npz'); xs = z['xs0']
Sx = jnp.asarray(reps[xs].astype(np.uint64)); Xs = bits(Sx).astype(jnp.float64)
d = np.asarray(vc(cth(0.01), Xs)) - np.asarray(vr(th, Xs))
out['c_median_lnAc_minus_lnAr'] = float(np.median(d)); out['c_mean'] = float(np.mean(d)); out['c_frac_ge0'] = float(np.mean(d >= 0)); out['c_p10_p90'] = [float(np.percentile(d, 10)), float(np.percentile(d, 90))]
# d: Jacobian Im columns
Xj = X[:50]
J0 = np.asarray(jc(cth(0.0), Xj)); J1 = np.asarray(jc(cth(0.01), Xj))
out['d_J_dtype'] = str(J0.dtype); out['d_J_shape'] = list(J0.shape)
out['d_im0_cols_max_abs'] = float(np.abs(J0[:, NP_REAL:]).max()); out['d_re0_cols_max_abs'] = float(np.abs(J0[:, :NP_REAL]).max())
out['d_im0.01_cols_max_abs'] = float(np.abs(J1[:, NP_REAL:]).max()); out['d_im0.01_cols_frac_nonzero'] = float(np.mean(np.abs(J1[:, NP_REAL:]).max(0) > 0))
Jr = np.asarray(jr(th, Xj)); out['d_Re_cols_cplx0_vs_real_maxdev'] = float(np.abs(J0[:, :NP_REAL] - Jr).max())
# g: finite differences, 10 Re + 10 Im random params, 50 reps, im_sigma 0.01
t0 = cth(0.01); h = 1e-6; ids = np.concatenate([rng.choice(np.arange(NP_REAL), 10, replace=False), NP_REAL + rng.choice(np.arange(NP_REAL), 10, replace=False)])
# avoid unused f-diagonal params (zero gradient) so the relative check is meaningful
diag = set(np.arange(N) * (N + 1)); ids = np.array([i for i in ids if (i % NP_REAL) not in diag])
vcj = vc
fd = np.zeros((50, len(ids)))
for k, i in enumerate(ids):
    e = jnp.zeros(2 * NP_REAL).at[i].set(h)
    fd[:, k] = (np.asarray(vcj(t0 + e, Xj)) - np.asarray(vcj(t0 - e, Xj))) / (2 * h)
Ja = J1[:, ids]; den = np.maximum(np.abs(Ja), np.abs(fd).max() * 1e-3)
out['g_n_params'] = int(len(ids)); out['g_max_rel_dev'] = float(np.max(np.abs(Ja - fd) / den)); out['g_max_abs_dev'] = float(np.max(np.abs(Ja - fd)))
out['g_scale_max_abs_J'] = float(np.abs(Ja).max())
out['g_max_rel_dev_vs_global_scale'] = float(np.max(np.abs(Ja - fd)) / np.abs(Ja).max())
json.dump(out, open(sys.argv[1], 'w'), indent=1); print(json.dumps(out, indent=1))
