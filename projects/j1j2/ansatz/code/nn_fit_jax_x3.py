"""nn_fit_jax.py OUTDIR MAPDIR EXTRA_LPART [shuf] : JAX float64 port of ref2hop/NN/nn_fit.py (sha e4eef208), arm C fast copy (cycle6f_store).
Same preamble (inputs, Z standardisation, U target) verbatim; same init (numpy rng 0), same gradient check (numpy finite differences, rng 1, now against the JAX gradient),
same holdout (rng 2), same batch draws (numpy rs = rng 0, one rs.random(B) per step, drawn on the host in step order and fed to the device; the row lookup
tr[searchsorted(cdf, u*wsum_tr)] runs on the device and is checked against numpy's on the first step of every 500-step segment), same manual backprop,
same Adam (1e-3, 0.9, 0.999, 1e-8; bias corrections and the cosine lr computed on the host in Python/numpy exactly as nn_fit.py), same logs and outputs
(tables/t_R.npy, tables/t_W.npy, nn.json, GRADCHECK line, D-norm eps2 via dnorm_lib). Extra outputs only: params.npz (final parameters), steploss.npy (every step's batch loss).
Only NN_LOSS=mse (the production loss) is ported. Differences to nn_fit.py are float reassociation only (reductions, matmul order on the device)."""
import os, sys, json, time
import numpy as np
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
C='/scratch/a/A.Otaifi/tmp/wt_ratnet/projects/j1j2/ansatz/code'; sys.path.insert(0, C)
NNDIR = '/scratch/a/A.Otaifi/ansatz_runs/consol_6x6/ref2hop/NN'
import bond_split as BS
OUT, MAP, LPART = sys.argv[1:4]; SHUF = 'shuf' in sys.argv[4:]; os.makedirs(f'{OUT}/tables', exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time()-T0:7.0f}s] {m}', flush=True)
log(f'nn_fit_jax: jax {jax.__version__} backend {jax.default_backend()} devices {jax.devices()} x64 {jax.config.jax_enable_x64}')
LOSS = os.environ.get('NN_LOSS', 'mse'); assert LOSS == 'mse', f'nn_fit_jax ports only NN_LOSS=mse, got {LOSS}'
# ---------------------------------------------------------------- preamble: verbatim nn_fit.py
OO = '/scratch/a/A.Otaifi/ansatz_runs/consol_6x6/ref2hop'; LP = f'{OO}/ref_parts/l_part.npy'; CH1 = '/scratch/a/A.Otaifi/ansatz_runs/consol_6x6/ratnet/lin_s1/la_table.npy'
MOM = '/scratch/a/A.Otaifi/ansatz_runs/consol_6x6/round2/moments'; A0 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy'
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert D == 15804956 and n.sum() == 9075135300
lB = np.load(A0).astype(np.float64); pt = np.load(f'{MAP}/p_out.npy').astype(np.float64); lT = np.load(f'{MAP}/lA_next.npy').astype(np.float64)
assert abs((n*pt*pt).sum()-1) < 1e-9 and np.isfinite(lT).all() and lT.shape == (D,); U = lT - lB
M = np.load(f'{MOM}/moments.npz'); NAMES = ['mu_H','mu_K','mu_B','lnB','mu_KH','mu_BH','mu_KK','mu_BB']
F = np.stack([M[k] if k != 'lnB' else lB for k in NAMES], 1); assert np.isfinite(F).all(); del M
h = np.load(CH1).astype(np.float64) - lB; assert np.isfinite(h).all() and h.shape == (D,); F = np.concatenate([F, h[:, None]], 1)
for pth in (LP, LPART):
    ex = np.load(pth).astype(np.float64); assert ex.shape == (D,) and np.isfinite(ex).all(); F = np.concatenate([F, ex[:, None]], 1)
assert F.shape[1] == 11
if os.environ.get('NN_EXTRA2'):
    ex2 = np.load(os.environ['NN_EXTRA2']).astype(np.float64); assert ex2.shape == (D,) and np.isfinite(ex2).all(); F = np.concatenate([F, ex2[:, None]], 1); log(f'NN_EXTRA2 column 11 from {os.environ["NN_EXTRA2"]}: range [{ex2.min():.4g},{ex2.max():.4g}]')
if os.environ.get('NN_EXTRA3'):
    ex3 = np.load(os.environ['NN_EXTRA3']).astype(np.float64); assert ex3.shape == (D,) and np.isfinite(ex3).all(); F = np.concatenate([F, ex3[:, None]], 1); log(f'NN_EXTRA3 column {F.shape[1] - 1} from {os.environ["NN_EXTRA3"]}: range [{ex3.min():.4g},{ex3.max():.4g}]')
NIN = F.shape[1]
if SHUF: F[:, 10] = F[np.random.default_rng(777).permutation(D), 10]; log('column 10 (l_prime) permuted rng 777')
wv = n*pt*pt; mu = (wv[:, None]*F).sum(0); sd = np.sqrt((wv[:, None]*(F-mu)**2).sum(0)); Z = (F-mu)/sd; del F
log(f'Z range [{Z.min():.4g},{Z.max():.4g}]; U range [{U.min():.4g},{U.max():.4g}]')
# ---------------------------------------------------------------- numpy MLP (verbatim; used for init and the finite differences)
H = int(os.environ.get('NN_WIDTH', 64)); rng = np.random.default_rng(0)
def init():
    return dict(W1=rng.normal(0, 1/np.sqrt(NIN), (NIN, H)), b1=np.zeros(H), W2=rng.normal(0, 1/np.sqrt(H), (H, H)), b2=np.zeros(H), w3=rng.normal(0, 1/np.sqrt(H), H), b3=np.zeros(()))
def fwd(P, z):
    a1 = np.tanh(z @ P['W1'] + P['b1']); a2 = np.tanh(a1 @ P['W2'] + P['b2']); return a2 @ P['w3'] + P['b3'], a1, a2
def lossgrad(P, z, y):
    f, a1, a2 = fwd(P, z); r = f - y; L = float((r*r).mean()); g = 2*r/len(r)
    G = {'w3': a2.T @ g, 'b3': g.sum()}; d2 = (g[:, None]*P['w3'])*(1-a2*a2); G['W2'] = a1.T @ d2; G['b2'] = d2.sum(0)
    d1 = (d2 @ P['W2'].T)*(1-a1*a1); G['W1'] = z.T @ d1; G['b1'] = d1.sum(0); return L, G
# ---------------------------------------------------------------- the same MLP and backprop in jax (float64)
KEYS = ('W1', 'b1', 'W2', 'b2', 'w3', 'b3')
def jfwd(P, z):
    a1 = jnp.tanh(z @ P['W1'] + P['b1']); a2 = jnp.tanh(a1 @ P['W2'] + P['b2']); return a2 @ P['w3'] + P['b3'], a1, a2
def jlossgrad(P, z, y):
    f, a1, a2 = jfwd(P, z); r = f - y; L = (r*r).mean(); g = 2*r/r.shape[0]
    G = {'w3': a2.T @ g, 'b3': g.sum()}; d2 = (g[:, None]*P['w3'])*(1-a2*a2); G['W2'] = a1.T @ d2; G['b2'] = d2.sum(0)
    d1 = (d2 @ P['W2'].T)*(1-a1*a1); G['W1'] = z.T @ d1; G['b1'] = d1.sum(0); return L, G
jlossgrad_jit = jax.jit(jlossgrad)
P = init()
# gradient check: numpy finite differences of the (verbatim) loss, 100 rows (rng 1), against the JAX gradient
ix = np.random.default_rng(1).choice(D, 100, replace=False); zc, yc = Z[ix], U[ix]
_, Gj = jlossgrad_jit({k: jnp.asarray(P[k]) for k in KEYS}, jnp.asarray(zc), jnp.asarray(yc)); G = {k: np.asarray(Gj[k]) for k in KEYS}
mx = 0.0; mxabs = 0.0; gmax = max(np.abs(np.atleast_1d(G[k])).max() for k in G); eps = 1e-6
for k in P:
    a = np.atleast_1d(P[k]); ga = np.atleast_1d(G[k])
    for j in np.ndindex(a.shape):
        old = a[j]; a[j] = old+eps; Lp = lossgrad({**P, k: a.reshape(P[k].shape)}, zc, yc)[0]; a[j] = old-eps; Lm = lossgrad({**P, k: a.reshape(P[k].shape)}, zc, yc)[0]; a[j] = old
        fd = (Lp-Lm)/(2*eps); g = ga[j]; mxabs = max(mxabs, abs(fd-g)); mx = max(mx, abs(fd-g)/max(abs(fd), abs(g), 1e-3*gmax))
_, Gn = lossgrad(P, zc, yc); gdiff = max(float(np.abs(np.asarray(Gn[k]) - G[k]).max()) for k in KEYS)
log(f'GRADCHECK max rel error {mx:.3e} (denominator max(|fd|,|g|,1e-3*max|g|)); max abs error {mxabs:.3e}; max|g| {gmax:.3e} [jax gradient vs numpy finite differences; max |jax - numpy backprop| {gdiff:.3e}]')
GC = mx
if mx > 1e-5: open(f'{OUT}/FAIL_gradcheck', 'w').write(str(mx)); sys.exit(1)
# data split, sampling (verbatim)
perm_ = np.random.default_rng(2).permutation(D); isva = np.zeros(D, bool); isva[perm_[:D//10]] = True; tr = np.nonzero(~isva)[0]; va = np.nonzero(isva)[0]; del perm_
cdf = np.cumsum(wv[tr]); wsum_tr = cdf[-1]; wva = wv[va]; wva_sum = wva.sum(); log(f'train rows {len(tr)} weight {wsum_tr:.4e} share {wsum_tr/(wsum_tr+wva_sum):.4f}; val rows {len(va)} weight {wva_sum:.4e} share {wva_sum/(wsum_tr+wva_sum):.4f} (random 10% holdout, rng 2)')
# ---------------------------------------------------------------- device arrays
dZ = jnp.asarray(Z); dU = jnp.asarray(U); dwv = jnp.asarray(wv); dtr = jnp.asarray(tr); dcdf = jnp.asarray(cdf); dwsum = jnp.asarray(wsum_tr, dtype=jnp.float64)
@jax.jit
def _chunk_wsums(P, r):
    f = jfwd(P, dZ[r])[0]; w = dwv[r]; return (w*(f-dU[r])**2).sum(), w.sum()
def wloss(rows):
    s = 0.0; ws = 0.0
    for i in range(0, len(rows), 1<<20):
        a, b = _chunk_wsums(Pd, jnp.asarray(rows[i:i+(1<<20)])); s += float(a); ws += float(b)
    return s/ws
@jax.jit
def _rows_of(u):
    return dtr[jnp.searchsorted(dcdf, u*dwsum)]
b1_, b2_, ea = 0.9, 0.999, 1e-8
def _step(carry, x):
    P, m, v = carry; u, lr, bc1, bc2 = x
    idx = dtr[jnp.searchsorted(dcdf, u*dwsum)]; L, G = jlossgrad(P, dZ[idx], dU[idx])
    m = {k: b1_*m[k]+(1-b1_)*G[k] for k in KEYS}; v = {k: b2_*v[k]+(1-b2_)*G[k]**2 for k in KEYS}
    P = {k: P[k] - lr*(m[k]/bc1)/(jnp.sqrt(v[k]/bc2)+ea) for k in KEYS}
    return (P, m, v), L
@jax.jit
def _segment(P, m, v, us, lrs, bc1s, bc2s):
    (P, m, v), Ls = jax.lax.scan(_step, (P, m, v), (us, lrs, bc1s, bc2s)); return P, m, v, Ls
# ---------------------------------------------------------------- Adam loop: host draws and scalars exactly as nn_fit.py, device steps in segments [1], [2..500], [501..1000], ...
Pd = {k: jnp.asarray(np.asarray(P[k], dtype=np.float64)) for k in KEYS}
md = {k: jnp.zeros_like(Pd[k]) for k in KEYS}; vd = {k: jnp.zeros_like(Pd[k]) for k in KEYS}
lr, B = 1e-3, 1 << 17; NSTEP = int(os.environ.get('NN_NSTEP', 20000)); rs = np.random.default_rng(0); curve = []; allL = np.empty(NSTEP)
SCHED = os.environ.get('NN_LRSCHED', '')
assert SCHED in ('', 'cosine'), SCHED
if SCHED: log(f'lr schedule cosine 1e-3 -> 1e-6 over NSTEP {NSTEP}')
segs = [(1, 1)] + ([(2, min(500, NSTEP))] if NSTEP >= 2 else []) + [(a, min(a + 499, NSTEP)) for a in range(501, NSTEP + 1, 500)]
def host_inputs(a, b):
    us = np.empty((b - a + 1, B)); lrs = np.empty(b - a + 1); bc1s = np.empty(b - a + 1); bc2s = np.empty(b - a + 1); lr_ = lr
    for j, step in enumerate(range(a, b + 1)):
        if SCHED == 'cosine': lr_ = 1e-6 + 0.5*(1e-3 - 1e-6)*(1 + np.cos(np.pi*(step-1)/NSTEP))
        us[j] = rs.random(B); lrs[j] = lr_; bc1s[j] = 1-b1_**step; bc2s[j] = 1-b2_**step
    return us, lrs, bc1s, bc2s
nidx = 0; tseg = time.time(); nxt = host_inputs(*segs[0])
for si, (a, b) in enumerate(segs):
    us, lrs, bc1s, bc2s = nxt
    Pd, md, vd, Ls = _segment(Pd, md, vd, jnp.asarray(us), jnp.asarray(lrs), jnp.asarray(bc1s), jnp.asarray(bc2s))   # async dispatch
    ridx = np.asarray(_rows_of(jnp.asarray(us[0]))); hidx = tr[np.searchsorted(cdf, us[0]*wsum_tr)]
    if not np.array_equal(ridx, hidx): log(f'IDXCHECK FAILED step {a}: device rows differ from numpy rows ({int((ridx != hidx).sum())} of {B})'); open(f'{OUT}/FAIL_idxcheck', 'w').write(str(a)); sys.exit(1)
    nidx += 1
    if si + 1 < len(segs): nxt = host_inputs(*segs[si + 1])   # host draws of the next segment overlap the device work
    Ls = np.asarray(Ls); allL[a-1:b] = Ls; run = [float(x) for x in Ls]
    if b % 500 == 0 or b == 1:
        vl = wloss(va); curve.append(dict(step=b, train_batch=float(np.mean(run)), val=float(vl))); log(f'step {b} train(batch mean) {curve[-1]["train_batch"]:.5e} val(weighted, holdout) {vl:.5e}')
    if b == 500 or (b == NSTEP and b < 500): log(f'TIMING {b} steps in {time.time()-tseg:.1f}s ({(time.time()-tseg)/b:.4f} s/step incl. compile); IDXCHECK passed on {nidx} segments')
log(f'IDXCHECK device rows == numpy rows on the first step of all {nidx} segments')
trl = wloss(tr); val = wloss(va); log(f'FINAL weighted loss train {trl:.5e} val {val:.5e}')
P = {k: np.asarray(Pd[k]) for k in KEYS}; np.savez(f'{OUT}/params.npz', **P); np.save(f'{OUT}/steploss.npy', allL)
_fwd_jit = jax.jit(lambda P, z: jfwd(P, z)[0])
f = np.empty(D)
for i in range(0, D, 1<<20): f[i:i+(1<<20)] = np.asarray(_fwd_jit(Pd, dZ[i:i+(1<<20)]))
EXTRA = {}
lo, hi = float(U.min()), float(U.max()); n_hi = int((f > hi).sum()); n_lo = int((f < lo).sum()); top = f >= hi-1e-9
log(f'raw net output range [{f.min():.4g},{f.max():.4g}]; target U [{lo:.4g},{hi:.4g}]; rows > hi {n_hi}, rows < lo {n_lo}, rows >= hi-1e-9 (guard flag) {int(top.sum())}')
assert np.isfinite(f).all(); np.save(f'{OUT}/tables/t_R.npy', lB+f); g = f.copy(); g[top] = 0.0; np.save(f'{OUT}/tables/t_W.npy', lB+g)
sys.path.insert(0, NNDIR); import dnorm_lib as DL; e2c, e2u, Duu_tot = DL.eps2_full(f, U, lo, hi, n, pt); log(f'D-norm eps2 of the copy: clipped {e2c:.5f}, unclipped {e2u:.5f} (D_UU {Duu_tot:.5e})'); EXTRA.update(eps2_direct_clipped=e2c, eps2_direct_unclipped=e2u, D_UU=Duu_tot)
json.dump(dict(**EXTRA, width=H, grad_check_max_rel=GC, final_train_loss=trl, final_val_loss=val, rows_gt_hi=n_hi, rows_lt_lo=n_lo, guard_rows=int(top.sum()), raw_range=[float(f.min()), float(f.max())], U_range=[lo, hi], curve=curve, shuf=SHUF, seed=0, sec=time.time()-T0,
               engine=dict(script=os.path.abspath(__file__), jax=jax.__version__, backend=jax.default_backend(), devices=[str(d_) for d_ in jax.devices()], idxcheck_segments=nidx, port_of='ref2hop/NN/nn_fit.py sha e4eef208')), open(f'{OUT}/nn.json', 'w'), indent=1)
log('DONE fit')
