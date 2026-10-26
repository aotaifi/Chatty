"""Deliverable 1: the sampled gradient of E_1 = E_VMC((1 + alpha(H - E)) v) at 4x4 vs the exact gradient of l8_loop4x4.py (stage gradref).
  python hopsr_check4x4.py REF.npz OUT.json [--chains 4096 --sweeps 8 --groups 32 --seed 1]
REF.npz = l8_loop4x4.py gradref output (flat, w, grad (all P), idx (class), E1).
(A) exact mode: the estimator on all 12870 configurations with weights |psi|^2: must equal the exact gradient to <= 1e-8 relative;
    broken 'noDE' and 'p0' must differ by > 1e-2; the moment chain rule (alpha differentiated) equals the g-form (alpha frozen) to <= 1e-8:
    Ritz stationarity, so 'alpha frozen' is not a failing control for E_1 (it is for E_FN, l8 hopb).
(B) sampled: per-component z = (g_s - g_ex)/err_jk over the class; PASS if chi2/dof <= 1.5 and |mean z| <= 3/sqrt(n);
    the broken versions must FAIL (chi2/dof > 3).  Pre-registered in hopsr/README.md.
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from hopsr_lib import *

ap = argparse.ArgumentParser(); ap.add_argument('ref'); ap.add_argument('out')
ap.add_argument('--chains', type=int, default=4096); ap.add_argument('--sweeps', type=int, default=8); ap.add_argument('--groups', type=int, default=32)
ap.add_argument('--seed', type=int, default=1); ap.add_argument('--therm', type=int, default=50); ap.add_argument('--skip_exact', action='store_true')
a = ap.parse_args()
E_ref = None
z = np.load(a.ref); flat = z['flat']; w = complex(z['w']); gex = z['grad']; idx = z['idx']; E1ex = float(z['E1'])
lat = Lattice(4, 0.5, 'translations'); N = lat.N
p0 = np.load('/scratch/a/A.Otaifi/l8/check4x4/params4.npz')
_, unrav = ravel_pytree({k: jnp.asarray(p0[k]) for k in KEYS})
est = HopEstimator(lat, True, unrav, chunk=1024, jchunk=512)
res = dict(ref=a.ref, E1ex_N=E1ex / N, n_class=int(len(idx)))
VAR = ('noDE',)


def cmp(g, err=None):
    ge, gs = gex[idx], g[idx]
    d = dict(cos=float(gs @ ge / np.linalg.norm(gs) / np.linalg.norm(ge)), rel=float(np.linalg.norm(gs - ge) / np.linalg.norm(ge)))
    if err is not None:
        zz = (gs - ge) / err[idx]; d.update(chi2dof=float(np.mean(zz ** 2)), meanz=float(zz.mean()), maxabsz=float(np.abs(zz).max()))
    return d


# ---------- (A) exact mode
if a.skip_exact: print('exact mode skipped'); A = dict(PASS=False, skipped=True)
else:
    basis = np.array([s for s in range(1 << N) if bin(s).count('1') == N // 2], np.int64)
    X = (((basis[:, None] >> np.arange(N)) & 1) * 2 - 1).astype(np.float64)
    t = time.time()
    _, l = est.at_y(jnp.asarray(flat), X); wt = np.exp(2 * (l.real - l.real.max()))
    E_ref = E1ex  # any reference; exact algebra
    r = est.estimate(flat, X, w, E_ref, wt=wt, groups=1, variants=VAR, with_O=False)
    A = dict(E1_N=r['E1'] / N, E1_rel=abs(r['E1'] - E1ex) / abs(E1ex), check_h2=r['check_h2'], true=cmp(r['grad']),
             BROKEN_noDE=cmp(r['grad_noDE']), BROKEN_p0=cmp(r['grad_p0']), IDENTITY_mom=cmp(r['grad_mom']), E1_gform_rel=abs(r['E1_gform'] - E1ex) / abs(E1ex), sec=time.time() - t)
    A['PASS'] = bool(A['true']['rel'] < 1e-8 and A['IDENTITY_mom']['rel'] < 1e-8 and A['E1_rel'] < 1e-10 and A['BROKEN_noDE']['rel'] > 1e-2 and A['BROKEN_p0']['rel'] > 1e-2)
    res['exact_mode'] = A
    print('EXACT MODE', json.dumps(A), flush=True)

# ---------- (B) sampled
t = time.time()
sweep = make_sampler(lat, make_fns(lat, jnp.float32, True)); p32 = to32(unrav(jnp.asarray(flat)))
key = jax.random.PRNGKey(5000 + a.seed); S = random_configs(N, a.chains, 8000 + a.seed)
for _ in range(a.therm):
    key, k = jax.random.split(key); S, acc = sweep(p32, S, k, N)
Xs = []
for _ in range(a.sweeps):
    key, k = jax.random.split(key); S, acc = sweep(p32, S, k, N); Xs.append(np.asarray(S).astype(np.float64))
Xs = np.concatenate(Xs)            # sample i belongs to chain i % chains; groups = chain % G (chains % G == 0)
assert a.chains % a.groups == 0
E_ref = E1ex if E_ref is None else E_ref
r = est.estimate(flat, Xs, w, E_ref, groups=a.groups, variants=VAR, with_O=False)
B = dict(samples=int(Xs.shape[0]), acc=float(acc), E1_N=r['E1'] / N, E1_err_N=r['E1_err'] / N, E1_z=(r['E1'] - E1ex) / r['E1_err'],
         check_h2=r['check_h2'], true=cmp(r['grad'], r['err']), BROKEN_noDE=cmp(r['grad_noDE'], r['err_noDE']),
         BROKEN_p0=cmp(r['grad_p0'], r['err_p0']), MOMENT_FORM=cmp(r['grad_mom'], r['err_mom']), E1_gform_N=r['E1_gform'] / N,
         rel_err_jk=float(np.linalg.norm(r['err'][idx]) / np.linalg.norm(gex[idx])), sec=time.time() - t, t_x=r['t_x'], t_y=r['t_y'])
n = len(idx)
B['PASS'] = bool(B['true']['chi2dof'] <= 1.5 and abs(B['true']['meanz']) <= 3 / np.sqrt(n))
B['BROKEN_FAIL'] = {k: bool(B[k]['chi2dof'] > 3) for k in ('BROKEN_noDE', 'BROKEN_p0')}
res['sampled'] = B
print('SAMPLED', json.dumps(B), flush=True)
json.dump(res, open(a.out, 'w'), indent=1)
ok = A['PASS'] and B['PASS'] and all(B['BROKEN_FAIL'].values())
print(f"VERDICT HOPSR_CHECK4X4 {'PASS' if ok else 'FAIL'}: exact-mode rel {A['true']['rel']:.2e} (noDE {A['BROKEN_noDE']['rel']:.2e}, p0 {A['BROKEN_p0']['rel']:.2e}, "
      f"moment-form {A['IDENTITY_mom']['rel']:.2e}); sampled cos {B['true']['cos']:.4f} chi2/dof {B['true']['chi2dof']:.3f} meanz {B['true']['meanz']:+.3f} "
      f"(noDE cos {B['BROKEN_noDE']['cos']:.4f} chi2/dof {B['BROKEN_noDE']['chi2dof']:.2f}; p0 cos {B['BROKEN_p0']['cos']:.4f} chi2/dof {B['BROKEN_p0']['chi2dof']:.2f}); "
      f"E1 z {B['E1_z']:+.2f}", flush=True)
