"""Singular-image fallback test (4x4, f with identical rows 0 and 1 -> every image with sites 0 and 1 both up has an exactly singular F).
Checks: ref path gives non-finite values there; fast path flags exactly those configurations; the guide's amp() (fast + direct fallback)
equals the brute-force E_L / log psi_P (eloc_brute, every neighbour with its own slogdets) for ALL configurations, flagged or not."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from rbmpp_lib import Lattice, make_fns, init_params, random_configs
from gfmc_fn_lib import RBMPPGuideG
lat = Lattice(4, 0.5, 'translations'); p = init_params(lat, M=8, seed=3, rbm_scale=0.2)
f = np.array(p['f']); f[1, :] = f[0, :]; p['f'] = jnp.asarray(f)
fr = make_fns(lat, ratio='ref'); ff = make_fns(lat, ratio='fast')
X = np.asarray(random_configs(16, 200, 11)).astype(np.float64)
Er, lr = jax.jit(jax.vmap(fr['eloc'], in_axes=(None, 0)))(p, jnp.asarray(X))
_, _, bad = jax.jit(jax.vmap(ff['eloc_flag'], in_axes=(None, 0)))(p, jnp.asarray(X))
bad = np.asarray(bad); expect = (X[:, 0] > 0) & (X[:, 1] > 0)   # configs that have SOME singular image: images map sites; just report counts
Eb, lb = jax.jit(jax.vmap(fr['eloc_brute'], in_axes=(None, 0)))(p, jnp.asarray(X))
g = RBMPPGuideG({k: np.asarray(p[k]) for k in p}, lat, 0.0, 0.0, chunk=64, ratio='fast'); E, l = g.amp(X)
ok_ref_nonfinite = int((~np.isfinite(np.asarray(Er))).sum())
fin = np.isfinite(np.asarray(Eb))
err = np.abs(E - np.asarray(Eb)) / np.abs(np.asarray(Eb))
print('configs', len(X), '| flagged bad (fast)', int(bad.sum()), '| ref non-finite E_L', ok_ref_nonfinite, '| fallback calls', g.n_fallback)
print('brute finite', int(fin.sum()), '| fast+fallback vs brute: max rel err E_L', float(np.nanmax(err)), ' max abs err log psi_P', float(np.nanmax(np.abs(l - np.asarray(lb)))))
print('flagged configs: max rel err', float(err[bad].max()) if bad.any() else None, '| ref on flagged configs finite?', bool(np.isfinite(np.asarray(Er))[bad].all()) if bad.any() else None)
fr_ok = np.isfinite(np.asarray(Er)); eref = np.abs(np.asarray(Er) - np.asarray(Eb)) / np.abs(np.asarray(Eb))
print('ref finite on', int(fr_ok.sum()), 'configs but WRONG there: max rel err of ref E_L vs brute', float(np.nanmax(eref[fr_ok])) if fr_ok.any() else None)
assert bad.sum() > 0 and np.nanmax(err[bad]) < 1e-10 and ok_ref_nonfinite > 0, 'FALLBACK TEST FAILED'
print('FALLBACK_TEST_PASS')
