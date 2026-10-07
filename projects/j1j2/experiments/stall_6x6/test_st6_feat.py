"""Check st6_feat: (1) cluster features are invariant under the space group x flip; (2) the linear-method optimiser
reaches the same optimum as a dense scipy minimisation on the synthetic sector of test_st6_small."""
import numpy as np, scipy.optimize, jax, jax.numpy as jnp
import st6_sector as SS, st6_feat as FT
# (1) invariance
rng = np.random.default_rng(1)
x = np.zeros(64, np.uint64)
for i in range(64):
    b = rng.choice(36, 18, replace=False); x[i] = np.sum(np.uint64(1) << b.astype(np.uint64))
T = jnp.asarray(SS.byte_tables(SS.space_group()))
err = 0.0
for kind in ('pair_all', 'w3b4', 'w3b6'):
    cl = FT.cluster_classes(kind)
    for name, masks in cl:
        f0 = FT._feat_chunk(jnp.asarray(x), jnp.asarray(masks))
        for g, fl in ((7, False), (123, True), (287, True)):
            y = SS.image(T, jnp.asarray(x), jnp.full(64, g), jnp.full(64, fl))
            err = max(err, float(jnp.max(jnp.abs(FT._feat_chunk(y, jnp.asarray(masks)) - f0))))
    print(kind, len(cl), 'classes')
print('feature invariance max err', err)
# (2) optimiser on the synthetic sector
exec(open('test_st6_small.py').read().split("if len(sys.argv) > 1:")[0].split("x = rng.normal(size=D)")[0])
K = 4
Fm = rng.normal(size=(K, D)).astype(np.float16); F = [jnp.asarray(r) for r in Fm]
base = jnp.asarray(np.log(np.abs(amp)) + 0.5 * rng.normal(size=D))
s = jnp.asarray(np.sign(v0))
fam = FT.Family(S, base, F)
c, h = fam.optimize(np.zeros(K), 'var', s, iters=30, tol=1e-13)
def Ec(c):
    la = np.asarray(base) + Fm.astype(np.float64).T @ c
    vv = np.asarray(s) * np.exp(la - la.max()) * np.sqrt(n)
    return vv @ Hd @ vv / (vv @ vv)
ref = scipy.optimize.minimize(Ec, np.zeros(K), method='BFGS', options=dict(gtol=1e-12))
print('LM', h[0], '->', h[-1], ' scipy', ref.fun, ' diff', h[-1] - ref.fun)
# fn mode: frozen FN energy minimiser vs dense
la_g = base
c2, h2 = fam.optimize(np.zeros(K), 'fn', s, la_g=la_g, iters=30, tol=1e-13)
A = S.fn_op(la_g, s)
def Efn(c):
    la = np.asarray(base) + Fm.astype(np.float64).T @ c
    vv = np.exp(la - la.max()) * np.sqrt(n); vv /= np.linalg.norm(vv)
    return float(jnp.asarray(vv) @ A(jnp.asarray(vv)))
ref2 = scipy.optimize.minimize(Efn, np.zeros(K), method='BFGS', options=dict(gtol=1e-12))
print('LM fn', h2[0], '->', h2[-1], ' scipy', ref2.fun, ' diff', h2[-1] - ref2.fun)
