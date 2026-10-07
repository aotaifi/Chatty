"""Smoke test (CPU) of st6_net: model variants build, fit steps run, eval on reps.  Usage: python test_st6_fit_cpu.py TABLE CKPT"""
import sys, types, time
import numpy as np
import jax, jax.numpy as jnp
import st6_sector as SS
import st6_net as NT
z = np.load(sys.argv[1])
sec = types.SimpleNamespace(n=jnp.asarray(z['orbit'].astype(float)), T=jnp.asarray(SS.byte_tables(SS.space_group())),
                            reps=jnp.asarray(z['reps']))
la0 = jnp.log(jnp.abs(jnp.asarray(z['amp'])))
for spec in ({'kind': 'vit'}, {'kind': 'vit+cnn', 'C': 8}, {'kind': 'vit+rbm', 'C': 4},
             {'kind': 'vitbig', 'layers': 4, 'd_model': 96, 'heads': 12}):
    m = NT.Model(spec, sys.argv[2])
    t = time.time()
    fl, h = NT.fit(m, m.flat0, sec, la0, steps=4, lr=1e-4, B=16, K=4, nval=1, val_B_mult=1, warmup=2)
    la = m.eval_reps(fl, sec.reps[:3000], batch=1024)
    la_ref = m.eval_reps(m.flat0, sec.reps[:3000], batch=1024)
    print(spec, 'npar', m.npar, 'vit', m.npar_vit, 'sec', round(time.time() - t, 1), 'dla rms', float(jnp.std(la - la_ref)),
          'corr with psi0', float(jnp.corrcoef(la_ref, la0[:3000])[0, 1]), flush=True)
print('FIT CPU OK')
