import time, numpy as np, jax
import gfmc_6x6_gr_population_size as b
pool=np.load("energy_krylov_vs_vit_6x6_indep.npz")["states"].astype(np.uint64)
states=np.tile(pool,64)[:16384]
for n in (512,4096,16384):
    x=b.bits2x(states[:n])
    _=b.evalz(x)
    times=[]
    checksum=None
    for k in range(3):
        t=time.time(); z=b.evalz(x); dt=time.time()-t
        times.append(dt); checksum=float(np.real(z[:8]).sum())
    print("BENCH",n,"times",times,"median",float(np.median(times)),
          "states_per_sec",n/float(np.median(times)),"checksum8",checksum,flush=True)
print("BACKEND",jax.default_backend(),jax.devices(),flush=True)
