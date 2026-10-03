import json,numpy as np
from vit8_callable_amplitude_adapter import load
from callable_fn_loop import CachedAmplitude,K1FNEngine,SquareJ1J2

T0=-28.37107876288694
H=np.load("vit8_fnmle_sr_pooled_handoff.npz")
tr=H["threshold_states"].astype(np.uint64)
wt=H["threshold_weights"].astype(float); wt/=wt.sum()
pool=H["pool_states"].astype(np.uint64)
wp=H["pool_weights"].astype(float); wp/=wp.sum()
r0tr=H["r0_threshold"].astype(float)

b=load("vit8_fnmle_sr_pooled.mpack")
amp=CachedAmplitude(b.log_amplitude_bits)
eng=K1FNEngine(SquareJ1J2(8,.5),amp)
T,rtr,info=eng.fit_threshold(tr,wt)
oldtr=r0tr<=T0; newtr=rtr<=T
fliptr=oldtr!=newtr

pd=np.load("krylov_phys8_a2_fixedT.npz")
r0p=pd["r"].astype(float)
eng.ensure_r(pool)
rp=np.array([eng.r_cache[int(x)] for x in pool])
oldp=r0p<=T0; newp=rp<=T
flipp=oldp!=newp
out={
 "T0":T0,"T1":float(T),"centers_trim":[float(x) for x in info],
 "threshold_flip_fraction":float(fliptr.mean()),
 "threshold_flip_mass_a1alpha":float(wt@fliptr),
 "physical_flip_fraction":float(flipp.mean()),
 "physical_flip_mass_a1sq":float(wp@flipp),
 "threshold_ess":float(1/(wt@wt)),
 "pool_ess":float(1/(wp@wp)),
 "r1_threshold_q":np.quantile(rtr,[.01,.1,.5,.9,.99]).tolist(),
 "r1_pool_q":np.quantile(rp,[.01,.1,.5,.9,.99]).tolist(),
 "amp_eval":int(amp.neval)}
json.dump(out,open("vit8_fnmle_sr_pooled_node.json","w"),indent=2)
print("SR_NODE",json.dumps(out),flush=True)
