import os,json,numpy as np
from vit8_callable_amplitude_adapter import load
from callable_fn_loop import CachedAmplitude,K1FNEngine,SquareJ1J2

BASE=os.environ.get("BASE_CK","vit_J2=0.50_N=8x8_k=0.mpack")
NEW=os.environ.get("NEW_CK","vit8_fnmle_rep1_best.mpack")
DATA=os.environ.get("THRESHOLD_DATA","krylov_scaling_8x8_a1.20_tr4096_va2048.npz")
OUT=os.environ.get("NODE_OUT","vit8_fnmle_node_diag.json")
T0=-28.37107876288694

z=np.load(DATA)
states=z["train_states"].astype(np.uint64)
r0=z["rtrain"].astype(float)
w0=z["iwtrain"].astype(float)
w0/=w0.sum()

b0=load(BASE)
bn=load(NEW)
log0=b0.log_amplitude_bits(states)
amp=CachedAmplitude(bn.log_amplitude_bits)
amp.ensure(states)
logn=np.asarray([amp.loga(x) for x in states],float)
d=logn-log0
w=w0*np.exp(2*d-np.max(2*d))
w/=w.sum()
eng=K1FNEngine(SquareJ1J2(8,.5),amp)
T,rn,info=eng.fit_threshold(states,w)
old=(r0<=T0)
new=(rn<=T)
flip=old!=new
out=dict(
    T0=T0,T=float(T),
    flip_frac=float(flip.mean()),
    flip_mass_old=float(w0@flip),
    flip_mass_new=float(w@flip),
    threshold_ESS=float(1/(w@w)),
    amp_eval=int(amp.neval),
    delta_logamp_mean=float(np.mean(d)),
    delta_logamp_std=float(np.std(d)),
    delta_logamp_q=np.quantile(d,[.01,.1,.5,.9,.99]).tolist(),
    rnew_q=np.quantile(rn,[.01,.1,.5,.9,.99]).tolist())
print("MLE8_NODE",json.dumps(out),flush=True)
json.dump(out,open(OUT,"w"),indent=2)
