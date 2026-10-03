import math,json,os,sys
import numpy as np
sys.path.insert(0,os.getcwd())
from learned_fn8_callable_amplitude_adapter import load
from callable_fn_loop import robust_two_means_threshold,SquareJ1J2

T0=-28.37107876288694
ETAS=[0.0,0.01,0.02,0.05,0.10,0.20,0.35966756939888]
base=np.load("krylov_scaling_8x8_a1.20_tr4096_va2048.npz")
hand=np.load("learned_fn_handoff_8x8.npz")
states=base["train_states"].astype(np.uint64)
r0=base["rtrain"].astype(float)
w0=base["iwtrain"].astype(float); w0/=w0.sum()
gcen=hand["train_g"].astype(float)

lat=SquareJ1J2(8,.5)
ci=[];ns=[];cf=[];diag=np.empty(len(states),float)
for k,s0 in enumerate(states):
    s=int(s0);diag[k]=lat.diag_energy(s)
    for y,J,mr in lat.neigh(s):
        ci.append(k);ns.append(y);cf.append(.5*J*mr)
ci=np.asarray(ci,np.int32);ns=np.asarray(ns,np.uint64);cf=np.asarray(cf,float)
allst=np.concatenate([states,ns]);uniq,inv=np.unique(allst,return_inverse=True)
ic=inv[:len(states)];inn=inv[len(states):]
print("ETA_SCAN_GRAPH","centers",len(states),"edges",len(ns),"unique_states",len(uniq),flush=True)

b=load("fn_residual_cnn_8x8.mpack")
la_prod=b.log_amplitude_bits(uniq)
spin=2*(((uniq.reshape(-1,1)>>np.arange(64,dtype=np.uint64))&1).astype(np.float32))-1
spin=spin.reshape(-1,8,8,1)
raw=[]
for i in range(0,len(spin),4096): raw.append(np.asarray(b.res_apply(spin[i:i+4096])))
g=np.concatenate(raw).astype(float)
la0=la_prod-b.eta*g
dg=g[inn]-g[ic[ci]]
print("ETA_SCAN_GDIFF",np.quantile(dg,[0,.001,.01,.1,.5,.9,.99,.999,1]).tolist(),flush=True)

rows=[]
base_labels=None
for eta in ETAS:
    la=la0+eta*g
    term=cf*np.exp(la[inn]-la[ic[ci]])
    r=diag+np.bincount(ci,weights=term,minlength=len(states))
    w=w0*np.exp(2*eta*gcen);w/=w.sum()
    T,info=robust_two_means_threshold(r,w,.10,.90)
    labels=(r<=T)
    if eta==0.0:
        base_labels=labels.copy()
        reg=float(np.max(np.abs(r-r0)))
        print("ETA0_REGRESSION","T",T,"T0",T0,"dT",T-T0,"max_dr",reg,flush=True)
    flip=base_labels!=labels
    row=dict(eta=float(eta),T=float(T),flip_frac=float(flip.mean()),
             flip_mass_old=float(np.sum(w0*flip)),flip_mass_new=float(np.sum(w*flip)),
             ess=float(1/np.sum(w*w)))
    rows.append(row);print("ETA_SCAN",json.dumps(row),flush=True)
with open("eta_trust_scan_8x8.json","w") as f:json.dump({"rows":rows},f,indent=2)
