import os,sys,math,json,numpy as np
sys.path.insert(0,"/Users/aliotaifi/Chatty/projects/j1j2/experiments")
from learned_fn8_callable_amplitude_adapter import load
from callable_fn_loop import robust_two_means_threshold,SquareJ1J2
BASE="/Users/aliotaifi/Chatty/projects/j1j2/results/8x8_krylov_3471544/krylov_scaling_8x8_a1.20_tr4096_va2048.npz"
HAND="/Users/aliotaifi/Chatty/projects/j1j2/results/learned_fn_handoff_8x8.npz"
CK="/Users/aliotaifi/Chatty/projects/j1j2/results/fn_residual_cnn_8x8.mpack"
T0=-28.37107876288694
ETAS=[0.,.005,.01,.02,.05,.1,.2,.35966756939888]
z=np.load(BASE); h=np.load(HAND)
states=z["train_states"][:1024].astype(np.uint64)
r0=z["rtrain"][:1024].astype(float)
w0=z["iwtrain"][:1024].astype(float); w0/=w0.sum()
gcen=h["train_g"][:1024].astype(float)
lat=SquareJ1J2(8,.5)
ci=[]; ns=[]; cf=[]; diag=np.empty(len(states))
for k,s0 in enumerate(states):
    s=int(s0);diag[k]=lat.diag_energy(s)
    for y,J,mr in lat.neigh(s):
        ci.append(k);ns.append(y);cf.append(.5*J*mr)
ci=np.asarray(ci,np.int32);ns=np.asarray(ns,np.uint64);cf=np.asarray(cf,float)
allst=np.concatenate([states,ns]);uniq,inv=np.unique(allst,return_inverse=True)
ic=inv[:len(states)];inn=inv[len(states):]
print("QUICK_GRAPH",len(states),len(ns),len(uniq),flush=True)
b=load(CK)
la_prod=b.log_amplitude_bits(uniq)
spin=2*(((uniq.reshape(-1,1)>>np.arange(64,dtype=np.uint64))&1).astype(np.float32))-1
spin=spin.reshape(-1,8,8,1)
raw=[]
for i in range(0,len(spin),4096):raw.append(np.asarray(b.res_apply(spin[i:i+4096])))
g=np.concatenate(raw).astype(float)
la0=la_prod-b.eta*g
rows=[]
for eta in ETAS:
    la=la0+eta*g
    term=cf*np.exp(la[inn]-la[ic[ci]])
    r=diag+np.bincount(ci,weights=term,minlength=len(states))
    w=w0*np.exp(2*eta*gcen);w/=w.sum()
    T,_=robust_two_means_threshold(r,w,.10,.90)
    flip=(r0<=T0)!=(r<=T)
    row=(eta,float(T),float(flip.mean()),float(w0@flip),float(w@flip),float(1/(w@w)))
    rows.append(row)
    print("QUICK_ETA",*row,flush=True)
json.dump({"rows":[dict(eta=a,T=t,raw=f,old_mass=om,new_mass=nm,ess=e) for a,t,f,om,nm,e in rows]},open("/Users/aliotaifi/Chatty/projects/j1j2/results/eta_trust_scan_8x8_quick.json","w"),indent=2)
