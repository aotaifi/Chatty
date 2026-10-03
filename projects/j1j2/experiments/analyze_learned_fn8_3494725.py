import numpy as np
root="/pc2/users/h/hpcalot/chatty/j1j2/results/learned_fn8_3494725"
d=np.load(root+"/learned_fn8_k1fn_seed12001.npz")
b=np.load(root+"/krylov_scaling_8x8_a1.20_tr4096_va2048.npz")
h=np.load(root+"/learned_fn_handoff_8x8.npz")
ro=np.r_[b["rtrain"],b["rval"]]
rn=d["threshold_r"]
w=h["threshold_weights"]
T0=-28.37107876288694
T1=float(d["threshold"])
fo=ro<=T0
fn=rn<=T1
print("T1",T1,"T0",T0)
print("node_change_w",np.sum(w*(fo!=fn))/np.sum(w))
print("old_flip_w",np.sum(w*fo)/np.sum(w),"new_flip_w",np.sum(w*fn)/np.sum(w))
print("train_change",np.sum(w[:4096]*(fo[:4096]!=fn[:4096]))/np.sum(w[:4096]))
print("val_change",np.sum(w[4096:]*(fo[4096:]!=fn[4096:]))/np.sum(w[4096:]))
print("rn_quant",np.quantile(rn,[0,.001,.01,.1,.5,.9,.99,.999,1]))
print("ro_quant",np.quantile(ro,[0,.001,.01,.1,.5,.9,.99,.999,1]))
print("corr",np.corrcoef(ro,rn)[0,1])
e=[]; tails=[]
for s in (12001,12002):
    z=np.load(root+f"/learned_fn8_k1fn_seed{s}.npz")
    e.append(float(z["Es"].mean()))
    tails.append(float(z["Es"][-8:].mean()))
print("Ereps",e,"mean",np.mean(e),"between_SE",np.std(e,ddof=1)/np.sqrt(2))
print("tail8",tails,"tailmean",np.mean(tails),"tailSE",np.std(tails,ddof=1)/np.sqrt(2))
print("delta_vs_base",np.mean(e)-(-31.8854241096))
