import os,json,numpy as np
BASE="/Users/aliotaifi/j1j2_vit_bench"; os.chdir(BASE)
src=open("true_k2_6x6.py").read().split("d=np.load('krylov_scaling_6x6_a1.20_tr4096_va2048.npz')")[0]
exec(src)
d=np.load("/Users/aliotaifi/Chatty/projects/j1j2/results/true_k2_6x6_3471990/true_k2_6x6.npz")
b=np.load("krylov_scaling_6x6_a1.20_tr4096_va2048.npz")
with open("krylov_scaling_6x6_a1.20_tr4096_va2048.json") as f: phi=float(json.load(f)["phi"])
def pack(states,r0,r1,w):
    w=np.asarray(w,float); w/=w.sum(); ensure_z(states)
    h=np.asarray([1 if np.cos(zc[int(x)].imag-phi)>=0 else -1 for x in states])
    m=np.asarray([mar(x) for x in states]); q1=np.where(T1-r0>=0,1,-1); s1=m*q1
    return dict(states=states,r1=np.asarray(r1),w=w,h=h,s1=s1,C1=float(np.sum(w*s1*h)))
TR=pack(d["train_states"],d["r0train"],d["r1train"],b["iwtrain"])
VA=pack(d["val_states"],d["r0val"],d["r1val"],b["iwval"])
def corr_at_t(P,t):
    q=np.where(P["r1"]<=t,1,-1)
    return float(np.sum(P["w"]*P["s1"]*q*P["h"])),float(P["w"][P["r1"]>t].sum()),int(np.sum(P["r1"]>t))
def threshold_for_tail(P,tail):
    o=np.argsort(P["r1"])[::-1]; c=np.cumsum(P["w"][o])
    k=int(np.searchsorted(c,tail,side="left"))
    k=min(k,len(o)-1)
    if k==len(o)-1:return float(P["r1"][o[k]]-1)
    return float(.5*(P["r1"][o[k]]+P["r1"][o[k+1]]))
for name,P in [("train",TR),("val",VA)]:
    print(name,"C1",P["C1"],"r1max",float(P["r1"].max()),"r1q",np.quantile(P["r1"],[.99,.995,.999,1]).tolist(),flush=True)
for tail in [1e-4,2e-4,3e-4,5e-4,7e-4,1e-3,2e-3,5e-3,1e-2]:
    tt=threshold_for_tail(TR,tail); tv=threshold_for_tail(VA,tail)
    print("TAIL",tail,"t_train",tt,"train",corr_at_t(TR,tt),"val_at_trainT",corr_at_t(VA,tt),
          "t_val",tv,"val",corr_at_t(VA,tv),"train_at_valT",corr_at_t(TR,tv),flush=True)
