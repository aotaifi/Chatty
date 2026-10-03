import os,json,numpy as np
BASE="/Users/aliotaifi/j1j2_vit_bench"; os.chdir(BASE)
src=open("true_k2_6x6.py").read().split("d=np.load('krylov_scaling_6x6_a1.20_tr4096_va2048.npz')")[0]
exec(src)
d=np.load("/Users/aliotaifi/Chatty/projects/j1j2/results/true_k2_6x6_3471990/true_k2_6x6.npz")
b=np.load("krylov_scaling_6x6_a1.20_tr4096_va2048.npz")
with open("krylov_scaling_6x6_a1.20_tr4096_va2048.json") as f: phi=float(json.load(f)["phi"])
def prep(states,r0,r1,w):
    w=np.asarray(w,float);w/=w.sum();ensure_z(states)
    h=np.asarray([1 if np.cos(zc[int(x)].imag-phi)>=0 else -1 for x in states])
    m=np.asarray([mar(x) for x in states]);q1=np.where(T1-r0>=0,1,-1);s1=m*q1
    wrong=(s1*h)<0
    return dict(r0=np.asarray(r0),r1=np.asarray(r1),w=w,wrong=wrong,margin=np.abs(T1-np.asarray(r0)))
TR=prep(d["train_states"],d["r0train"],d["r1train"],b["iwtrain"])
VA=prep(d["val_states"],d["r0val"],d["r1val"],b["iwval"])
def report(name,P,t):
    sel=P["r1"]>t; w=P["w"]; wrong=P["wrong"]; mar=P["margin"]
    sm=float(w[sel].sum()); wm=float(w[wrong].sum()); hit=float(w[sel&wrong].sum())
    prec=hit/sm if sm else 0.; rec=hit/wm if wm else 0.
    print(name,"t",t,"nsel",int(sel.sum()),"selmass",sm,"wrongmass",wm,
          "hitmass",hit,"precision",prec,"recall",rec,flush=True)
    print(name,"margin_all_q",np.quantile(mar,[.001,.01,.1,.5,.9,.99]).tolist(),flush=True)
    print(name,"margin_sel_q",np.quantile(mar[sel],[0,.1,.25,.5,.75,.9,1]).tolist(),flush=True)
    print(name,"margin_wrong_q",np.quantile(mar[wrong],[0,.1,.25,.5,.75,.9,1]).tolist(),flush=True)
    print(name,"corr_log_inv_margin_r1",float(np.corrcoef(-np.log(mar),P["r1"])[0,1]),flush=True)
report("TRAIN_oracleT",TR,1.8568338884780973)
report("VAL_trainT",VA,1.8568338884780973)
report("VAL_oracleT",VA,-4.2752692229842175)
