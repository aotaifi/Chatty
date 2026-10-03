import os, json, numpy as np
BASE="/Users/aliotaifi/j1j2_vit_bench"
os.chdir(BASE)

src=open("true_k2_6x6.py").read().split("d=np.load('krylov_scaling_6x6_a1.20_tr4096_va2048.npz')")[0]
exec(src)

d=np.load("/Users/aliotaifi/Chatty/projects/j1j2/results/true_k2_6x6_3471990/true_k2_6x6.npz")
with open("krylov_scaling_6x6_a1.20_tr4096_va2048.json") as f:
    meta=json.load(f)
phi=float(meta["phi"])

def hidden_for(states):
    ensure_z(states)
    return np.asarray([1 if np.cos(zc[int(x)].imag-phi)>=0 else -1 for x in states],np.int8)

def mar_arr(states):
    return np.asarray([mar(x) for x in states],np.int8)
def oracle_threshold(score, base_sign, target_sign, w):
    score=np.asarray(score,float); base=np.asarray(base_sign,float)
    target=np.asarray(target_sign,float); w=np.asarray(w,float); w=w/w.sum()
    order=np.argsort(score)
    z=w*base*target
    # t below min: all q=-1; then flip one-by-one to q=+1 in score order.
    corr=-float(z.sum())
    best=(corr, float(score[order[0]]-1.0), 1.0)
    prefix=0.0
    for k,ix in enumerate(order):
        prefix += 2.0*float(z[ix])
        corr_k = corr + prefix
        t = float(score[ix]) if k==len(order)-1 else float(0.5*(score[ix]+score[order[k+1]]))
        flipmass=float(w[score>t].sum())
        if corr_k>best[0]:
            best=(corr_k,t,flipmass)
    # t above max = no flip is included at final k.
    return best

rows=[]
for name,states,r0,r1,w in [
    ("train",d["train_states"],d["r0train"],d["r1train"],np.load("krylov_scaling_6x6_a1.20_tr4096_va2048.npz")["iwtrain"]),
    ("val",d["val_states"],d["r0val"],d["r1val"],np.load("krylov_scaling_6x6_a1.20_tr4096_va2048.npz")["iwval"])]:
    w=np.asarray(w,float); w/=w.sum()
    h=hidden_for(states); m=mar_arr(states)
    q1=np.where(T1-r0>=0,1,-1)
    s1=m*q1
    C0=float(np.sum(w*m*h)); C1=float(np.sum(w*s1*h))
    k1=oracle_threshold(r0,m,h,w)
    k2=oracle_threshold(r1,s1,h,w)
    print(name,"C_M",C0,"C_K1",C1,"K1_ORACLE",k1,"K2_ORACLE",k2,flush=True)
    rows.append([C0,C1,*k1,*k2])
out="/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/results/k2_oracle_information_audit_6x6.npz"
np.savez_compressed(out,rows=np.asarray(rows,float))
print("SAVED",out,flush=True)
