import json, csv, hashlib, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

L=4; N=16
CASES=[(0.5,"marshall"),(0.6,"marshall"),(1.0,"stripe_x")]
MAXITER=12
OUT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/results")
site=lambda x,y:(x%L)+L*(y%L)

NN=[]; NNN=[]
for y in range(L):
    for x in range(L):
        i=site(x,y)
        NN += [(i,site(x+1,y)),(i,site(x,y+1))]
        NNN += [(i,site(x+1,y+1)),(i,site(x+1,y-1))]

basis=np.array([s for s in range(1<<N) if s.bit_count()==N//2],np.uint32)
D=len(basis); pos={int(s):i for i,s in enumerate(basis)}

def build_H(J2):
    rr=[];cc=[];vv=[];diag=np.zeros(D)
    for bi,s0 in enumerate(basis):
        s=int(s0); e=0.
        for bonds,J in ((NN,1.0),(NNN,J2)):
            if J==0: continue
            for u,v in bonds:
                if ((s>>u)&1)==((s>>v)&1): e+=0.25*J
                else:
                    e-=0.25*J
                    t=s^(1<<u)^(1<<v)
                    rr.append(bi);cc.append(pos[t]);vv.append(0.5*J)
        diag[bi]=e
    rr.extend(range(D));cc.extend(range(D));vv.extend(diag.tolist())
    return sp.coo_matrix((vv,(rr,cc)),shape=(D,D)).tocsr(),diag

def mask(kind):
    if kind=="marshall":
        return sum(1<<site(x,y) for y in range(L) for x in range(L) if (x+y)%2==0)
    if kind=="stripe_x":
        return sum(1<<site(x,y) for y in range(L) for x in range(L) if x%2==0)
    raise ValueError(kind)

def gauge(kind):
    m=mask(kind)
    return np.array([1 if ((int(s)&m).bit_count()%2)==0 else -1 for s in basis],np.int8)

def canonical(s,anchor):
    s=np.asarray(s,np.int8).copy()
    if s[anchor]<0: s=-s
    return s

def group_sorted_values(r,atol=1e-10,rtol=1e-10):
    order=np.argsort(r,kind="mergesort")
    groups=[]; cur=[int(order[0])]; ref=float(r[order[0]])
    for jj in order[1:]:
        v=float(r[jj])
        if abs(v-ref) <= atol + rtol*max(abs(v),abs(ref),1.0):
            cur.append(int(jj))
        else:
            groups.append(np.array(cur,dtype=np.int32))
            cur=[int(jj)]; ref=v
    groups.append(np.array(cur,dtype=np.int32))
    return groups

def fixed_energy(H,a,s):
    psi=a*s
    return float(psi@(H@psi))

def overlap(p,s,t):
    return float(abs(np.sum(p*s*t)))

def pstable(H,diag,a,s,p):
    psi=a*s
    h=(H@psi)-diag*psi
    return float(np.sum(p*((s*h)<=1e-12)))

def energy_opt_update(H,diag,a,s,r,anchor):
    # Exact minimization over the realizable threshold family.
    W=sp.diags(a)@(H-sp.diags(diag))@sp.diags(a)
    diagE=float(np.sum(diag*a*a))
    cand=(-s).astype(np.int8)  # threshold below all r
    E=diagE+float(cand@(W@cand))
    h=np.asarray(W@cand).ravel()
    groups=group_sorted_values(r)
    bestE=float(E); best_s=cand.copy(); best_k=-1
    for k,g in enumerate(groups):
        olds=cand[g].copy()
        # Flip the whole exactly/near-degenerate r group simultaneously.
        for ii,old in zip(g,olds):
            E += -4.0*float(old)*h[ii]
            cand[ii]=-old
            st,en=W.indptr[ii],W.indptr[ii+1]
            js=W.indices[st:en]; ws=W.data[st:en]
            h[js] += -2.0*float(old)*ws
        # Recompute exact energy to remove sequential-cross-term bookkeeping error inside groups.
        Eexact=diagE+float(cand@(W@cand))
        E=Eexact
        h=np.asarray(W@cand).ravel()
        if Eexact < bestE-1e-12:
            bestE=float(Eexact);best_s=cand.copy();best_k=k
    best_s=canonical(best_s,anchor)
    if best_k<0:
        thr=float(np.min(r)-max(1.0,0.01*abs(np.min(r))))
    elif best_k==len(groups)-1:
        thr=float(np.max(r)+max(1.0,0.01*abs(np.max(r))))
    else:
        lo=float(np.max(r[groups[best_k]])); hi=float(np.min(r[groups[best_k+1]]))
        thr=0.5*(lo+hi)
    return best_s,thr,bestE,len(groups)

def oracle_overlap_update(a,p,truth,s,r,anchor):
    groups=group_sorted_values(r)
    cand=(-s).astype(np.int8)
    bestO=overlap(p,cand,truth);best_s=cand.copy();best_k=-1
    for k,g in enumerate(groups):
        cand[g]*=-1
        O=overlap(p,cand,truth)
        if O>bestO+1e-14:
            bestO=O;best_s=cand.copy();best_k=k
    return canonical(best_s,anchor),float(bestO),best_k

def key(s):
    c=np.packbits((s>0).astype(np.uint8))
    return hashlib.sha1(c.tobytes()).hexdigest()[:12]

allrows=[]; summaries=[]
for J2,base in CASES:
    print("CASE",J2,base,flush=True)
    H,diag=build_H(J2)
    ev,V=sla.eigsh(H,k=6,which="SA",tol=1e-12,maxiter=100000)
    o=np.argsort(ev);ev=ev[o];V=V[:,o]
    E0=float(ev[0]);vec=np.asarray(V[:,0],float)
    anchor=int(np.argmax(np.abs(vec)))
    if vec[anchor]<0:vec=-vec
    a=np.abs(vec);p=a*a;p/=p.sum();truth=np.where(vec>=0,1,-1).astype(np.int8)
    s=canonical(gauge(base),anchor)
    seen={}
    trajectory=[]
    for it in range(MAXITER+1):
        E=fixed_energy(H,a,s)
        row=dict(J2=J2,baseline=base,iteration=it,O_S=overlap(p,s,truth),
                 energy_error_per_site=(E-E0)/N,P_stable=pstable(H,diag,a,s,p),
                 state_key=key(s))
        trajectory.append(row);allrows.append(row)
        print("ITER",J2,base,it,"O",row["O_S"],"err",row["energy_error_per_site"],
              "Pstable",row["P_stable"],"key",row["state_key"],flush=True)
        if it==MAXITER: break
        if row["state_key"] in seen:
            print("CYCLE",J2,base,"at",it,"first",seen[row["state_key"]],flush=True)
            break
        seen[row["state_key"]]=it
        psi=a*s
        r=(H@psi)/np.where(np.abs(psi)>1e-300,psi,1e-300)
        snew,t,Ebest,ng=energy_opt_update(H,diag,a,s,r,anchor)
        s_or,O_or,_=oracle_overlap_update(a,p,truth,s,r,anchor)
        changed=min(float(np.sum(p*(snew!=s))),float(np.sum(p*((-snew)!=s))))
        row["threshold_to_next"]=float(t);row["changed_weight_to_next"]=changed
        row["oracle_next_O"]=O_or;row["n_r_groups"]=int(ng)
        if np.array_equal(snew,s):
            print("FIXED_POINT",J2,base,"at",it,"oracle_next_O",O_or,flush=True)
            break
        s=snew
    final=trajectory[-1]
    summaries.append(dict(J2=J2,baseline=base,E0=E0,spectrum=ev[:6].tolist(),
                          iterations_reached=final["iteration"],final_O=final["O_S"],
                          final_energy_error_per_site=final["energy_error_per_site"],
                          final_P_stable=final["P_stable"],
                          exact_signs=bool(final["O_S"]>1-1e-12)))
    print("SUMMARY",json.dumps(summaries[-1]),flush=True)

with open(OUT/"iterated_krylov_exact.json","w") as f:
    json.dump(dict(method={
      "system":"4x4 periodic square J1-J2, Sz=0 exact diagonalization",
      "amplitudes":"held fixed to exact ground-state |psi| throughout",
      "update":"r_k=(H a s_k)/(a s_k); s_{k+1}=s_k sign(t_k-r_k)",
      "threshold":"label-free exact minimization of fixed-amplitude energy over all realizable r_k threshold groups",
      "degeneracies":"equal/near-equal r_k values are flipped as a group; no threshold can split them",
      "oracle_next_O":"diagnostic only: best hidden-sign-overlap threshold on same r_k coordinate"
    },summaries=summaries,trajectory=allrows),f,indent=2)

fields=["J2","baseline","iteration","O_S","energy_error_per_site","P_stable",
        "threshold_to_next","changed_weight_to_next","oracle_next_O","n_r_groups","state_key"]
with open(OUT/"iterated_krylov_exact.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader()
    for q in allrows:w.writerow(q)
print("DONE",flush=True)
