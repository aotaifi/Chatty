import numpy as np
SRC="/Users/aliotaifi/j1j2_vit_bench/fn_krylov_loop_4x4_kmeans_a12.py"
src=open(SRC).read().split('report_direct("exact_abs",a0)')[0]
exec(src)

def amp_overlap(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    a=a/np.linalg.norm(a);b=b/np.linalg.norm(b)
    return float(abs(np.dot(a,b)))

def logratio_rms(a,b,w=None):
    la=np.log(np.maximum(a,1e-300));lb=np.log(np.maximum(b,1e-300))
    d=la-lb
    if w is None:w=np.ones(len(d))/len(d)
    else:w=np.asarray(w,float);w=w/w.sum()
    d-=np.sum(w*d)
    return float(np.sqrt(np.sum(w*d*d)))

def run_half(label,a_init,s_init,niter=12):
    a=np.asarray(a_init,float).copy();a/=np.linalg.norm(a)
    s=np.asarray(s_init,float).copy()
    print("HALF_START",label,"E",energy(a,s),"O",sign_overlap(s),flush=True)
    for it in range(niter):
        Efn,afn,mis=fn_solve(a,s)
        # MLE optimum on mixed walkers f ~ a*afn when model probability is a_new^2.
        ah=np.sqrt(np.maximum(a*afn,1e-300));ah/=np.linalg.norm(ah)
        sh,t,cent,_=krylov_sign(ah)
        sf,tf,centf,_=krylov_sign(afn)
        print("HALF_ITER",label,it,
              "guideE",energy(a,s),"Efn",Efn,
              "halfE",energy(ah,sh),"fullE",energy(afn,sf),
              "halfO",sign_overlap(sh),"fullO",sign_overlap(sf),
              "half_to_FN_ov",amp_overlap(ah,afn),
              "old_to_FN_ov",amp_overlap(a,afn),
              "half_logdist_FN",logratio_rms(ah,afn,afn*afn),
              "old_logdist_FN",logratio_rms(a,afn,afn*afn),
              "half_sign_change_true",float(np.sum(ptr*(sh!=s))),
              "full_sign_change_true",float(np.sum(ptr*(sf!=s))),
              "T_half",t,"T_full",tf,flush=True)
        a,s=ah,sh
    print("HALF_END",label,"E",energy(a,s),"O",sign_overlap(s),flush=True)

run_half("exactAbs_Marshall",a0,sM,12)
run_half("uniform_Marshall",np.ones(D)/np.sqrt(D),sM,12)
