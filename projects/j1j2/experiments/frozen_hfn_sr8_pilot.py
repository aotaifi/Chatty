#!/usr/bin/env python3
import argparse, json, math, time
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax
import netket as nk
import netket.jax as nkjax
from jax.flatten_util import ravel_pytree
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d
from callable_fn_loop import CachedAmplitude, SquareJ1J2, K1FNEngine

N=64; L=8
T1=-28.824166903057147

def bits2x(ss):
    a=np.asarray(ss,np.uint64).reshape(-1,1)
    return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float64)-1

def spins2bits(x):
    x=np.asarray(x).reshape(-1,N)
    bb=(x>0).astype(np.uint64)
    pw=(np.uint64(1)<<np.arange(N,dtype=np.uint64))
    return np.sum(bb*pw[None,:],axis=1,dtype=np.uint64)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a1",required=True)
    ap.add_argument("--out-ck",required=True)
    ap.add_argument("--out-json",required=True)
    ap.add_argument("--edge-cache",default=None)
    ap.add_argument("--nsamp",type=int,default=512)
    ap.add_argument("--steps",type=int,default=5)
    ap.add_argument("--shift",type=float,default=0.1)
    ap.add_argument("--trust",type=float,default=0.002)
    ap.add_argument("--seed-train",type=int,default=31001)
    ap.add_argument("--seed-val1",type=int,default=31002)
    ap.add_argument("--seed-val2",type=int,default=31003)
    args=ap.parse_args()

    model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,
              transl_invariant=True,two_dimensional=True)
    apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
    template=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N)))
    obj=flax.serialization.msgpack_restore(open(args.a1,"rb").read())
    variables=flax.serialization.from_state_dict(template,obj)
    p_ref=variables["params"]
    flat,unravel=ravel_pytree(p_ref)
    flat_ref=flat

    def eval_real_flat(ff,states,batch=1024):
        p=unravel(ff); X=bits2x(states); out=[]
        for i in range(0,len(X),batch):
            z=apply({"params":p},jnp.asarray(X[i:i+batch]))
            out.append(np.asarray(z).real)
        return np.concatenate(out)

    def eval_real_ref(states,batch=1024):
        return eval_real_flat(flat_ref,states,batch=batch)

    # Fresh samples from the validated a1^2 guide. Three independent seeds.
    hi=nk.hilbert.Spin(s=.5,N=N,total_sz=0)
    graph=nk.graph.Hypercube(length=L,n_dim=2,pbc=True,max_neighbor_order=2)
    sampler=nk.sampler.MetropolisExchange(hi,graph=graph,d_max=2,n_chains=32,sweep_size=N)
    def draw(seed):
        v=nk.vqs.MCState(sampler=sampler,apply_fun=apply,n_samples=args.nsamp,
                         variables=variables,n_discard_per_chain=20,seed=seed)
        xx=np.asarray(v.sample()).reshape(-1,N)
        return spins2bits(xx)
    print("SAMPLE train",flush=True); train=draw(args.seed_train)
    print("SAMPLE val1",flush=True); val1=draw(args.seed_val1)
    print("SAMPLE val2",flush=True); val2=draw(args.seed_val2)
    print("SAMPLE_UNIQ",len(np.unique(train)),len(np.unique(val1)),len(np.unique(val2)),flush=True)

    amp1=CachedAmplitude(eval_real_ref)
    lat=SquareJ1J2(L,.5)
    signs=K1FNEngine(lat,amp1); signs.threshold=T1

    # Freeze H_FN[a1,s1]. For each base x, store frozen diagonal and allowed
    # off-diagonal rates evaluated with a1; for any trial b=a1*exp(delta),
    # E_L^F(x;b)=D_F(x)-sum_allowed rate_a1(x,y)*exp(delta_y-delta_x).
    def build_data(base,label):
        base=np.asarray(base,np.uint64)
        print("EDGE_BUILD",label,"n",len(base),flush=True)
        t=time.time(); signs.ensure_local(base)
        diag=[]; child=[]; owner=[]; rate=[]
        for i,x in enumerate(base):
            d,e,ys,rr=signs.local_cache[int(x)]
            diag.append(d)
            child.extend(ys.tolist()); owner.extend([i]*len(ys)); rate.extend(rr.tolist())
        child=np.asarray(child,np.uint64); owner=np.asarray(owner,np.int32)
        rate=np.asarray(rate,float); diag=np.asarray(diag,float)
        amp1.ensure(np.concatenate([base,child]))
        l1b=np.asarray([amp1.loga(x) for x in base],float)
        l1c=np.asarray([amp1.loga(x) for x in child],float)
        print("EDGE_DONE",label,"edges",len(child),"amp_eval",amp1.neval,"sec",time.time()-t,flush=True)
        return dict(base=base,diag=diag,child=child,owner=owner,rate=rate,l1b=l1b,l1c=l1c)

    dtr=build_data(train,"train")
    dv1=build_data(val1,"val1")
    dv2=build_data(val2,"val2")
    if args.edge_cache:
        np.savez_compressed(args.edge_cache,
            train=train,val1=val1,val2=val2,
            tr_diag=dtr["diag"],tr_child=dtr["child"],tr_owner=dtr["owner"],tr_rate=dtr["rate"],tr_l1b=dtr["l1b"],tr_l1c=dtr["l1c"],
            v1_diag=dv1["diag"],v1_child=dv1["child"],v1_owner=dv1["owner"],v1_rate=dv1["rate"],v1_l1b=dv1["l1b"],v1_l1c=dv1["l1c"],
            v2_diag=dv2["diag"],v2_child=dv2["child"],v2_owner=dv2["owner"],v2_rate=dv2["rate"],v2_l1b=dv2["l1b"],v2_l1c=dv2["l1c"])

    def local_energy(ff,d):
        lb=eval_real_flat(ff,d["base"])
        lc=eval_real_flat(ff,d["child"])
        db=lb-d["l1b"]; dc=lc-d["l1c"]
        er=d["rate"]*np.exp(np.clip(dc-db[d["owner"]],-40,40))
        off=np.bincount(d["owner"],weights=er,minlength=len(d["base"]))
        return d["diag"]-off,lb

    def stats(ff,d):
        E,lb=local_energy(ff,d)
        dl=lb-d["l1b"]
        z=2*(dl-np.max(dl))
        w=np.exp(z); w/=w.sum()
        mean=float(w@E)
        var=float(w@((E-mean)**2))
        ess=float(1/(w@w))
        return dict(E=mean,var=var,ESS=ess,dlog_rms=float(np.sqrt(w@((dl-w@dl)**2))),Evec=E,w=w,lb=lb)

    base_tr=stats(flat,dtr); base_v1=stats(flat,dv1); base_v2=stats(flat,dv2)
    print("BASE",json.dumps({k:{"E":v["E"],"var":v["var"],"ESS":v["ESS"]} for k,v in [("tr",base_tr),("v1",base_v1),("v2",base_v2)]}),flush=True)

    Xtr=jnp.asarray(bits2x(train))
    def logvec(ff):
        return jnp.real(apply({"params":unravel(ff)},Xtr))

    rows=[]
    accepted_steps=0
    for step in range(1,args.steps+1):
        st=stats(flat,dtr)
        E=st["Evec"]; w=st["w"]; Ebar=st["E"]
        coeff=jnp.asarray(2*w*(E-Ebar))
        def surrogate(ff):
            return jnp.dot(coeff,logvec(ff))
        _,g=jax.value_and_grad(surrogate)(flat)

        # weighted matrix-free QGT on current importance-reweighted train measure
        wj=jnp.asarray(w)
        _,pullback=jax.vjp(logvec,flat)
        def fisher(v):
            _,u=jax.jvp(logvec,(flat,),(v,))
            uc=u-jnp.sum(wj*u)
            return pullback(wj*uc)[0]
        fisher=jax.jit(fisher)
        _=fisher(jnp.zeros_like(flat)).block_until_ready()
        def A(v): return fisher(v)+args.shift*v
        sol,info=jax.scipy.sparse.linalg.cg(A,g,tol=1e-5,atol=0.0,maxiter=250)
        sol.block_until_ready()
        resid=float(jnp.linalg.norm(A(sol)-g)/jnp.linalg.norm(g))
        d=-sol
        _,tan=jax.jvp(logvec,(flat,),(d,))
        tan=np.asarray(tan,float); tc=tan-float(w@tan)
        rms=float(np.sqrt(w@(tc*tc)))
        eta=args.trust/max(rms,1e-300)

        before=[stats(flat,z) for z in (dtr,dv1,dv2)]
        cand_rows=[]; chosen=None
        for fac in (1.0,.5,.25,.125,.0625,.03125):
            ff=flat+eta*fac*d
            aft=[stats(ff,z) for z in (dtr,dv1,dv2)]
            ok=(all(a["E"]<b["E"] for a,b in zip(aft,before)) and
                all(a["ESS"]>0.8*args.nsamp for a in aft))
            row={"factor":fac,
                 "train_delta":aft[0]["E"]-before[0]["E"],
                 "val1_delta":aft[1]["E"]-before[1]["E"],
                 "val2_delta":aft[2]["E"]-before[2]["E"],
                 "train_ESS":aft[0]["ESS"],"val1_ESS":aft[1]["ESS"],"val2_ESS":aft[2]["ESS"],
                 "train_dlog_rms":aft[0]["dlog_rms"],"val1_dlog_rms":aft[1]["dlog_rms"],"val2_dlog_rms":aft[2]["dlog_rms"],
                 "pass":bool(ok)}
            cand_rows.append(row)
            print("LINE",step,json.dumps(row),flush=True)
            if ok:
                chosen=(ff,aft,row); break
        if chosen is None:
            rows.append({"step":step,"accepted":False,"cg_relres":resid,"tangent_rms_per_eta":rms,"eta":float(eta),"candidates":cand_rows})
            print("STOP no transferable step",step,flush=True)
            break
        flat,aft,crow=chosen; accepted_steps+=1
        row={"step":step,"accepted":True,"shift":args.shift,"trust":args.trust,
             "cg_relres":resid,"tangent_rms_per_eta":rms,"eta":float(eta),
             "chosen":crow,
             "train_E":aft[0]["E"],"val1_E":aft[1]["E"],"val2_E":aft[2]["E"]}
        rows.append(row); print("ACCEPT",json.dumps(row),flush=True)

    final_tr=stats(flat,dtr); final_v1=stats(flat,dv1); final_v2=stats(flat,dv2)
    out={"nsamp":args.nsamp,"steps_requested":args.steps,"steps_accepted":accepted_steps,
         "shift":args.shift,"trust":args.trust,"threshold":T1,
         "base":{"train_E":base_tr["E"],"val1_E":base_v1["E"],"val2_E":base_v2["E"]},
         "final":{"train_E":final_tr["E"],"val1_E":final_v1["E"],"val2_E":final_v2["E"],
                  "train_delta":final_tr["E"]-base_tr["E"],
                  "val1_delta":final_v1["E"]-base_v1["E"],
                  "val2_delta":final_v2["E"]-base_v2["E"],
                  "train_ESS":final_tr["ESS"],"val1_ESS":final_v1["ESS"],"val2_ESS":final_v2["ESS"],
                  "train_dlog_rms":final_tr["dlog_rms"],"val1_dlog_rms":final_v1["dlog_rms"],"val2_dlog_rms":final_v2["dlog_rms"]},
         "amp1_eval_for_frozen_operator":amp1.neval,"rows":rows,
         "pass":bool(accepted_steps>0 and final_v1["E"]<base_v1["E"] and final_v2["E"]<base_v2["E"])}
    json.dump(out,open(args.out_json,"w"),indent=2)
    print("RESULT",json.dumps({k:v for k,v in out.items() if k!="rows"},sort_keys=True),flush=True)
    if out["pass"]:
        vnew=dict(variables); vnew["params"]=unravel(flat)
        open(args.out_ck,"wb").write(flax.serialization.to_bytes(vnew))
        print("SAVED",args.out_ck,flush=True)
    else:
        print("REJECTED",flush=True)

if __name__=="__main__":
    main()
