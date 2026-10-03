#!/usr/bin/env python3
import argparse, gc, json, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import flax
import netket.jax as nkjax
from jax.flatten_util import ravel_pytree
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d
from callable_fn_loop import SquareJ1J2

jax.config.update("jax_enable_x64", True)
N=64
T1=-28.824166903057147

def bits2x(ss):
    a=np.asarray(ss,np.uint64).reshape(-1,1)
    return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float64)-1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a1",required=True)
    ap.add_argument("--handoff1",required=True)
    ap.add_argument("--out-ck",required=True)
    ap.add_argument("--out-json",required=True)
    ap.add_argument("--out-freeze",required=True)
    ap.add_argument("--ntrain",type=int,default=256)
    ap.add_argument("--nval",type=int,default=256)
    ap.add_argument("--fisher-n",type=int,default=256)
    ap.add_argument("--threshold",type=float,default=T1)
    ap.add_argument("--eval-batch",type=int,default=4096)
    args=ap.parse_args()
    t0=time.time()

    model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,
              transl_invariant=True,two_dimensional=True)
    apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
    template=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N)))
    obj=flax.serialization.msgpack_restore(open(args.a1,"rb").read())
    variables=flax.serialization.from_state_dict(template,obj)
    p1=variables["params"]; flat1,unravel=ravel_pytree(p1)

    @jax.jit
    def eval_batch_jit(p,x):
        return jnp.real(apply({"params":p},x))

    def eval_states(p,states,batch=None):
        states=np.asarray(states,np.uint64)
        batch=args.eval_batch if batch is None else batch
        out=[]
        for i in range(0,len(states),batch):
            x=jnp.asarray(bits2x(states[i:i+batch]))
            out.append(np.asarray(eval_batch_jit(p,x)))
        return np.concatenate(out) if out else np.empty(0,float)

    h=np.load(args.handoff1)
    pool=h["pool_states"].astype(np.uint64)
    w=np.asarray(h["pool_weights"],float); w/=w.sum()
    rng=np.random.default_rng(20261003)
    n=args.ntrain+args.nval
    idx=rng.choice(len(pool),size=n,replace=False,p=w)
    train=pool[idx[:args.ntrain]]
    val=pool[idx[args.ntrain:]]
    base=np.unique(np.concatenate([train,val]))
    lat=SquareJ1J2(8,.5)
    print("VEC_STAGE sample","train",len(train),"val",len(val),
          "base_unique",len(base),"sec",time.time()-t0,flush=True)
    # Radius-1 set needed for signs on each base and its Hamiltonian neighbours.
    p1parts=[base]
    for i,j,J,mr in lat.bonds:
        sel=(((base>>np.uint64(i))^(base>>np.uint64(j)))&np.uint64(1)).astype(bool)
        if np.any(sel):
            p1parts.append(base[sel]^np.uint64((1<<i)|(1<<j)))
    r1=np.unique(np.concatenate(p1parts))
    del p1parts
    print("VEC_STAGE r1","n",len(r1),"sec",time.time()-t0,flush=True)

    # Radius-2 support needed only to compute exact analytic K1 local fields on r1.
    p2parts=[r1]
    edge_count=0
    for i,j,J,mr in lat.bonds:
        sel=(((r1>>np.uint64(i))^(r1>>np.uint64(j)))&np.uint64(1)).astype(bool)
        yy=r1[sel]^np.uint64((1<<i)|(1<<j))
        edge_count+=len(yy); p2parts.append(yy)
    r2=np.unique(np.concatenate(p2parts))
    del p2parts
    print("VEC_STAGE r2","n",len(r2),"edges",edge_count,
          "sec",time.time()-t0,flush=True)

    # One batched neural pass over the exact radius-2 support.
    log2=eval_states(p1,r2)
    print("VEC_STAGE amp_r2","n",len(log2),"sec",time.time()-t0,flush=True)
    i1=np.searchsorted(r2,r1)
    if not np.all(r2[i1]==r1): raise RuntimeError("r1 not subset of r2")
    log1=log2[i1]

    # Vectorized exact K1 local field r1=(H psi)/psi.
    rv=np.zeros(len(r1),float)
    for i,j,J,mr in lat.bonds:
        same=(((r1>>np.uint64(i))&1)==((r1>>np.uint64(j))&1))
        rv += J*np.where(same,0.25,-0.25)
        sel=~same
        if np.any(sel):
            yy=r1[sel]^np.uint64((1<<i)|(1<<j))
            iy=np.searchsorted(r2,yy)
            rv[sel] += 0.5*J*mr*np.exp(np.clip(log2[iy]-log1[sel],-40,40))
    mar=np.fromiter((lat.marshall(int(x)) for x in r1),dtype=np.int8,count=len(r1))
    sign1=mar*np.where(rv<=args.threshold,1,-1).astype(np.int8)
    print("VEC_STAGE signs","r_min",float(rv.min()),"r_max",float(rv.max()),
          "sec",time.time()-t0,flush=True)

    # r2 is no longer needed after signs are frozen.
    del r2,log2,i1,rv,mar
    gc.collect()

    # Build the exact frozen H_FN local operator on unique base states.
    ib=np.searchsorted(r1,base)
    if not np.all(r1[ib]==base): raise RuntimeError("base not subset of r1")
    logb=log1[ib]; sb=sign1[ib]
    df=np.zeros(len(base),float)
    allowed_src=[]; allowed_dst=[]; allowed_rate=[]
    for i,j,J,mr in lat.bonds:
        same=(((base>>np.uint64(i))&1)==((base>>np.uint64(j))&1))
        df += J*np.where(same,0.25,-0.25)
        sel=~same
        if not np.any(sel): continue
        ii=np.flatnonzero(sel)
        yy=base[sel]^np.uint64((1<<i)|(1<<j))
        iy=np.searchsorted(r1,yy)
        rat=0.5*J*np.exp(np.clip(log1[iy]-logb[sel],-40,40))
        good=(sb[sel]*sign1[iy] < 0)
        if np.any(good):
            allowed_src.append(ii[good].astype(np.int32))
            allowed_dst.append(yy[good])
            allowed_rate.append(rat[good])
        if np.any(~good):
            np.add.at(df,ii[~good],rat[~good])
    src=np.concatenate(allowed_src) if allowed_src else np.empty(0,np.int32)
    dst=np.concatenate(allowed_dst) if allowed_dst else np.empty(0,np.uint64)
    rate=np.concatenate(allowed_rate) if allowed_rate else np.empty(0,float)
    start_el=df.copy()
    if len(src): np.add.at(start_el,src,-rate)
    tr_idx=np.searchsorted(base,train); va_idx=np.searchsorted(base,val)
    Etr=start_el[tr_idx]; Eva=start_el[va_idx]
    needed=np.unique(np.concatenate([base,dst]))
    log1_need=eval_states(p1,needed)
    pos_base=np.searchsorted(needed,base)
    pos_dst=np.searchsorted(needed,dst) if len(dst) else np.empty(0,int)
    print("VEC_STAGE frozen","allowed_edges",len(src),"needed",len(needed),
          "train_E",float(Etr.mean()),"val_E",float(Eva.mean()),
          "sec",time.time()-t0,flush=True)

    np.savez_compressed(args.out_freeze,base=base,r1=r1,sign_r1=sign1,
        train=train,val=val,df=df,allowed_src=src,allowed_dst=dst,
        allowed_rate=rate,start_el=start_el,threshold=args.threshold)

    def frozen_local_unique(p):
        logn=eval_states(p,needed)
        delta=logn-log1_need
        out=df.copy()
        if len(src):
            rr=rate*np.exp(np.clip(delta[pos_dst]-delta[pos_base[src]],-40,40))
            np.add.at(out,src,-rr)
        return out

    # Finite-sample VMC force at the starting guide.
    Ebar=float(Etr.mean())
    coeff=jnp.asarray(2.0*(Etr-Ebar)/len(Etr))
    Xtrj=jnp.asarray(bits2x(train))
    def surrogate(pp):
        lv=jnp.real(apply({"params":pp},Xtrj))
        return jnp.dot(coeff,lv)
    _,g=jax.value_and_grad(surrogate)(p1)
    gf,_=ravel_pytree(g)
    print("VEC_STAGE force","gnorm",float(jnp.linalg.norm(gf)),
          "sec",time.time()-t0,flush=True)

    ns=min(args.fisher_n,len(train))
    Xq=jnp.asarray(bits2x(train[:ns]))
    def logvec(flat):
        return jnp.real(apply({"params":unravel(flat)},Xq))
    _,pullback=jax.vjp(logvec,flat1)
    def fisher(v):
        _,u=jax.jvp(logvec,(flat1,),(v,))
        u=u-jnp.mean(u)
        return pullback(u/ns)[0]
    fisher=jax.jit(fisher)
    _=fisher(jnp.zeros_like(flat1)).block_until_ready()
    print("VEC_STAGE fisher_ready","sec",time.time()-t0,flush=True)

    l1tr=eval_states(p1,train); l1va=eval_states(p1,val)
    rows=[]; accepted=[]
    for shift in (1.0,0.1):
        def A(v): return fisher(v)+shift*v
        sol,info=jax.scipy.sparse.linalg.cg(A,gf,tol=1e-5,atol=0.,maxiter=250)
        sol.block_until_ready()
        resid=float(jnp.linalg.norm(A(sol)-gf)/jnp.linalg.norm(gf))
        descent=float(jnp.vdot(gf,sol).real)
        _,tan=jax.jvp(logvec,(flat1,),(-sol,))
        tan=np.asarray(tan,float); tan_rms=float(np.std(tan))
        for trust in (0.001,0.002,0.005,0.01):
            eta=trust/max(tan_rms,1e-300)
            p2=unravel(flat1-eta*sol)
            eu=frozen_local_unique(p2)
            E2tr=eu[tr_idx]; E2va=eu[va_idx]
            l2tr=eval_states(p2,train); l2va=eval_states(p2,val)
            dtr=2*(l2tr-l1tr); dtr-=np.max(dtr)
            rwtr=np.exp(dtr); rwtr/=rwtr.sum()
            dva=2*(l2va-l1va); dva-=np.max(dva)
            rwva=np.exp(dva); rwva/=rwva.sum()
            E1tr=float(Etr.mean()); E1va=float(Eva.mean())
            Etr2=float(rwtr@E2tr); Eva2=float(rwva@E2va)
            ess_tr=float(1/(rwtr@rwtr)); ess_va=float(1/(rwva@rwva))

            # Delta-method paired influence estimate for the validation energy difference.
            # Candidate SNIS influence: w_i*(E2_i-E2), with mean-normalized weights N*w_i.
            infl2=len(val)*rwva*(E2va-Eva2)
            infl1=Eva-E1va
            dinfl=infl2-infl1
            se=float(np.std(dinfl,ddof=1)/np.sqrt(len(val)))
            z=float((Eva2-E1va)/se) if se>0 else float("-inf")

            row={"shift":shift,"trust":trust,"eta":float(eta),
                 "cg_rel_resid":resid,"g_dot_Sinv_g":descent,
                 "tangent_rms_per_eta":tan_rms,
                 "train_before":E1tr,"train_after":Etr2,"train_delta":Etr2-E1tr,
                 "val_before":E1va,"val_after":Eva2,"val_delta":Eva2-E1va,
                 "val_delta_se":se,"val_z":z,
                 "train_ESS":ess_tr,"val_ESS":ess_va}
            row["pass"]=bool(Etr2<E1tr and Eva2<E1va and
                             ess_tr>0.8*len(train) and ess_va>0.8*len(val))
            row["strong_pass"]=bool(row["pass"] and z < -2.0)
            rows.append(row)
            print("FROZEN_HFN_SR8_VEC_LINE",json.dumps(row),flush=True)
            if row["pass"]:
                accepted.append((Eva2-E1va,Etr2-E1tr,shift,trust,eta,p2,row))

    out={"ntrain":len(train),"nval":len(val),"base_unique":len(base),
         "r1_unique":len(r1),"fisher_n":ns,"threshold":args.threshold,
         "objective":"Rayleigh energy of exact frozen H_FN[a1,s1]",
         "rows":rows,"pass":bool(accepted),
         "strong_pass":bool(any(r["strong_pass"] for r in rows)),
         "elapsed_sec":time.time()-t0}
    if accepted:
        _,_,shift,trust,eta,pbest,best=min(accepted,key=lambda z:(z[0],z[1]))
        out["chosen"]=best
        v2=dict(variables); v2["params"]=pbest
        open(args.out_ck,"wb").write(flax.serialization.to_bytes(v2))
        print("FROZEN_HFN_SR8_VEC_SAVED",args.out_ck,flush=True)
    else:
        print("FROZEN_HFN_SR8_VEC_REJECTED",flush=True)
    Path(args.out_json).write_text(json.dumps(out,indent=2))
    print("FROZEN_HFN_SR8_VEC_RESULT",json.dumps(out),flush=True)

if __name__=="__main__":
    main()
