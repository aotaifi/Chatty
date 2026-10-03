import argparse, json, math
import numpy as np
import jax, jax.numpy as jnp
import flax
import netket.jax as nkjax
from jax.flatten_util import ravel_pytree
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d
from callable_fn_loop import CachedAmplitude, SquareJ1J2, K1FNEngine

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
    ap.add_argument("--ntrain",type=int,default=256)
    ap.add_argument("--nval",type=int,default=256)
    ap.add_argument("--fisher-n",type=int,default=256)
    ap.add_argument("--shift",type=float,default=0.1)
    ap.add_argument("--threshold",type=float,default=T1)
    args=ap.parse_args()

    model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,
              transl_invariant=True,two_dimensional=True)
    apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
    template=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N)))
    obj=flax.serialization.msgpack_restore(open(args.a1,"rb").read())
    variables=flax.serialization.from_state_dict(template,obj)
    p1=variables["params"]; flat1,unravel=ravel_pytree(p1)

    def eval_real(p,states,batch=1024):
        X=bits2x(states); out=[]
        for i in range(0,len(X),batch):
            out.append(np.asarray(apply({"params":p},jnp.asarray(X[i:i+batch]))).real)
        return np.concatenate(out)

    def loga_fn(p):
        return lambda states: eval_real(p,np.asarray(states,np.uint64))

    h=np.load(args.handoff1)
    pool=h["pool_states"].astype(np.uint64)
    w=np.asarray(h["pool_weights"],float); w/=w.sum()
    rng=np.random.default_rng(20261003)
    n=args.ntrain+args.nval
    idx=rng.choice(len(pool),size=n,replace=False,p=w)
    train=pool[idx[:args.ntrain]]
    val=pool[idx[args.ntrain:]]

    amp1=CachedAmplitude(loga_fn(p1))
    lat=SquareJ1J2(8,.5)
    signs=K1FNEngine(lat,amp1); signs.threshold=args.threshold
    base=np.concatenate([train,val])
    signs.ensure_local(base)

    def frozen_edges(states):
        states=np.asarray(states,np.uint64)
        d=[]; child=[]; owner=[]; coeff=[]; bad=[]
        for i,x0 in enumerate(states):
            x=int(x0)
            # Frozen diagonal Vsf built from old a1,s1:
            d.append(signs.local_cache[x][0])
            sx=signs.sign_cache[x]
            na=0
            for y,J,_ in lat.neigh(x):
                sy=signs.sign_cache[int(y)]
                if sx*sy<0:
                    child.append(int(y)); owner.append(i); coeff.append(0.5*J); na+=1
            bad.append(na)
        return (np.asarray(d,float),np.asarray(child,np.uint64),
                np.asarray(owner,np.int32),np.asarray(coeff,float),np.asarray(bad,int))

    dtr,ctr,otr,ktr,natr=frozen_edges(train)
    dva,cva,ova,kva,nava=frozen_edges(val)
    print("FROZEN8_SETUP",json.dumps({
        "ntrain":len(train),"nval":len(val),
        "train_allowed_edges":len(ctr),"val_allowed_edges":len(cva),
        "train_allowed_mean":float(natr.mean()),"val_allowed_mean":float(nava.mean()),
        "sign_r_cache":len(signs.r_cache),"amp_eval":amp1.neval}),flush=True)

    l1tr=eval_real(p1,train); l1va=eval_real(p1,val)

    def frozen_local(p,states,d,child,owner,coeff):
        lb=eval_real(p,states)
        lc=eval_real(p,child)
        acc=np.zeros(len(states),float)
        np.add.at(acc,owner,coeff*np.exp(lc-lb[owner]))
        return d-acc,lb

    Etr,ltr=frozen_local(p1,train,dtr,ctr,otr,ktr)
    Eva,lva=frozen_local(p1,val,dva,cva,ova,kva)
    Ebar=float(Etr.mean())
    coeff_force=jnp.asarray(2.0*(Etr-Ebar)/len(Etr))
    Xtr=jnp.asarray(bits2x(train))
    def surrogate(pp):
        lv=jnp.real(apply({"params":pp},Xtr))
        return jnp.dot(coeff_force,lv)
    _,g=jax.value_and_grad(surrogate)(p1)
    gf,_=ravel_pytree(g)

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
    def A(v): return fisher(v)+args.shift*v
    sol,info=jax.scipy.sparse.linalg.cg(A,gf,tol=1e-5,atol=0.,maxiter=250)
    sol.block_until_ready()
    resid=float(jnp.linalg.norm(A(sol)-gf)/jnp.linalg.norm(gf))
    deriv=float(jnp.vdot(gf,sol).real)
    _,tan=jax.jvp(logvec,(flat1,),(-sol,))
    tan=np.asarray(tan,float); tan_rms=float(np.std(tan))

    rows=[]; accepted=[]
    for trust in (0.002,0.005,0.01):
        eta=trust/max(tan_rms,1e-300)
        p2=unravel(flat1-eta*sol)
        E2tr,l2tr=frozen_local(p2,train,dtr,ctr,otr,ktr)
        E2va,l2va=frozen_local(p2,val,dva,cva,ova,kva)

        dltr=l2tr-l1tr
        rwtr=np.exp(2*dltr-np.max(2*dltr)); rwtr/=rwtr.sum()
        dlva=l2va-l1va
        rwva=np.exp(2*dlva-np.max(2*dlva)); rwva/=rwva.sum()

        e1tr=float(Etr.mean()); e2tr=float(rwtr@E2tr)
        e1va=float(Eva.mean()); e2va=float(rwva@E2va)
        ess_tr=float(1/(rwtr@rwtr)); ess_va=float(1/(rwva@rwva))
        row={
          "trust":trust,"eta":float(eta),
          "train_before":e1tr,"train_after":e2tr,"train_delta":e2tr-e1tr,
          "val_before":e1va,"val_after":e2va,"val_delta":e2va-e1va,
          "train_ESS":ess_tr,"val_ESS":ess_va,
          "train_dlog_rms":float(np.std(dltr)),
          "val_dlog_rms":float(np.std(dlva))}
        rows.append(row); print("FROZEN8_LINE",json.dumps(row),flush=True)
        if e2tr<e1tr and e2va<e1va and ess_tr>0.8*len(train) and ess_va>0.8*len(val):
            accepted.append((e2va-e1va,trust,p2,row))

    out={
      "shift":args.shift,"fisher_n":ns,"cg_rel_resid":resid,
      "g_dot_Sinv_g":deriv,"tangent_rms_per_eta":tan_rms,
      "train_E_before":float(Etr.mean()),"val_E_before":float(Eva.mean()),
      "rows":rows,"pass":bool(accepted)}
    if accepted:
        _,trust,pbest,best=min(accepted,key=lambda z:z[0])
        out["chosen"]=best
        v2=dict(variables); v2["params"]=pbest
        open(args.out_ck,"wb").write(flax.serialization.to_bytes(v2))
        print("FROZEN8_SAVED",args.out_ck,flush=True)
    json.dump(out,open(args.out_json,"w"),indent=2)
    print("FROZEN8_RESULT",json.dumps(out),flush=True)

if __name__=="__main__":
    main()
