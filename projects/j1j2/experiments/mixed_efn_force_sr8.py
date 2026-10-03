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
    ap.add_argument("--rep1",required=True)
    ap.add_argument("--rep2",required=True)
    ap.add_argument("--out-ck",required=True)
    ap.add_argument("--out-json",required=True)
    ap.add_argument("--nforce",type=int,default=128)
    ap.add_argument("--fisher-n",type=int,default=256)
    ap.add_argument("--shift",type=float,default=1.0)
    ap.add_argument("--threshold",type=float,default=T1)
    ap.add_argument("--pool-npz",default=None)
    ap.add_argument("--pool-states-key",default="pool_states")
    ap.add_argument("--pool-weights-key",default="pool_weights")
    ap.add_argument("--grad-base-batch",type=int,default=8)
    ap.add_argument("--edge-cache",default=None)
    args=ap.parse_args()

    model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,
              transl_invariant=True,two_dimensional=True)
    apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
    template=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N)))
    obj=flax.serialization.msgpack_restore(open(args.a1,"rb").read())
    variables=flax.serialization.from_state_dict(template,obj)
    p1=variables["params"]; flat1,unravel=ravel_pytree(p1)

    def eval_real(p,X,batch=1024):
        out=[]
        for i in range(0,len(X),batch):
            out.append(np.asarray(apply({"params":p},jnp.asarray(X[i:i+batch]))).real)
        return np.concatenate(out)

    def loga_bits_from_params(p):
        def fn(states):
            return eval_real(p,bits2x(states))
        return fn

    r1all=np.load(args.rep1)["mixed"].astype(np.uint64)
    r2all=np.load(args.rep2)["mixed"].astype(np.uint64)
    rng=np.random.default_rng(20261001)
    def draw(a,n):
        return a[rng.choice(len(a),size=n,replace=(n>len(a)))]
    r1=draw(r1all,args.nforce)
    r2=draw(r2all,args.nforce)

    lat=SquareJ1J2(8,.5)
    amp1=CachedAmplitude(loga_bits_from_params(p1))
    parent=K1FNEngine(lat,amp1); parent.threshold=args.threshold

    def edge_data(base):
        meta=[]; alln=[]
        for x in base:
            ls=lat.neigh(int(x)); meta.append(ls)
            alln.extend(y for y,_,_ in ls)
        parent.ensure_sign(np.concatenate([base,np.asarray(alln,np.uint64)]))
        child=[]; owner=[]; coeff=[]
        bad_per=[]
        for i,(x,ls) in enumerate(zip(base,meta)):
            sx=parent.sign_cache[int(x)]; nb=0
            for y,J,_ in ls:
                if sx*parent.sign_cache[int(y)]>0:
                    child.append(y); owner.append(i); coeff.append(.5*J); nb+=1
            bad_per.append(nb)
        return (np.asarray(base,np.uint64),np.asarray(child,np.uint64),
                np.asarray(owner,np.int32),np.asarray(coeff,float),np.asarray(bad_per))

    if args.edge_cache and __import__("os").path.exists(args.edge_cache):
        ec=np.load(args.edge_cache)
        b1,c1,o1,k1,nb1=[ec[x] for x in ("b1","c1","o1","k1","nb1")]
        b2,c2,o2,k2,nb2=[ec[x] for x in ("b2","c2","o2","k2","nb2")]
        print("MIXED_EFN_EDGE_CACHE_LOADED",args.edge_cache,flush=True)
    else:
        b1,c1,o1,k1,nb1=edge_data(r1)
        b2,c2,o2,k2,nb2=edge_data(r2)
        if args.edge_cache:
            np.savez_compressed(args.edge_cache,b1=b1,c1=c1,o1=o1,k1=k1,nb1=nb1,
                                b2=b2,c2=c2,o2=o2,k2=k2,nb2=nb2)
            print("MIXED_EFN_EDGE_CACHE_SAVED",args.edge_cache,flush=True)
    print("MIXED_EFN_EDGES",json.dumps({
        "n1":len(b1),"n2":len(b2),"edges1":len(c1),"edges2":len(c2),
        "bad_mean1":float(nb1.mean()),"bad_mean2":float(nb2.mean()),
        "amp_evals_for_signs":amp1.neval}),flush=True)

    # Accumulate the exact same mean force in small base-state blocks.  The
    # original all-edge reverse-mode graph exceeded H100 memory (~69 GiB).
    def block_loss(pp,b,c,o,k,i0,i1):
        mask=(o>=i0)&(o<i1)
        cc=c[mask]; oo=o[mask]-i0; kk=k[mask]
        Xb=jnp.asarray(bits2x(b[i0:i1])); Xc=jnp.asarray(bits2x(cc))
        oj=jnp.asarray(oo); kj=jnp.asarray(kk)
        lb=jnp.real(apply({"params":pp},Xb))
        lc=jnp.real(apply({"params":pp},Xc))
        return jnp.sum(kj*jnp.exp(lc-lb[oj]))/len(b)

    def force_value(p,b,c,o,k):
        val=0.0
        for i0 in range(0,len(b),args.grad_base_batch):
            i1=min(len(b),i0+args.grad_base_batch)
            val += float(block_loss(p,b,c,o,k,i0,i1))
        return val

    def loss_and_grad(p,b,c,o,k):
        val=0.0; gtot=None
        for i0 in range(0,len(b),args.grad_base_batch):
            i1=min(len(b),i0+args.grad_base_batch)
            fn=lambda pp: block_loss(pp,b,c,o,k,i0,i1)
            vb,gb=jax.value_and_grad(fn)(p)
            val += float(vb)
            gtot = gb if gtot is None else jax.tree_util.tree_map(lambda x,y:x+y,gtot,gb)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(),gtot)
        return val,gtot

    L1,g1=loss_and_grad(p1,b1,c1,o1,k1)
    L2,g2=loss_and_grad(p1,b2,c2,o2,k2)
    g1f,_=ravel_pytree(g1); g2f,_=ravel_pytree(g2)
    eu=float(jnp.vdot(g1f,g2f).real)
    n1=float(jnp.linalg.norm(g1f)); n2=float(jnp.linalg.norm(g2f))

    pool_path=args.pool_npz or args.handoff1
    h=np.load(pool_path)
    pool=h[args.pool_states_key].astype(np.uint64)
    if args.pool_weights_key in h.files:
        w=np.asarray(h[args.pool_weights_key],float); w/=w.sum()
    else:
        w=np.full(len(pool),1.0/len(pool),float)
    ns=min(args.fisher_n,len(pool))
    iq=rng.choice(len(pool),size=ns,replace=True,p=w)
    Xq=jnp.asarray(bits2x(pool[iq]))

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
    sol,info=jax.scipy.sparse.linalg.cg(A,g1f,tol=1e-5,atol=0.,maxiter=200)
    sol.block_until_ready()
    resid=float(jnp.linalg.norm(A(sol)-g1f)/jnp.linalg.norm(g1f))
    dtrain=float(jnp.vdot(g1f,sol).real)
    dval=float(jnp.vdot(g2f,sol).real)

    _,tan=jax.jvp(logvec,(flat1,),(-sol,))
    tan=np.asarray(tan,float)
    tan_rms=float(np.std(tan))
    lp1=eval_real(p1,bits2x(pool))
    rows=[]; accepted=[]
    for trust in (0.01,0.02,0.03,0.05):
        eta=trust/max(tan_rms,1e-300)
        p2=unravel(flat1-eta*sol)
        lv1=force_value(p2,b1,c1,o1,k1); lv2=force_value(p2,b2,c2,o2,k2)
        lp2=eval_real(p2,bits2x(pool))
        dl=lp2-lp1
        rw=w*np.exp(2*dl-np.max(2*dl)); rw/=rw.sum()
        ess=float(1/(rw@rw))
        row={"trust":trust,"eta":float(eta),
             "train_before":L1,"train_after":lv1,"train_delta":lv1-L1,
             "val_before":L2,"val_after":lv2,"val_delta":lv2-L2,
             "pool_ESS":ess,"pool_dlog_rms":float(np.sqrt(np.sum(w*(dl-np.sum(w*dl))**2))),
             "pool_dlog_max":float(np.max(np.abs(dl-np.sum(w*dl))))}
        rows.append(row); print("MIXED_EFN_LINE",json.dumps(row),flush=True)
        if lv1<L1 and lv2<L2 and ess>0.8*len(pool):
            accepted.append((lv2-L2,trust,eta,p2,row))

    out={"nforce":args.nforce,"fisher_n":ns,"shift":args.shift,"threshold":args.threshold,"pool_npz":pool_path,"grad_base_batch":args.grad_base_batch,
         "mixed_force_identity":"g_exact = 2*g_mixed + O((phi-a)^2)",
         "replica1_vsf":L1,"replica2_vsf":L2,
         "raw_grad_cos":eu/max(n1*n2,1e-300),
         "raw_grad_dot":eu,"g1_norm":n1,"g2_norm":n2,
         "sr_train_deriv":dtrain,"sr_val_deriv":dval,
         "cg_rel_resid":resid,"tangent_rms_per_eta":tan_rms,
         "rows":rows,"pass":bool(accepted)}
    if accepted:
        _,trust,eta,pbest,best=min(accepted,key=lambda z:z[0])
        out["chosen"]=best
        v2=dict(variables); v2["params"]=pbest
        open(args.out_ck,"wb").write(flax.serialization.to_bytes(v2))
        print("MIXED_EFN_SAVED",args.out_ck,flush=True)
    json.dump(out,open(args.out_json,"w"),indent=2)
    print("MIXED_EFN_RESULT",json.dumps(out),flush=True)

if __name__=="__main__":
    main()
