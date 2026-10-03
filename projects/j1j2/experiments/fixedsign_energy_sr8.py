import json, math, argparse
import numpy as np
import jax, jax.numpy as jnp
import flax
import netket.jax as nkjax
from jax.flatten_util import ravel_pytree
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d
from callable_fn_loop import CachedAmplitude, SquareJ1J2, K1FNEngine
from frozen_s1_a2_fn8 import FrozenSignGuideFN

jax.config.update("jax_enable_x64", True)
N=64; T1=-28.824166903057147

def bits2x(ss):
    a=np.asarray(ss,np.uint64).reshape(-1,1)
    return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float64)-1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a1",required=True); ap.add_argument("--handoff1",required=True)
    ap.add_argument("--out-ck",required=True); ap.add_argument("--out-json",required=True)
    ap.add_argument("--ntrain",type=int,default=256); ap.add_argument("--nval",type=int,default=256)
    ap.add_argument("--fisher-n",type=int,default=256); ap.add_argument("--shift",type=float,default=1.0)
    ap.add_argument("--trust-rms",type=float,default=0.05)
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

    h=np.load(args.handoff1)
    pool=h["pool_states"].astype(np.uint64)
    w=np.asarray(h["pool_weights"],float); w/=w.sum()
    rng=np.random.default_rng(20261001)
    n=args.ntrain+args.nval
    idx=rng.choice(len(pool),size=n,replace=False,p=w)
    train=pool[idx[:args.ntrain]]; val=pool[idx[args.ntrain:]]
    Xtr=bits2x(train); Xva=bits2x(val)

    amp1=CachedAmplitude(loga_bits_from_params(p1))
    lat=SquareJ1J2(8,.5)
    signs=K1FNEngine(lat,amp1); signs.threshold=T1
    signs.ensure_local(np.concatenate([train,val]))
    Etr=np.array([signs.local_cache[int(x)][1] for x in train],float)
    Eva=np.array([signs.local_cache[int(x)][1] for x in val],float)
    Ebar=float(Etr.mean())

    # Exact VMC energy force for fixed signs, with E_L stop-gradient.
    coeff=jnp.asarray(2.0*(Etr-Ebar)/len(Etr))
    Xtrj=jnp.asarray(Xtr)
    def surrogate(pp):
        lv=jnp.real(apply({"params":pp},Xtrj))
        return jnp.dot(coeff,lv)
    _,g=jax.value_and_grad(surrogate)(p1)
    gf,_=ravel_pytree(g)
    # Matrix-free SR/QGT on the same a1^2 sample.
    ns=min(args.fisher_n,len(Xtr))
    Xq=jnp.asarray(Xtr[:ns])
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
    sol,info=jax.scipy.sparse.linalg.cg(A,gf,tol=1e-5,atol=0.,maxiter=200)
    sol.block_until_ready()
    resid=float(jnp.linalg.norm(A(sol)-gf)/jnp.linalg.norm(gf))
    descent=float(jnp.vdot(gf,sol).real)

    # Choose eta from a geometric trust radius, not an LR scan.
    _,tangent=jax.jvp(logvec,(flat1,),(-sol,))
    tangent=np.asarray(tangent,float)
    tangent_rms=float(np.std(tangent))
    eta=args.trust_rms/tangent_rms
    flat2=flat1-eta*sol
    p2=unravel(flat2)
    v2=dict(variables); v2["params"]=p2

    # Held-out physical-energy gate with fixed s1.
    amp2=CachedAmplitude(loga_bits_from_params(p2))
    cand=FrozenSignGuideFN(lat,amp2,signs)
    cand.ensure_local(val)
    E2=np.array([cand.local_cache[int(x)][1] for x in val],float)
    l1=eval_real(p1,Xva); l2=eval_real(p2,Xva)
    rw=np.exp(2*(l2-l1)-np.max(2*(l2-l1))); rw/=rw.sum()
    E1_val=float(Eva.mean()); E2_val=float(rw@E2)
    V1=float(Eva.var(ddof=1)); V2=float(rw@((E2-E2_val)**2))
    ess=float(1/(rw@rw))
    out={
      "ntrain":len(train),"nval":len(val),"fisher_n":ns,"shift":args.shift,
      "trust_rms_target":args.trust_rms,"tangent_rms_per_eta":tangent_rms,
      "eta":float(eta),"cg_rel_resid":resid,"g_dot_Sinv_g":descent,
      "train_E_mean":Ebar,"train_E_sd":float(Etr.std(ddof=1)),
      "val_E_before":E1_val,"val_E_after":E2_val,"val_delta_E":E2_val-E1_val,
      "val_var_before":V1,"val_var_after":V2,"val_ESS":ess,
      "pass":bool(E2_val<E1_val and ess>0.8*len(val))}
    print("FIXEDSIGN_SR",json.dumps(out),flush=True)
    json.dump(out,open(args.out_json,"w"),indent=2)
    if out["pass"]:
        open(args.out_ck,"wb").write(flax.serialization.to_bytes(v2))
        print("FIXEDSIGN_SR_SAVED",args.out_ck,flush=True)
    else:
        print("FIXEDSIGN_SR_REJECTED",flush=True)

if __name__=="__main__": main()
