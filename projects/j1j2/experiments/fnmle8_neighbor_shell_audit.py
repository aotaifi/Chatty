import json, argparse
import numpy as np
from callable_fn_loop import CachedAmplitude, SquareJ1J2, K1FNEngine
from vit8_callable_amplitude_adapter import load

def weighted_sample(states,w,n,seed=20261001):
    rng=np.random.default_rng(seed); w=np.asarray(w,float); w/=w.sum()
    return np.asarray(states,np.uint64)[rng.choice(len(states),n,replace=True,p=w)]

def arrstats(x):
    x=np.asarray(x,float)
    return {"n":int(len(x)),"mean":float(x.mean()) if len(x) else None,
            "sd":float(x.std(ddof=1)) if len(x)>1 else 0.0,
            "rms":float(np.sqrt(np.mean(x*x))) if len(x) else None,
            "q50":float(np.median(x)) if len(x) else None,
            "q90":float(np.quantile(np.abs(x),.9)) if len(x) else None}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a0",required=True); ap.add_argument("--a1",required=True); ap.add_argument("--a2",required=True)
    ap.add_argument("--h1",required=True); ap.add_argument("--rep1",required=True); ap.add_argument("--rep2",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--nbase",type=int,default=96)
    args=ap.parse_args()
    T0=-28.37107876288694; T1=-28.824166903057147; T2=-28.88171175338251
    lat=SquareJ1J2(8,.5)
    amps=[CachedAmplitude(load(q).log_amplitude_bits) for q in [args.a0,args.a1,args.a2]]
    eng=[K1FNEngine(lat,a) for a in amps]
    for e,T in zip(eng,[T0,T1,T2]): e.threshold=T
    h=np.load(args.h1); pool=h["pool_states"].astype(np.uint64); w=h["pool_weights"].astype(float)
    base=weighted_sample(pool,w,args.nbase)
    edges=[]
    for x in base:
        for y,J,b in lat.neigh(int(x)):
            edges.append((int(x),int(y),float(J)))
    ys=np.array(sorted(set(y for _,y,_ in edges)),dtype=np.uint64)
    xs=np.array(sorted(set(x for x,_,_ in edges)),dtype=np.uint64)
    for e in eng:
        e.ensure_sign(np.concatenate([xs,ys]))

    support_ref=set(map(int,pool.tolist()))
    r1=np.load(args.rep1)["mixed"].astype(np.uint64); r2=np.load(args.rep2)["mixed"].astype(np.uint64)
    support_walk=set(map(int,np.concatenate([r1,r2]).tolist()))
    support_union=support_ref|support_walk

    rows=[]
    for x,y,J in edges:
        l0x,l0y=amps[0].loga(x),amps[0].loga(y)
        l1x,l1y=amps[1].loga(x),amps[1].loga(y)
        l2x,l2y=amps[2].loga(x),amps[2].loga(y)
        d01=(l1y-l1x)-(l0y-l0x)
        d12=(l2y-l2x)-(l1y-l1x)
        r0=eng[0].r_cache[y]; rr1=eng[1].r_cache[y]; rr2=eng[2].r_cache[y]
        s0=1 if T0-r0>=0 else -1; s1=1 if T1-rr1>=0 else -1; s2=1 if T2-rr2>=0 else -1
        rows.append((d01,d12,s0!=s1,s1!=s2,abs(T0-r0),abs(T1-rr1),
                     y in support_ref,y in support_walk,y in support_union))
    a=np.asarray(rows,object)
    d01=np.asarray(a[:,0],float); d12=np.asarray(a[:,1],float)
    f01=np.asarray(a[:,2],bool); f12=np.asarray(a[:,3],bool)
    m0=np.asarray(a[:,4],float); m1=np.asarray(a[:,5],float)
    ref=np.asarray(a[:,6],bool); walk=np.asarray(a[:,7],bool); union=np.asarray(a[:,8],bool)

    out={"nbase":int(len(base)),"nedges":int(len(edges)),"n_unique_y":int(len(ys)),
         "support":{"ref_frac":float(ref.mean()),"walker_frac":float(walk.mean()),"union_frac":float(union.mean())},
         "update01":{},"update12":{}}
    for name,d,flip,margin in [("update01",d01,f01,m0),("update12",d12,f12,m1)]:
        z={}
        for label,mask in [("all",np.ones(len(d),bool)),("seen",union),("unseen",~union)]:
            dd=d[mask]; ff=flip[mask]; mm=margin[mask]
            z[label]={"delta_logratio":arrstats(dd),
                      "flip_rate":float(ff.mean()) if len(ff) else None,
                      "margin":arrstats(mm),
                      "median_margin_flips":float(np.median(mm[ff])) if ff.any() else None,
                      "median_margin_noflip":float(np.median(mm[~ff])) if (~ff).any() else None}
        out[name]=z
    print("NEIGHBOR_AUDIT",json.dumps(out),flush=True)
    with open(args.out,"w") as f: json.dump(out,f,indent=2)

if __name__=="__main__": main()
