import argparse, json, math
import numpy as np
from callable_fn_loop import CachedAmplitude, SquareJ1J2, K1FNEngine
from vit8_callable_amplitude_adapter import load

T1=-28.824166903057147
ALPHA=1.2

class CurrentSignField:
    def __init__(self,lat,amp,parent,chunk=256):
        self.lat=lat; self.amp=amp; self.parent=parent; self.r={}
        self.chunk=int(chunk)

    def _ensure_parent_sign_chunked(self,states):
        uu=np.unique(np.asarray(states,np.uint64))
        miss=[int(x) for x in uu if int(x) not in self.parent.sign_cache]
        for i in range(0,len(miss),self.chunk):
            self.parent.ensure_sign(np.asarray(miss[i:i+self.chunk],np.uint64))

    def ensure_r(self,states):
        miss=[int(x) for x in np.unique(np.asarray(states,np.uint64)) if int(x) not in self.r]
        if not miss: return
        # Bound peak memory: current-sign r(x) needs parent signs on the one-hop shell,
        # and each parent sign itself needs a Marshall-r one-hop shell. Never materialize
        # the full nested shell for all sampled nodes at once.
        for i in range(0,len(miss),self.chunk):
            batch=miss[i:i+self.chunk]
            meta=[]; alln=[]
            for x in batch:
                ls=self.lat.neigh(x); meta.append(ls); alln.extend(y for y,_,_ in ls)
            need=np.unique(np.asarray(batch+alln,np.uint64))
            self._ensure_parent_sign_chunked(need)
            self.amp.ensure(need)
            for x,ls in zip(batch,meta):
                lx=self.amp.loga(x); sx=self.parent.sign_cache[x]
                rr=self.lat.diag_energy(x)
                for y,J,_ in ls:
                    sy=self.parent.sign_cache[y]
                    rr += 0.5*J*(sy/sx)*math.exp(self.amp.loga(y)-lx)
                self.r[x]=float(rr)

def make_groups(nodes,field,atol=1e-10,rtol=1e-10):
    vals=np.array([field.r[int(x)] for x in nodes],float)
    order=np.argsort(vals,kind="mergesort")
    groups=np.empty(len(nodes),np.int32)
    g=0; ref=vals[order[0]]; groups[order[0]]=0
    for jj in order[1:]:
        v=vals[jj]
        if abs(v-ref)>atol+rtol*max(abs(v),abs(ref),1.0):
            g+=1; ref=v
        groups[jj]=g
    return vals,groups,g+1
def energy_curve(states,weights,field):
    states=np.asarray(states,np.uint64); w=np.asarray(weights,float); w/=w.sum()
    edges=[]; nodes=set(map(int,states.tolist()))
    for x,wx in zip(states,w):
        for y,J,_ in field.lat.neigh(int(x)):
            edges.append((int(x),int(y),float(J),float(wx))); nodes.add(int(y))
    nodes=np.array(sorted(nodes),np.uint64)
    field.ensure_r(nodes)
    vals,gids,G=make_groups(nodes,field)
    gid={int(x):int(g) for x,g in zip(nodes,gids)}
    rv={int(x):float(v) for x,v in zip(nodes,vals)}
    base=0.0; diff=np.zeros(G,float)
    for x,wx in zip(states,w):
        x=int(x); lx=field.amp.loga(x); sx=field.parent.sign_cache[x]
        base += wx*field.lat.diag_energy(x)
        gx=gid[x]
        for y,J,_ in field.lat.neigh(x):
            sy=field.parent.sign_cache[y]
            c=wx*0.5*J*(sy/sx)*math.exp(field.amp.loga(y)-lx)
            base += c
            gy=gid[y]
            if gx!=gy:
                lo=min(gx,gy); hi=max(gx,gy)
                diff[lo] += -2*c
                diff[hi] += +2*c
    after=base+np.cumsum(diff)
    cand=np.r_[base,after]
    ib=int(np.argmin(cand)); k=ib-1
    if k<0:
        T=float(min(rv.values())-max(1.0,0.1*abs(min(rv.values()))))
    elif k>=G-1:
        T=float(max(rv.values())+max(1.0,0.1*abs(max(rv.values()))))
    else:
        lo=max(v for x,v in rv.items() if gid[x]==k)
        hi=min(v for x,v in rv.items() if gid[x]==k+1)
        T=0.5*(lo+hi)
    return {"T":T,"Ebase":float(base),"Ebest":float(cand[ib]),"gain":float(cand[ib]-base),
            "groups":int(G),"nodes":int(len(nodes)),"edges":int(len(edges))},rv

def eval_threshold(states,weights,field,T):
    states=np.asarray(states,np.uint64); w=np.asarray(weights,float); w/=w.sum()
    nodes=set(map(int,states.tolist()))
    for x in states:
        nodes.update(int(y) for y,_,_ in field.lat.neigh(int(x)))
    field.ensure_r(np.array(sorted(nodes),np.uint64))
    E0=0.0; E1=0.0; changed=0.0; drows=[]
    for x,wx in zip(states,w):
        x=int(x); lx=field.amp.loga(x); sx=field.parent.sign_cache[x]
        mx=1 if field.r[x]<=T else -1
        changed += wx*(mx<0)
        e0=field.lat.diag_energy(x); e1=e0
        for y,J,_ in field.lat.neigh(x):
            sy=field.parent.sign_cache[y]
            c=0.5*J*(sy/sx)*math.exp(field.amp.loga(y)-lx)
            e0 += c
            my=1 if field.r[y]<=T else -1
            e1 += c*mx*my
        E0 += wx*e0; E1 += wx*e1; drows.append((wx,e1-e0))
    ww=np.array([z[0] for z in drows],float); dd=np.array([z[1] for z in drows],float); ww/=ww.sum()
    ess=float(1/(ww@ww)); mu=float(ww@dd); var=float(ww@((dd-mu)**2)); se=float(np.sqrt(var/max(ess,1.0)))
    return {"Ebase":float(E0),"Ecand":float(E1),"delta":float(E1-E0),"delta_se_ess":se,"ESS":ess,
            "change_mass_global_aware":float(min(changed,1-changed))}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a1",required=True); ap.add_argument("--handoff1",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--ntrain",type=int,default=256)
    ap.add_argument("--nval",type=int,default=256)
    args=ap.parse_args()
    lat=SquareJ1J2(8,.5)
    amp=CachedAmplitude(load(args.a1).log_amplitude_bits)
    parent=K1FNEngine(lat,amp); parent.threshold=T1
    field=CurrentSignField(lat,amp,parent)
    h=np.load(args.handoff1)
    # Use the physical guide pool directly. pool_states were drawn from a0^2 and
    # pool_weights reweight them to a1^2. Resampling by those weights therefore
    # gives an (approximately) unweighted a1^2 sample for the energy objective.
    ppool=h["pool_states"].astype(np.uint64); pw=np.asarray(h["pool_weights"],float); pw/=pw.sum()
    rng=np.random.default_rng(20261001)
    tr=ppool[rng.choice(len(ppool),args.ntrain,replace=True,p=pw)]
    va=ppool[rng.choice(len(ppool),args.nval,replace=True,p=pw)]
    wt=np.ones(len(tr),float)
    wv=np.ones(len(va),float)
    train,_=energy_curve(tr,wt,field)
    val=eval_threshold(va,wv,field,train["T"])
    out={"T1_parent":T1,"proposal":"resampled_a1_squared_pool","train":train,"val":val,
         "parent_sign_cache":len(parent.sign_cache),"parent_r_cache":len(parent.r_cache),
         "current_r_cache":len(field.r),"amp_eval":int(amp.neval),
         "proposal_val_direction":"down" if val["delta"]<0 else "up",
         "proposal_val_z":float(-val["delta"]/max(val["delta_se_ess"],1e-300))}
    print("CURRENT_SIGN_ENERGY_THRESHOLD",json.dumps(out),flush=True)
    json.dump(out,open(args.out,"w"),indent=2)

if __name__=="__main__": main()
