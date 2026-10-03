import argparse, json, math, time
import numpy as np
from callable_fn_loop import CachedAmplitude, SquareJ1J2, K1FNEngine
from vit8_callable_amplitude_adapter import load

class FrozenSignGuideFN:
    """FN projector using guide-amplitude ratios from amp_guide but signs frozen from sign_engine."""
    def __init__(self, lattice, amp_guide, sign_engine):
        self.lat=lattice; self.amp=amp_guide; self.sign_engine=sign_engine
        self.local_cache={}

    def ensure_local(self, states):
        miss=[int(x) for x in np.unique(np.asarray(states,np.uint64)) if int(x) not in self.local_cache]
        if not miss: return
        meta=[]; alln=[]
        for x in miss:
            ls=self.lat.neigh(x); meta.append(ls); alln.extend(y for y,_,_ in ls)
        # Crucial isolation: all signs are computed from a1,T1 only.
        self.sign_engine.ensure_sign(miss+alln)
        self.amp.ensure(miss+alln)
        for x,ls in zip(miss,meta):
            lx=self.amp.loga(x); sx=self.sign_engine.sign_cache[x]
            d=self.lat.diag_energy(x); ys=[]; rates=[]
            for y,J,_ in ls:
                rat=math.exp(self.amp.loga(y)-lx)
                sy=self.sign_engine.sign_cache[y]
                if sx*sy < 0:
                    ys.append(y); rates.append(0.5*J*rat)
                else:
                    d += 0.5*J*rat
            rates=np.asarray(rates,float)
            self.local_cache[x]=(float(d),float(d-rates.sum()),np.asarray(ys,np.uint64),rates)

    @staticmethod
    def _systematic(w,rng,M):
        c=np.cumsum(w); c[-1]=1.0
        return np.searchsorted(c,rng.random()/M+np.arange(M)/M,'right')

    def run(self,pool,pool_weights,seed,M=128,beta_target=1.2,burn_beta=.4,tau_max=.025):
        rng=np.random.default_rng(seed); pool=np.asarray(pool,np.uint64)
        p=np.asarray(pool_weights,float); p/=p.sum()
        walkers=pool[rng.choice(len(pool),M,replace=False,p=p)].copy()
        beta=0.; it=0; Es=[]; snaps=[]; t0=time.time()
        while beta < beta_target:
            self.ensure_local(walkers)
            dat=[self.local_cache[int(x)] for x in walkers]
            diag=np.array([z[0] for z in dat]); el=np.array([z[1] for z in dat])
            Eref=float(el.mean())
            mx=max(0.,float(np.max(diag-Eref)))
            tau=min(tau_max,0.8/mx if mx>0 else tau_max)
            nxt=np.empty(M,np.uint64); bw=np.empty(M)
            for k,(x,(d,e,ys,rates)) in enumerate(zip(walkers,dat)):
                stay=1-tau*(d-Eref); ws=tau*rates; tot=stay+ws.sum()
                if stay<0 or tot<=0: raise RuntimeError(("bad",stay,tot,tau,d,Eref))
                u=rng.random()*tot
                if u<stay: y=x
                else:
                    j=np.searchsorted(np.cumsum(ws),u-stay,'right')
                    y=int(ys[min(j,len(ys)-1)])
                nxt[k]=y; bw[k]=tot
            bw/=bw.sum(); walkers=nxt[self._systematic(bw,rng,M)]
            beta+=tau; it+=1
            if beta>=burn_beta:
                self.ensure_local(walkers)
                Es.append(float(np.mean([self.local_cache[int(x)][1] for x in walkers])))
                snaps.append(walkers.copy())
            if it==1 or it%5==0:
                print("FROZEN_PROG",seed,it,"beta",beta,"tau",tau,"E",Eref,
                      "uniq",len(np.unique(walkers)),
                      "sign_r",len(self.sign_engine.r_cache),
                      "sign_amp",self.sign_engine.amp.neval,
                      "guide_amp",self.amp.neval,
                      "local",len(self.local_cache),"sec",time.time()-t0,flush=True)
        E=np.asarray(Es,float); mixed=np.concatenate(snaps)
        print("FROZEN_RESULT",seed,"Emean",float(E.mean()),"tail8",float(E[-8:].mean()),
              "nE",len(E),"mixed",len(mixed),"unique",len(np.unique(mixed)),
              "sign_amp",self.sign_engine.amp.neval,"guide_amp",self.amp.neval,flush=True)
        return mixed,E

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a1",required=True); ap.add_argument("--a2",required=True)
    ap.add_argument("--handoff1",required=True); ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--diag-n",type=int,default=256); ap.add_argument("--diag-only",action="store_true")
    args=ap.parse_args()
    T1=-28.824166903057147
    lat=SquareJ1J2(8,.5)
    a1=CachedAmplitude(load(args.a1).log_amplitude_bits)
    a2=CachedAmplitude(load(args.a2).log_amplitude_bits)
    signs=K1FNEngine(lat,a1); signs.threshold=T1
    h=np.load(args.handoff1); pool=h["pool_states"].astype(np.uint64)
    w1=np.asarray(h["pool_weights"],float); w1/=w1.sum()

    # Common-state fixed-sign local-energy diagnostic, sampled from a1^2.
    rg=np.random.default_rng(24681357)
    ds=pool[rg.choice(len(pool),args.diag_n,replace=True,p=w1)]
    signs.ensure_local(ds)  # a1,s1 local energies
    e1=np.array([signs.local_cache[int(x)][1] for x in ds],float)
    frozen=FrozenSignGuideFN(lat,a2,signs)
    frozen.ensure_local(ds)
    e2=np.array([frozen.local_cache[int(x)][1] for x in ds],float)
    a1.ensure(ds); a2.ensure(ds)
    lr=np.array([a2.loga(x)-a1.loga(x) for x in ds])
    rw=np.exp(2*lr-np.max(2*lr)); rw/=rw.sum()
    diag={
      "n":int(len(ds)),
      "a1_common_mean":float(e1.mean()),"a1_common_var":float(e1.var(ddof=1)),
      "a2_common_mean":float(e2.mean()),"a2_common_var":float(e2.var(ddof=1)),
      "a2_reweighted_mean":float(rw@e2),
      "a2_reweighted_var":float(rw@((e2-rw@e2)**2)),
      "reweight_ESS":float(1/(rw@rw)),
      "delta_local_mean":float(np.mean(e2-e1)),
      "delta_local_sd":float(np.std(e2-e1,ddof=1)),
      "delta_local_se":float(np.std(e2-e1,ddof=1)/np.sqrt(len(e2))),
      "frac_local_lower":float(np.mean(e2<e1)),
      "delta_local_rms":float(np.sqrt(np.mean((e2-e1)**2)))}
    print("FROZEN_DIAG",json.dumps(diag),flush=True)

    if args.diag_only:
        json.dump(diag,open(args.out+".json","w"),indent=2)
        return
    mixed,E=frozen.run(pool,w1,args.seed)
    out={"seed":args.seed,"T1":T1,"diag":diag,
         "Emean":float(E.mean()),"tail8":float(E[-8:].mean()),
         "nE":len(E),"Nmixed":len(mixed),"unique":len(np.unique(mixed))}
    np.savez_compressed(args.out,Es=E,mixed=mixed,T1=T1,
                        diag_json=np.array(json.dumps(diag)))
    print("FROZEN_SUMMARY",json.dumps(out),flush=True)

if __name__=="__main__": main()
