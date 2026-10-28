#!/usr/bin/env python3
"""AVSL09 K1 local-field vs individual profitable-orbit diagnostic (20 site)."""
import json,numpy as np,argparse,time
from collections import defaultdict
from avsl_cooperative20 import Oracle,BONDS

def build(path):
 o=Oracle(path); d=np.load(path); diagonal=d['diag']
 for x in o.states:o.key(int(x))
 edge={}
 rr={}
 for g in o.members:
  ix=o.look[g];a=o.a(g);s=o.s(g)
  field=float(diagonal[ix])
  weights=defaultdict(float)
  for bits,h in BONDS:
   if (g&bits).bit_count()!=1:continue
   y=g^bits; ay=o.a(y);sy=o.s(y);v=h*s*sy*ay/a
   field+=v
   gy=o.key(y)
   if gy!=g:weights[gy]+=len(o.members[g])*h*s*sy*a*ay
  rr[g]=field;edge[g]=weights
 gain={g:4*sum(w.values())/o.norm for g,w in edge.items()}
 return o,rr,edge,gain

def thresholds(o,rr,edge,gain):
 keys=sorted(rr,key=rr.get,reverse=True)
 flipped=set(); cum=0.; best=(-1,0,0)
 for n,g in enumerate(keys):
  weights=edge[g]
  marginal=4/o.norm*(sum(w for h,w in weights.items() if h not in flipped)-sum(w for h,w in weights.items() if h in flipped))
  cum+=marginal
  flipped.add(g)
  if cum>best[1]:best=(n,cum,rr[g])
 allpositive=set(g for g,gainval in gain.items() if gainval>1e-12)
 if best[0]>=0:
  sel=set(keys[:best[0]+1])
 else:sel=set()
 rows=[]
 for T in [0,1,2,3,4,5,6,7,8,10,12,15,20,30,40]:
  kvals={g for g,v in rr.items() if v>T}
  overlap=len(kvals & allpositive)
  rows.append(dict(T=T,n_k1=len(kvals),positive_single_overlap=overlap,
       gain_if_flip_all=float(sum(gain[g] for g in kvals)-8/o.norm*sum(w for g in kvals for h,w in edge[g].items() if h in kvals)/2)))
 return dict(n_orbits=len(keys),n_positive_singles=len(allpositive),
   r_min=float(min(rr.values())),r_max=float(max(rr.values())),
   best_single_K1_threshold=float(best[2]),
   best_K1_joint_gain=float(best[1]),best_K1_n_flips=int(best[0]+1),
   best_K1_positive_single_overlap=int(len(sel & allpositive)),
   selected_orbits=list(map(int,sorted(sel))),rows=rows, rkeys={str(g):float(v) for g,v in rr.items()})

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 tic=time.perf_counter();o,r,e,g=build(a.data);v=thresholds(o,r,e,g);v['runtime']=time.perf_counter()-tic
 print(json.dumps({k:z for k,z in v.items() if k not in ('selected_orbits','rkeys')},indent=2));open(a.out,'w').write(json.dumps(v,indent=2))
if __name__=='__main__':main()
