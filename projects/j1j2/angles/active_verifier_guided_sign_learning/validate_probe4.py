#!/usr/bin/env python3
"""Independent post-hoc full-H check for AVSL 4x4 frozen-amplitude orbit gains.
Does not enter the acquisition policy or train the learner.
"""
import itertools,json
from pathlib import Path
import discovery_probe4 as P

def full_energy(keys,kappa):
 g=P.Oracle(kappa)
 configs=[sum(1<<i for i in cmb) for cmb in itertools.combinations(range(P.N),P.K)]
 selected=set(keys); norm=0.;num=0.
 for x in configs:
  a,s=g.evaluate(x)
  if P.key(x) in selected:s=-s
  norm+=a*a
  diag=0.
  for i,j,J,hop,mask in P.BONDS:
   if ((x>>i)&1)==((x>>j)&1):diag+=J*.25
   else:
    diag-=J*.25
    y=x^mask;b,sy=g.evaluate(y)
    if P.key(y) in selected:sy=-sy
    num+=hop*s*sy*a*b
  num+=diag*a*a
 return num/norm,norm

def validate(fname):
 data=json.loads(Path(fname).read_text());k=data['kappa']
 allkeys={P.key(sum(1<<i for i in cmb)) for cmb in itertools.combinations(range(P.N),P.K)}
 g=P.Oracle(k);npos=sum(P.score_orbit(g,key)[0]>1.e-8 for key in allkeys)
 orig,norm=full_energy([],k)
 errors=[]
 for row in data['per_seed']:
  for method,values in row['methods'].items():
   after,n2=full_energy(values['selected_keys'],k)
   assert abs(n2-norm)<1e-5
   discrepancy=abs((orig-after)*norm-values['gain'])
   errors.append(discrepancy)
   assert discrepancy<5e-7,(method,discrepancy)
 print(fname,'orbits',len(allkeys),'single-orbit-profitable',npos,'guide energy',orig,'norm',norm,'max discrepancy',max(errors))

if __name__=='__main__':
 import sys
 for fn in sys.argv[1:]:validate(fn)
