#!/usr/bin/env python3
"""Independent post-hoc full-H check of the 4x4 callable-guide orbit-flip score.

Only used for validation; NOT used to select candidates in the experiment.
"""
import argparse,itertools,json,numpy as np
import avsl_probe4 as p

def full_energy(keys,kappa):
    g=p.Oracle(kappa);z=set(keys);norm=0.;numer=0.
    for cmb in itertools.combinations(range(p.N),p.K):
        x=sum(1<<i for i in cmb)
        a,s=g.evaluate(x)
        if p.key(x) in z:s=-s
        norm+=a*a
        for i,j,J,h,mask in p.BONDS:
            if ((x>>i)&1)==((x>>j)&1):
                numer+=J*.25*a*a
            else:
                numer-=J*.25*a*a
                y=x^mask;b,sy=g.evaluate(y)
                if p.key(y) in z:sy=-sy
                numer+=h*s*sy*a*b
    return numer/norm,norm

def validate(kappa,B=40,seed=101):
    base,pool,scores,meta=p.generate(seed,n_seeds=180,kappa=kappa)
    prior,norm=full_energy([],kappa)
    print('Guide kappa',kappa,'full-space energy',prior,'norm',norm)
    for method in ('random','observed_signed','max_violation'):
        res=p.evaluate_choice(base,pool,scores,method,seed,B=B)
        after,_=full_energy(res['selected_keys'],kappa)
        delta=(prior-after)*norm
        assert abs(delta-res['gain'])<5.e-7,(method,delta,res['gain'])
        print(method,'verified E gain',prior-after,'raw numerator gain',res['gain'])
    return True

if __name__=='__main__':
 ap=argparse.ArgumentParser()
 ap.add_argument('--kappas',type=float,nargs='+',default=[0.32,0.9])
 a=ap.parse_args()
 for k in a.kappas:validate(k)
