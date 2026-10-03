import itertools, json, math, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

L=4; N=L*L
J2S=np.round(np.arange(0.0,1.0001,0.05),2)
q=lambda x,y:(x%L)+L*(y%L)
NN=[]; NNN=[]
for y in range(L):
    for x in range(L):
        i=q(x,y)
        NN += [(i,q(x+1,y)),(i,q(x,y+1))]
        NNN += [(i,q(x+1,y+1)),(i,q(x+1,y-1))]

states=[]
for comb in itertools.combinations(range(N),N//2):
    s=0
    for i in comb:s|=1<<i
    states.append(s)
states=np.asarray(states,np.uint32); dim=len(states)
idx={int(s):i for i,s in enumerate(states)}
print("DIM",dim,flush=True)

def build_piece(bonds):
    rows=[];cols=[];vals=[];diag=np.zeros(dim)
    orows=[];ocols=[];ovals=[]
    for a,s0 in enumerate(states):
        s=int(s0);d=0.
        for i,j in bonds:
            same=((s>>i)&1)==((s>>j)&1)
            d += .25 if same else -.25
            if not same:
                b=idx[s^(1<<i)^(1<<j)]
                rows.append(a);cols.append(b);vals.append(.5)
                orows.append(a);ocols.append(b);ovals.append(.5)
        diag[a]=d
    rows.extend(range(dim));cols.extend(range(dim));vals.extend(diag.tolist())
    return (sp.coo_matrix((vals,(rows,cols)),shape=(dim,dim)).tocsr(),
            sp.coo_matrix((ovals,(orows,ocols)),shape=(dim,dim)).tocsr(),
            diag)
H1,O1,D1=build_piece(NN)
H2,O2,D2=build_piece(NNN)

A_mask=sum(1<<(x+L*y) for y in range(L) for x in range(L) if (x+y)%2==0)
sM=np.asarray([1 if ((int(s)&A_mask).bit_count()%2)==0 else -1 for s in states],float)

def weighted_quantile(x,w,p):
    o=np.argsort(x);xx=np.asarray(x)[o];ww=np.asarray(w)[o]
    c=np.cumsum(ww);c/=c[-1]
    return float(np.interp(p,c,xx))

def otsu(score,w,lo=.01,hi=.99):
    qlo=weighted_quantile(score,w,lo); qhi=weighted_quantile(score,w,hi)
    keep=(score>=qlo)&(score<=qhi);x=score[keep];ww=w[keep]
    o=np.argsort(x);x=x[o];ww=ww[o];ww/=ww.sum()
    cw=np.cumsum(ww);cx=np.cumsum(ww*x);mu=cx[-1]
    good=(cw>1e-7)&(cw<1-1e-7)
    sep=np.full(len(x),-np.inf)
    sep[good]=cw[good]*(1-cw[good])*(cx[good]/cw[good]-(mu-cx[good])/(1-cw[good]))**2
    k=int(np.argmax(sep))
    t=float(.5*(x[k]+x[min(k+1,len(x)-1)]))
    flip=float(np.sum(w[score>t]))
    return t,float(sep[k]),flip

def oracle(score,y,w):
    # correction is +1 below threshold and -1 above threshold.
    # Sort once per orientation and evaluate every threshold by cumulative sums.
    best=(-1.,None,None,None)
    wy=np.asarray(w)*np.asarray(y)
    for orient in (1.,-1.):
        s=orient*np.asarray(score)
        order=np.argsort(s)
        so=s[order]; wo=np.asarray(w)[order]; z=wy[order]
        # threshold below all points: all predictions -1
        ovs=-float(np.sum(z))+2*np.cumsum(z)
        k=int(np.argmax(ovs)); ov=float(ovs[k])
        if k==len(so)-1:
            t=float(so[k]+1e-12)
        else:
            t=float(.5*(so[k]+so[k+1]))
        flipmass=float(1.0-np.sum(wo[:k+1]))
        # also permit the all-minus threshold
        ov_allminus=-float(np.sum(z))
        if ov_allminus>ov:
            ov=ov_allminus; t=float(so[0]-1e-12); flipmass=1.0
        if ov>best[0]:
            best=(ov,t,orient,flipmass)
    return best

rows=[]
v0=None
LAM=np.round(np.arange(-.25,2.5001,.025),3)
for J2 in J2S:
    H=H1+float(J2)*H2
    ev,g=sla.eigsh(H,k=1,which="SA",tol=2e-11,maxiter=200000,v0=v0)
    E0=float(ev[0]);g=np.asarray(g[:,0]);v0=g.copy()
    if np.dot(g,sM)<0:g=-g
    a=np.abs(g);strue=np.where(g>=0,1.,-1.)
    w=a*a;w/=w.sum()
    y=(sM*strue).astype(int)
    OM=float(np.sum(w*y)); wrongM=float(np.sum(w[y<0]))

    psi=a*sM
    eps=1e-300
    safe=np.where(np.abs(psi)>eps,psi,np.sign(psi)*eps+eps)
    F1=np.asarray(O1@psi)/safe
    F2=np.asarray(O2@psi)/safe
    r=(np.asarray(H@psi)/safe)

    # Physical one-step label-free Otsu correction.
    t,sep,flip=otsu(r,w)
    p=np.where(r<=t,1,-1)
    OK=float(np.sum(w*p*y)); wrongK=float(np.sum(w[p!=y]))
    removed=(wrongM-wrongK)/wrongM if wrongM>1e-15 else np.nan

    sweep=[]
    for lam in LAM:
        sc=F1+float(lam)*F2
        ov,tt,orient,fm=oracle(sc,y,w)
        sweep.append((float(lam),float(ov),float(tt),float(orient),float(fm)))
    best=max(sweep,key=lambda z:z[1])
    phys=min(sweep,key=lambda z:abs(z[0]-J2))

    # Margin diagnostics for the physical off-diagonal field.
    sc=F1+float(J2)*F2
    neg=(y<0);pos=~neg
    if np.any(neg) and np.any(pos):
        mu_pos=float(np.sum(w[pos]*sc[pos])/np.sum(w[pos]))
        mu_neg=float(np.sum(w[neg]*sc[neg])/np.sum(w[neg]))
        var_pos=float(np.sum(w[pos]*(sc[pos]-mu_pos)**2)/np.sum(w[pos]))
        var_neg=float(np.sum(w[neg]*(sc[neg]-mu_neg)**2)/np.sum(w[neg]))
        pooled=math.sqrt(max(.5*(var_pos+var_neg),1e-300))
        effect=abs(mu_pos-mu_neg)/pooled
    else:
        mu_pos=mu_neg=effect=float("nan")

    row=dict(J2=float(J2),E0=E0,MarshallO=OM,MarshallWrong=wrongM,
             K1O=OK,K1Wrong=wrongK,K1RemovedFraction=removed,
             K1Threshold=t,OtsuSeparation=sep,K1FlipMass=flip,
             lambda_best=best[0],lambda_best_O=best[1],
             lambda_physical=phys[0],lambda_physical_O=phys[1],
             lambda_gap=float(best[0]-J2),
             physical_field_effect_size=effect,
             physical_field_mu_correct=mu_pos,
             physical_field_mu_wrong=mu_neg)
    rows.append(row)
    print("ROW",json.dumps(row,sort_keys=True),flush=True)

with open("frustration_scan_exact4x4.json","w") as f:json.dump(rows,f,indent=2)
print("DONE",flush=True)
