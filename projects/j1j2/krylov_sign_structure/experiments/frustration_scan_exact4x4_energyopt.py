import itertools, json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

L=4; N=16
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

def build_piece(bonds):
    rows=[];cols=[];vals=[];diag=np.zeros(dim)
    for a,s0 in enumerate(states):
        s=int(s0);d=0.
        for i,j in bonds:
            same=((s>>i)&1)==((s>>j)&1)
            d += .25 if same else -.25
            if not same:
                rows.append(a);cols.append(idx[s^(1<<i)^(1<<j)]);vals.append(.5)
        diag[a]=d
    rows.extend(range(dim));cols.extend(range(dim));vals.extend(diag.tolist())
    return sp.coo_matrix((vals,(rows,cols)),shape=(dim,dim)).tocsr()
H1=build_piece(NN);H2=build_piece(NNN)
A_mask=sum(1<<(x+L*y) for y in range(L) for x in range(L) if (x+y)%2==0)
sM=np.asarray([1. if ((int(s)&A_mask).bit_count()%2)==0 else -1. for s in states])

def energy_opt_threshold(H,a,r):
    W=H.copy().tolil(); W.setdiag(0); W=W.tocsr()
    W=sp.diags(a)@W@sp.diags(a)
    diagE=float(np.sum(H.diagonal()*a*a))
    s=-sM.copy()  # globally equivalent to Marshall; threshold below all
    E=diagE+float(s@(W@s))
    h=np.asarray(W@s).ravel()
    order=np.argsort(r)
    bestE=float(E); bestk=-1; bests=s.copy()
    for k,ii in enumerate(order):
        old=s[ii]
        E += -4.0*old*h[ii]
        s[ii]=-old
        st,en=W.indptr[ii],W.indptr[ii+1]
        js=W.indices[st:en]; ws=W.data[st:en]
        h[js] += -2.0*old*ws
        if E < bestE-1e-13:
            bestE=float(E); bestk=k; bests=s.copy()
    if bestk<0:
        t=float(r[order[0]]-1e-12)
    elif bestk==len(order)-1:
        t=float(r[order[-1]]+1e-12)
    else:
        t=float(.5*(r[order[bestk]]+r[order[bestk+1]]))
    return bestE,t,bests

rows=[];v0=None
for J2 in J2S:
    H=H1+float(J2)*H2
    ev,g=sla.eigsh(H,k=1,which='SA',tol=2e-11,maxiter=200000,v0=v0)
    E0=float(ev[0]);g=np.asarray(g[:,0]);v0=g.copy()
    if np.dot(g,sM)<0:g=-g
    a=np.abs(g);strue=np.where(g>=0,1.,-1.)
    p=a*a;p/=p.sum()
    psiM=a*sM
    r=np.asarray((H@psiM)/np.where(np.abs(psiM)>1e-300,psiM,1e-300),float)
    EM=float(psiM@(H@psiM))
    OM=float(np.sum(p*sM*strue)); wrongM=float(np.sum(p[sM!=strue]))
    Eb,t,sb=energy_opt_threshold(H,a,r)
    # global sign irrelevant: orient to true state for diagnostics
    if np.sum(p*sb*strue)<0: sb=-sb
    O=float(np.sum(p*sb*strue)); wrong=float(np.sum(p[sb!=strue]))
    removed=(wrongM-wrong)/wrongM if wrongM>1e-15 else np.nan
    flipmass=float(min(np.sum(p[sb!=sM]),np.sum(p[sb!=-sM])))
    row=dict(J2=float(J2),E0=E0,E_Marshall=EM,E_opt=Eb,
             deltaE=float(Eb-EM),MarshallO=OM,MarshallWrong=wrongM,
             EnergyOptO=O,EnergyOptWrong=wrong,
             EnergyOptRemovedFraction=removed,EnergyOptFlipMass=flipmass,
             threshold=t)
    rows.append(row);print('ROW',json.dumps(row,sort_keys=True),flush=True)
with open('frustration_scan_exact4x4_energyopt.json','w') as f:json.dump(rows,f,indent=2)
print('DONE',flush=True)
