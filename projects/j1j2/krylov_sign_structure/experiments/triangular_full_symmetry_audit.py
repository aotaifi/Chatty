import json
from pathlib import Path
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla

LX,LY=6,3; N=LX*LY
OUT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/results")
site=lambda x,y:(x%LX)+LX*(y%LY)

# triangular NN graph
B=[]
for y in range(LY):
    for x in range(LX):
        i=site(x,y)
        B += [(i,site(x+1,y)),(i,site(x,y+1)),(i,site(x+1,y+1))]
BSET={tuple(sorted(e)) for e in B}

basis=np.array([s for s in range(1<<N) if s.bit_count()==N//2],np.uint32)
D=len(basis); idx={int(s):i for i,s in enumerate(basis)}

# H
rr=[];cc=[];vv=[];diag=np.zeros(D)
for bi,s0 in enumerate(basis):
    s=int(s0);e=0.
    for u,v in B:
        if ((s>>u)&1)==((s>>v)&1): e+=.25
        else:
            e-=.25
            t=s^(1<<u)^(1<<v)
            rr.append(bi);cc.append(idx[t]);vv.append(.5)
    diag[bi]=e
rr.extend(range(D));cc.extend(range(D));vv.extend(diag.tolist())
H=sp.coo_matrix((vv,(rr,cc)),shape=(D,D)).tocsr()
ev,V=sla.eigsh(H,k=1,which="SA",tol=2e-11,maxiter=100000)
v=np.asarray(V[:,0],float); anchor=int(np.argmax(abs(v)))
if v[anchor]<0:v=-v
a=np.abs(v); p=a*a; p/=p.sum(); truth=np.where(v>=0,1,-1).astype(np.int8)

# gauges
def gauge(mask):
    return np.array([1 if ((int(s)&mask).bit_count()%2)==0 else -1 for s in basis],np.int8)
parity_y=gauge(sum(1<<site(x,y) for y in range(LY) for x in range(LX) if y%2==0))

# grouped projected-Krylov iteration to fixed point
co=sp.triu(H-sp.diags(diag),k=1).tocoo()
ei=co.row.astype(np.int32);ej=co.col.astype(np.int32);w=co.data*a[ei]*a[ej]
def canon(s):
    s=s.copy()
    if s[anchor]<0:s=-s
    return s
def gids(r):
    o=np.argsort(r,kind="mergesort");rs=r[o]
    sc=np.maximum(np.maximum(np.abs(rs[1:]),np.abs(rs[:-1])),1.)
    br=np.empty(D,bool);br[0]=1
    br[1:]=np.abs(rs[1:]-rs[:-1])>(1e-11+1e-11*sc)
    gs=np.cumsum(br,dtype=np.int32)-1
    g=np.empty(D,np.int32);g[o]=gs
    return g,int(gs[-1])+1
def upd(s):
    psi=a*s
    r=(H@psi)/np.where(a>1e-300,psi,1.)
    g,G=gids(r);c0=-s
    val=2*w*c0[ei]*c0[ej]
    e0=float(np.sum(a*a*diag)+np.sum(val))
    gi=g[ei];gj=g[ej];lo=np.minimum(gi,gj);hi=np.maximum(gi,gj);m=lo<hi
    de=np.bincount(lo[m],weights=-2*val[m],minlength=G)+np.bincount(hi[m],weights=2*val[m],minlength=G)
    es=e0+np.cumsum(de);k=int(np.argmin(np.r_[e0,es]))-1
    sn=c0.copy()
    if k>=0:sn[g<=k]*=-1
    return canon(sn)
sfix=canon(parity_y)
for it in range(50):
    sn=upd(sfix)
    if np.array_equal(sn,sfix): break
    sfix=sn
print("fixed_iter",it,"O",abs(np.sum(p*sfix*truth)),flush=True)

# enumerate all affine site automorphisms f(x,y)=(a*x+b*y+u mod6, c*x+d*y+v mod3)
syms=[]
seen_maps=set()
for aa in range(LX):
  for bb in range(LX):
    for cc0 in range(LY):
      for dd in range(LY):
        base=[]
        ok=True
        for y in range(LY):
          for x in range(LX):
            xp=(aa*x+bb*y)%LX
            yp=(cc0*x+dd*y)%LY
            base.append(site(xp,yp))
        if len(set(base))<N: continue
        # require graph automorphism
        image_edges={tuple(sorted((base[u],base[v]))) for u,v in BSET}
        if image_edges!=BSET: continue
        for u0 in range(LX):
          for v0 in range(LY):
            mp=[]
            for y in range(LY):
              for x in range(LX):
                xp=(aa*x+bb*y+u0)%LX
                yp=(cc0*x+dd*y+v0)%LY
                mp.append(site(xp,yp))
            key=tuple(mp)
            if key in seen_maps: continue
            seen_maps.add(key)
            image_edges={tuple(sorted((mp[u],mp[v]))) for u,v in BSET}
            if image_edges!=BSET: continue
            syms.append(dict(a=aa,b=bb,c=cc0,d=dd,u=u0,v=v0,map=mp))
print("num_syms",len(syms),flush=True)

# config permutation induced by site map
def cfg_perm(mp):
    perm=np.empty(D,np.int32)
    for bi,s0 in enumerate(basis):
        s=int(s0);t=0
        for i,j in enumerate(mp):
            if (s>>i)&1:t|=1<<j
        perm[bi]=idx[t]
    return perm

def eigchar(psi,perm):
    tv=np.empty_like(psi);tv[perm]=psi
    den=float(np.dot(psi,psi))
    c=float(np.dot(psi,tv)/den)
    res=float(np.linalg.norm(tv-c*psi)/np.linalg.norm(psi))
    return c,res

states={
    "ground":a*truth,
    "parity_y_start":a*canon(parity_y),
    "parity_y_fixed":a*sfix,
}
rows=[]
mismatches=[]
for si,sym in enumerate(syms):
    perm=cfg_perm(sym["map"])
    ca,ra=eigchar(a,perm)
    if ra>1e-8 or abs(ca-1)>1e-8: continue
    rec={"id":si,**{k:sym[k] for k in ("a","b","c","d","u","v")},"amp_res":ra}
    chars={}
    for name,psi in states.items():
        c,r=eigchar(psi,perm);chars[name]=(c,r)
        rec[name+"_char"]=c;rec[name+"_res"]=r
    rows.append(rec)
    # exact eigenstate mismatch start vs ground or fixed vs ground
    cg,rg=chars["ground"];cs,rs=chars["parity_y_start"];cf,rf=chars["parity_y_fixed"]
    if rg<1e-8 and rs<1e-8 and abs(cg-cs)>1e-6:
        mismatches.append(("start",rec))
    if rg<1e-8 and rf<1e-8 and abs(cg-cf)>1e-6:
        mismatches.append(("fixed",rec))

print("amp-invariant syms",len(rows),"mismatches",len(mismatches),flush=True)
for kind,r in mismatches[:30]:
    print("MISMATCH",kind,{k:r[k] for k in ("id","a","b","c","d","u","v","ground_char","parity_y_start_char","parity_y_fixed_char","ground_res","parity_y_start_res","parity_y_fixed_res")},flush=True)

# also list exact eigen symmetries for start/fixed regardless mismatch
for nm in ("parity_y_start","parity_y_fixed"):
    exact=[]
    for r in rows:
        if r[nm+"_res"]<1e-8:
            exact.append({k:r[k] for k in ("id","a","b","c","d","u","v","ground_char","ground_res",nm+"_char",nm+"_res")})
    print("EXACT",nm,"count",len(exact),flush=True)
    for e in exact[:40]: print(e,flush=True)

OUT.joinpath("triangular_full_symmetry_audit.json").write_text(json.dumps(dict(
    fixed_iterations=it,
    fixed_overlap=float(abs(np.sum(p*sfix*truth))),
    num_symmetries=len(syms),
    amp_invariant=rows,
    mismatches=[dict(kind=k,rec=r) for k,r in mismatches]
),indent=2))
