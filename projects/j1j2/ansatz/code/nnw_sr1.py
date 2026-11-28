"""nnw_sr1.py: copy of ref2hop/NNw/nnw.py (stage 2) with only the inputs changed, for STORAGE_PROBLEM.md 'WHICH INPUT DOES THE S_R1 COPY MISS?, REGISTERED'.
usage: nnw_sr1.py OUT.  d = chi_1 - T_1 (round 0 of cycle6, S_R1). Base = fit0's 12 inputs (nn_fit_jax.py preamble: 8 moments incl. lnB, h, ref_parts l_part, mapP0 l_part, + c1_0 as NN_EXTRA2).
Column sets: base, +c2_0, +chi_0, +c2_0+chi_0, +d (known case), +shuffled c2_0 (broken). cost_left and r dropped (no gradient for this round).
S: no g exists for this round; S is the old NNg/g.npy top-|g| 95% set (as in nnw.py), used only to define the row set for R2_S."""
import os, sys, json, time, numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata
C='/scratch/a/A.Otaifi/tmp/wt_ratnet/projects/j1j2/ansatz/code'; sys.path.insert(0, C)
import bond_split as BS
OO='/scratch/a/A.Otaifi/ansatz_runs/consol_6x6/ref2hop'; CS='/scratch/a/A.Otaifi/ansatz_runs/consol_6x6'; OUT=sys.argv[1]; os.makedirs(OUT,exist_ok=True)
MAP=f'{CS}/cycle6_store/map0'; LPART=f'{CS}/cycle6_store/mapP0/l_part.npy'; T0=time.time()
P_CHI1=f'{CS}/cycle6f_store/fit0/fit/tables/t_R.npy'; P_T1=f'{MAP}/lA_next.npy'; P_CHI0=f'{OO}/NNx/x1/x1fit/tables/t_R.npy'; P_C10=f'{MAP}/l_part.npy'
import hashlib
for p_,h_ in ((P_CHI1,'e40d974e'),(P_T1,'2e0985d3'),(P_CHI0,'4db4075d'),(P_C10,'ce35da48'),(LPART,'43ac77cb')):
    m_=hashlib.md5(); f_=open(p_,'rb')
    for b_ in iter(lambda: f_.read(1<<24),b''): m_.update(b_)
    assert m_.hexdigest().startswith(h_),(p_,m_.hexdigest()); print('md5 ok',h_,p_,flush=True)
def log(m): print(f'[{time.time()-T0:7.0f}s] {m}', flush=True)
LP=f'{OO}/ref_parts/l_part.npy'; CH1='/scratch/a/A.Otaifi/ansatz_runs/consol_6x6/ratnet/lin_s1/la_table.npy'
MOM='/scratch/a/A.Otaifi/ansatz_runs/consol_6x6/round2/moments'; A0='/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy'
n=np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D=len(n); assert D==15804956 and n.sum()==9075135300
lB=np.load(A0).astype(np.float64); pt=np.load(f'{MAP}/p_out.npy').astype(np.float64)
assert abs((n*pt*pt).sum()-1)<1e-9
M=np.load(f'{MOM}/moments.npz'); NAMES=['mu_H','mu_K','mu_B','lnB','mu_KH','mu_BH','mu_KK','mu_BB']
F=np.stack([M[k] if k!='lnB' else lB for k in NAMES],1); assert np.isfinite(F).all(); del M
h=np.load(CH1).astype(np.float64)-lB; F=np.concatenate([F,h[:,None]],1)
for pth in (LP,LPART,P_C10):
    ex=np.load(pth).astype(np.float64); assert ex.shape==(D,) and np.isfinite(ex).all(); F=np.concatenate([F,ex[:,None]],1)
assert F.shape[1]==12
wv=n*pt*pt; w=wv.copy()
Zr=np.empty(F.shape,np.float32)
for j in range(F.shape[1]): Zr[:,j]=(rankdata(F[:,j],method='average')/D).astype(np.float32)
del F; log(f'rank-transformed base12 {Zr.shape}, range [{Zr.min():.3g},{Zr.max():.3g}]')
g=np.load(f'{OO}/NNg/g.npy'); assert g.shape==(D,)
ag=np.abs(g); o=np.argsort(-ag); cs=np.cumsum(ag[o]); S=np.sort(o[:np.searchsorted(cs,0.95*cs[-1])+1]); log(f'|S|={len(S)}; sum|g| share {ag[S].sum()/ag.sum():.4f}')
den=float((g[S]**2/w[S]).sum())
chi1=np.load(P_CHI1).astype(np.float64); T1=np.load(P_T1).astype(np.float64); chi0=np.load(P_CHI0).astype(np.float64); c10=np.load(P_C10).astype(np.float64)
for a_ in (chi1,T1,chi0): assert a_.shape==(D,) and np.isfinite(a_).all()
d=chi1-T1; c2=T1-chi0-c10; log(f'd range [{d.min():.4g},{d.max():.4g}] c2 range [{c2.min():.4g},{c2.max():.4g}] chi0 range [{chi0.min():.4g},{chi0.max():.4g}]'); del chi1,T1,c10
rng=np.random.default_rng(11); SM=np.sort(rng.choice(D,200000,replace=False)); Q=np.union1d(S,SM); log(f'|sample|={len(SM)} |Q|={len(Q)} overlap {len(S)+len(SM)-len(Q)}')
inS=np.searchsorted(Q,S); inM=np.searchsorted(Q,SM)
perm=np.random.default_rng(7).permutation(D)
COLS=[('base',[]),('base+c2',[c2]),('base+chi0',[chi0]),('base+c2+chi0',[c2,chi0]),('base+d',[d]),('base+c2shuf',[c2[perm]])]
KS=[16,64]; KM=max(KS)+1; res=dict(S=len(S),n_sample=len(SM),den=den)
def wR2(dv,dh,ww): 
    db=(ww*dv).sum()/ww.sum(); return float(1-(ww*(dv-dh)**2).sum()/(ww*(dv-db)**2).sum())
def knn_nb(ZZ,rows,tag):
    t=time.time(); tree=cKDTree(ZZ.astype(np.float64),leafsize=32,balanced_tree=False,compact_nodes=False)
    dist,idx=tree.query(ZZ[rows].astype(np.float64),k=KM+1,workers=8); del tree
    nb=np.empty((len(rows),KM),np.int64); nself=0
    for r in range(len(rows)):
        row=idx[r]; hit=np.nonzero(row==rows[r])[0]
        if len(hit): row=np.delete(row,hit[0]); nself+=1
        else: row=row[:KM]
        nb[r]=row[:KM]
    log(f'{tag}: tree+query {time.time()-t:.0f}s; self found {nself}/{len(rows)}'); return nb
for name,c in COLS:
    ZZ=np.concatenate([Zr]+[(rankdata(c_,method='average')/D).astype(np.float32)[:,None] for c_ in c],1)
    nb=knn_nb(ZZ,Q,name); del ZZ
    for k in KS:
        nk=nb[:,:k]; dh=(w[nk]*d[nk]).sum(1)/w[nk].sum(1)          # w-weighted mean of d over neighbours
        dS=d[S]; dhS=dh[inS]; wS=w[S]; dM=d[SM]; dhM=dh[inM]; wM=w[SM]
        res[f'{name}_k{k}']=dict(R2_sample=wR2(dM,dhM,wM),R2_S=wR2(dS,dhS,wS))
        log(f'{name} k={k}: {res[f"{name}_k{k}"]}')
    json.dump(res,open(f'{OUT}/nnw.json','w'),indent=1)
log('DONE nnw'); open(f'{OUT}/DONE_ALL','w').write('ok')
