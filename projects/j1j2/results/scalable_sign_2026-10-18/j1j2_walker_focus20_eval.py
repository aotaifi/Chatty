import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,torch,numpy as np
sys.path.insert(0,'/mnt/data')
import j1j2_walker_only_amplitude20 as W
from j1j2_walker_only_amplitude20 import M,T
P='/mnt/data/'
fn=np.load(P+'j1j2_walker_only_fn_truth_audit20.npz')['fn']; logfn=np.log(fn)
selections=json.load(open(P+'j1j2_walker_only_amplitude20_eval.json'))['sign_trials']
s=W.LocalModelSampler(W.CountedGuide());keys=sorted({int(g) for r in selections for a in r['arms'].values() for g in a['selected_keys']});pairs=[]
for g in keys:
 s.orbit(g)
 for y,h in s.neighbors(g):
  if s.orbit(y)!=g:pairs.append((int(s.idx(g)),int(s.idx(y))))
idx=np.unique(np.array(pairs).ravel());rng=np.random.default_rng(41);ix=rng.choice(W.D,8000,p=fn**2);j=rng.integers(W.NB,size=len(ix));y=M.states[ix]^W.MASKS[j];iy=M.lookup[y];valid=iy>=0;ix=ix[valid];iy=iy[valid]
res={}
for tag in ('','_lowbudget','_m128','_focus'):
 name='j1j2_walker_focus20_model.pt' if tag=='_focus' else 'j1j2_walker_only_amplitude20'+tag+'_model.pt'
 ck=torch.load(P+name,weights_only=True,map_location='cpu');net=T.Net();net.load_state_dict(ck['model']);net.eval()
 pred=np.empty(W.D)
 with torch.no_grad():
  for start in range(0,W.D,512):pred[start:start+512]=W.LOGG0[start:start+512]+net.symmetric(T.X[start:start+512]).numpy()
 err=pred[iy]-pred[ix]-(logfn[iy]-logfn[ix]);err2=np.array([pred[a]-pred[b]-(logfn[a]-logfn[b]) for a,b in pairs]);q=np.exp(pred-max(pred));q/=np.linalg.norm(q)
 res[tag or 'base']=dict(bulk_RMS=float(np.sqrt(np.mean(err**2))),selected_bond_RMS=float(np.sqrt(np.mean(err2**2))),selected_bond_p90=float(np.quantile(abs(err2),.9)),overlap=float(fn@q))
print(json.dumps(res,indent=2));open(P+'j1j2_walker_focus20_eval.json','w').write(json.dumps(res,indent=2))