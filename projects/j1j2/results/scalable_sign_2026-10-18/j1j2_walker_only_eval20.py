#!/usr/bin/env python3
"""Evaluation-only exact 20-site FN diagonalization for analytic-guide walker experiment.
No exact FN target is accessed by walker or training scripts; this is an audit.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import j1j2_walker_only_amplitude20 as W
from j1j2_walker_only_amplitude20 import M, T
P=Path('/mnt/data');D=M.D

def build_fn():
 states=M.states;ix=np.arange(D);diag=M.diag.copy();rows=[];cols=[];values=[]
 for mask,h,i,j in M.bonds:
  ys=states^int(mask);iy=M.lookup[ys];valid=iy>=0
  good=np.flatnonzero(valid);jidx=iy[good]
  k=h*W.SIGNS[good]*W.SIGNS[jidx]
  ratios=np.exp(W.LOGG0[jidx]-W.LOGG0[good])
  pos=k>0;neg=k<0
  diag[good[pos]]+=k[pos]*ratios[pos]
  rows.append(good[neg]);cols.append(jidx[neg]);values.append(k[neg])
 rows.append(ix);cols.append(ix);values.append(diag)
 H=sp.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(D,D)).tocsr()
 diff=H-H.T;assert not diff.nnz or np.max(abs(diff.data))<1e-12
 return H

def solve_eval():
 t=time.time();H=build_fn();print('FN CSR',H.shape,'nnz',H.nnz,'built',round(time.time()-t,1),flush=True)
 E,V=eigsh(H,k=1,which='SA',tol=3e-9,maxiter=2000,v0=W.G0/np.linalg.norm(W.G0))
 a=np.abs(V[:,0]);a/=np.linalg.norm(a)
 assert a.min()>0,'Unexpected zeros';res=np.linalg.norm(H@a-E[0]*a)
 print('FN E',E[0],'res',res,'time',time.time()-t,flush=True)
 return a,float(E[0]),float(res)

def orbit_true_gain(groups,s,amp):
 selected=set(groups);acc=0.
 for g in selected:s.orbit(int(g))
 for g in selected:
  n=len(s.members[g]);ix=int(s.idx(g));ax=amp[ix];sx=int(W.SIGNS[ix])
  for y,h in s.neighbors(g):
   if s.orbit(y) in selected:continue
   acc+=4*n*h*ax*amp[s.idx(y)]*sx*int(W.SIGNS[s.idx(y)])
 return acc

def run():
 cache=P/'j1j2_walker_only_fn_truth_audit20.npz'
 if cache.exists():
  d=np.load(cache);fn,E,res=d['fn'],float(d['E']),float(d['res'])
 else:fn,E,res=solve_eval()
 ck=torch.load(P/'j1j2_walker_only_amplitude20_model.pt',map_location='cpu',weights_only=True)
 net=T.Net();net.load_state_dict(ck['model']);net.eval()
 pred=np.empty(D,np.float64)
 with torch.no_grad():
  for i in range(0,D,512):
   pred[i:i+512]=W.LOGG0[i:i+512]+net.symmetric(T.X[i:i+512]).numpy()
 logfn=np.log(fn);pred-=np.max(pred)
 guide=np.exp(np.clip(pred,-100,60));guide/=np.linalg.norm(guide)
 g0=W.G0.copy();g0/=np.linalg.norm(g0)
 # Exact FN-probability weighted local-bond metric, only held-out post-hoc.
 rng=np.random.default_rng(41);idx=rng.choice(D,8000,p=fn*fn);j=rng.integers(W.NB,size=len(idx));y=M.states[idx]^W.MASKS[j];iy=M.lookup[y];valid=iy>=0
 idx=idx[valid];iy=iy[valid]
 exact=logfn[iy]-logfn[idx]
 nn=(pred[iy]-pred[idx]);prior=W.LOGG0[iy]-W.LOGG0[idx]
 metrics=dict(fn_energy=E,fn_residual=res,n_bonds=len(idx),nn_rmse=float(np.sqrt(np.mean((nn-exact)**2))),jastrow_rmse=float(np.sqrt(np.mean((prior-exact)**2))),nn_mae=float(np.mean(abs(nn-exact))),jastrow_mae=float(np.mean(abs(prior-exact))),nn_overlap=float(np.dot(guide,fn)),jastrow_overlap=float(np.dot(g0,fn)),nn_logratio_p95=float(np.quantile(abs(nn-exact),.95)))
 trials=json.loads((P/'j1j2_walker_only_amplitude20_results.json').read_text())['sign_trials']
 for r in trials:
  ss=W.LocalModelSampler(W.CountedGuide())
  for m in ('fitted','jastrow'):
   sel=r['arms'][m]['selected_keys']
   true=orbit_true_gain(sel,ss,fn)
   r['arms'][m]['true_fn_amplitude_gain']=float(true)
 print('METRICS',json.dumps(metrics),flush=True)
 print('TRIAL_TRUE',[(r['seed'],r['arms']['fitted']['true_fn_amplitude_gain'],r['arms']['jastrow']['true_fn_amplitude_gain']) for r in trials],flush=True)
 results=dict(exact_FN_evaluation=metrics,sign_trials=trials,warning='All exact FN amplitudes are evaluation-only; FN operator depends on analytic reference g0; fixed-amplitude corrections, not self-consistent FN energy.')
 (P/'j1j2_walker_only_amplitude20_eval.json').write_text(json.dumps(results,indent=2))
 np.savez_compressed(P/'j1j2_walker_only_fn_truth_audit20.npz',fn=fn)
if __name__=='__main__':run()