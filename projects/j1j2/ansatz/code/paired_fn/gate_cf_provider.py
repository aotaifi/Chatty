import sys,hashlib,numpy as np
sys.path.insert(0,'/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code/paired_fn')
import cnn_sign as CS, r1_lib as R
B='/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad/'
z=np.load('/scratch/a/A.Otaifi/runs_bulk/sign_bad/tcf/table_r3_cf.npz'); keys=z['keys']
lo=np.load(B+'step8/cf_logits_all.npy').astype(np.float64)
rg=np.random.default_rng(5); idx=np.sort(rg.choice(len(keys),2000,replace=False)); k=keys[idx]
ref=(R.marshall_pm(k)*np.where(lo[idx]>=0,1,-1)).astype(np.int8)
# replicate branched_prod resolution of --netB cf
import re; src=open('/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code/paired_fn/branched_prod.py').read()
assert "cf='5dcca951'" in src
n=CS.NetSign(B+'tcf/CF/fit/net_8192.pkl','nB'); assert n.sha.startswith('5dcca951')
s=n.sign(k); mis=int((s!=ref).sum()); print('GATE correct: provider vs sgn(cf_logits_all): mismatches',mis,'of 2000 (flip fraction of ref vs Marshall:',float((ref!=R.marshall_pm(k)).mean()),')')
bad=(-s).astype(np.int8); mb=int((bad!=ref).sum()); print('GATE broken (sign flipped): mismatches',mb,'of 2000')
n2=CS.NetSign(B+'r2/fit/net_8192.pkl','nA'); s2=n2.sign(k); print('control: n_2 vs CF-ref mismatches',int((s2!=ref).sum()))
print('GATE', 'PASS' if mis==0 and mb==2000 else 'FAIL')
