from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
src=Path('/mnt/data/fn_collapse_exact20_krylov_v2.py').read_text()
exec(src[:src.index('import sys\nNITER=')])
a=a0.copy();s=sM.copy()
for k in range(6): _,a,_=fn_step(a,s,f'pre{k}',tol=1e-8)
s,_=krylov_step(a,s,'krylov')
for k in range(3): _,a,_=fn_step(a,s,f'post{k}',tol=1e-8)
g=s*(Hoff@(s*a));flip=g>0
np.savez_compressed('/mnt/data/sign20_training_data.npz',states=states,amp=a,guide=s.astype('int8'),flip=flip.astype('int8'),psi0=psi0,diag=diag,maskA=np.array([maskA]))
print('SAVED',int(flip.sum()),flush=True)