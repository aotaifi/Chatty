import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
src=Path('/mnt/data/fn_collapse_exact20_krylov_v2.py').read_text()
exec(src[:src.index('import sys\nNITER=')])
d=np.load('/mnt/data/sign20_training_data.npz'); a=d['amp'];s=d['guide'];old=d['flip'].astype(bool)
new,info=krylov_step(a,s,'audit_current_sign')
actual=new!=s
print('KRYLOV',info,flush=True)
print('LABEL_COMPARISON',{'g>0':int(old.sum()),'energy_opt_flip':int(actual.sum()),'disagree':int(np.sum(actual!=old)),'extra_g':int(np.sum(old & ~actual)),'missed_g':int(np.sum(actual & ~old)),'disagree_physical_weight':float(np.sum(psi0[actual!=old]**2))},flush=True)
np.savez_compressed('/mnt/data/audit_sign20_targets.npz',optimal_flip=actual.astype('int8'),old_flip=old.astype('int8'))