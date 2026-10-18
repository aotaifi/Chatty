import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
src=Path('/mnt/data/fn_collapse_exact20_krylov_v2.py').read_text()
exec(src[:src.index('import sys\nNITER=')])
d=np.load('/mnt/data/sign20_training_data.npz');aa=d['amp'];ss=d['guide']
E,b,info=fn_step(aa,ss,'measured_next_FN',tol=1e-8)
rate=np.log(np.maximum(aa,1e-220))-np.log(np.maximum(b,1e-220))
np.savez_compressed('/mnt/data/sign20_suppression.npz',rate=rate,next_amp=b,E=E)
print('SAVED',E,info,'quantiles',np.percentile(rate,[1,10,50,90,99]),flush=True)