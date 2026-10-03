#!/usr/bin/env python3
import json, numpy as np
from finite_tau_ct_local_ratio_20site import H,normalized
z=np.load("../results/reconstruct_M100000_y59279.npz")
s=z["s"].astype(float);lg=z["logg"].astype(float)
g=normalized(np.exp(np.clip(lg-lg.max(),-700,0)));psi=s*g;dt=.0125
h=H@psi;psi2=normalized(psi-dt*h+.5*dt*dt*(H@h))
p=psi2*psi2;p/=p.sum();m=p>0
out={"entropy_nats":float(-np.sum(p[m]*np.log(p[m]))),
     "perplexity":float(np.exp(-np.sum(p[m]*np.log(p[m])))),
     "support_1e6":int(np.sum(p>1e-6)),"support_1e5":int(np.sum(p>1e-5))}
print(json.dumps(out))
