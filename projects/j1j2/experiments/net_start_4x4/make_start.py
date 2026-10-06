#!/usr/bin/env python3
"""checkpoint (vmc_train_4x4.py) -> guide npz {a=|psi|, s=sgn Re(psi e^{-i th0}), psi} on the S^z=0 basis,
plus its score vs ED and exact E_FN of the guide."""
import argparse, json
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d
from common import *
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--layers', type=int, required=True)
ap.add_argument('--d-model', type=int, required=True); ap.add_argument('--heads', type=int, required=True)
ap.add_argument('--marshall', type=int, default=0)
ap.add_argument('--out', required=True)
a_ = ap.parse_args()
X = bits2x(BASIS)
model = ViT(num_layers=a_.layers, d_model=a_.d_model, heads=a_.heads, L_eff=4, b=B, transl_invariant=True, two_dimensional=True)
p0 = model.init(jax.random.PRNGKey(0), jnp.asarray(X[:2]))["params"]
flat0, unravel = ravel_pytree(p0)
flat = jnp.asarray(np.load(a_.ckpt)['flat']); assert flat.size == flat0.size
MA = jnp.asarray(np.array([1.0 if (xx + yy) % 2 == 0 else 0.0 for yy in range(L) for xx in range(L)]))
def lp(z, x):
    o = _logpsi_transl_2d(model.apply, B, {"params": unravel(z)}, x)
    if a_.marshall:
        o = o + 1j * jnp.pi * jnp.mod(jnp.sum(MA * (x + 1.0) / 2.0, axis=-1), 2.0)
    return o
f = jax.jit(lp)
lp = np.concatenate([np.asarray(f(flat, jnp.asarray(X[i:i + 1024]))) for i in range(0, D, 1024)])
psi = np.exp(lp - lp.real.max())
amp, sg, imf = guide_from_psi(psi)
H, diag, ei, ej, hij = build_H(0.5)
E0, psi0 = ground(H, tol=1e-12)
sM = canonical(marshall())
if np.dot(psi0, sM) < 0: psi0 = -psi0
strue = canonical(np.where(psi0 >= 0, 1, -1).astype(np.int8))
pg = amp * sg
Eg = float(pg @ (H @ pg)); O = abs(float(np.sum(psi0 ** 2 * sg * strue)))
Enet = float(np.real(np.vdot(psi, H @ psi)) / np.sum(abs(psi) ** 2))
print(json.dumps(dict(E0=E0, eps_net=(Enet - E0) / abs(E0), eps_guide=(Eg - E0) / abs(E0), w_s=(1 - O) / 2,
                      fidelity_guide=float((pg @ psi0) ** 2), imag_frac=imf)))
np.savez(a_.out, a=amp, s=sg, psi=psi)
