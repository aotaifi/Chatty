"""T-BAD10 step 0 helpers: f32h symm ViT evaluator (chunked, padded), Marshall sign at 10x10, lattice bonds, one-swap neighbours.
Sigma convention: +-1 arrays (B,100); the ViT (site = 10 x + y) is fed the array as is (published driver convention; the bond set is symmetric under the label transpose)."""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
L = 10; N = 100
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import vit10_jax as V
import validate_vit10 as VV
BI = VV.BI[:, 0].astype(np.int64); BJ = VV.BI[:, 1].astype(np.int64); JB = np.asarray(VV.BJ, float); NB = len(BI)    # 400 bonds, 200 NN (J1=1) then 200 NNN (J2=.5)
HW = JB / 2
MARSH_A = np.array([(i % L + i // L) % 2 == 0 for i in range(N)])

def marshall(S):
    """+1 if the number of up spins on sublattice (x+y) even is even, else -1.  S (B,100) +-1."""
    return np.where(((np.asarray(S)[:, MARSH_A] > 0).sum(1)) % 2 == 0, 1, -1).astype(np.int8)

def make_vit(prec='f32h'):
    """returns ev(S (K,100) any float) -> complex128 logpsi (K,), padded chunks of `chunk`; f32h = f32 matmul precision highest."""
    if prec == 'f64':
        f = jax.jit(V.make_symm(V.load(V.SYMM, jnp.float64), dt=jnp.float64)); dt = np.float64; jd = jnp.float64
    else:
        f = jax.jit(V.make_symm(V.load(V.SYMM, jnp.float32), dt=jnp.float32, prec='highest')); dt = np.float32; jd = jnp.float32
    def ev(S, chunk=1024):
        S = np.asarray(S); K = len(S); out = np.empty(K, np.complex128); i = 0
        while i < K:
            n = min(chunk, K - i); blk = S[i:i + n]
            if n < chunk: blk = np.concatenate([blk, np.repeat(blk[:1], chunk - n, 0)])
            out[i:i + n] = np.asarray(f(jnp.asarray(blk, jd)))[:n]; i += n
        return out
    return ev

def neighbours(S):
    """all distinct one-swap neighbours (antiparallel J1+J2 bonds) of the rows of S: returns wi (parent index), bi (bond), Y (n,100)."""
    S = np.asarray(S); anti = S[:, BI] != S[:, BJ]; wi, bi = np.nonzero(anti)
    Y = S[wi].copy(); r = np.arange(len(wi)); Y[r, BI[bi]] = S[wi, BJ[bi]]; Y[r, BJ[bi]] = S[wi, BI[bi]]
    return wi, bi, Y

def keys_of(S):
    """hashable row keys (packed bits as void) for np.unique."""
    P = np.packbits(np.asarray(S) > 0, axis=1); return np.ascontiguousarray(P).view(np.dtype((np.void, P.shape[1]))).ravel()

def diag0(S):
    return 0.25 * np.sum(JB[None] * S[:, BI] * S[:, BJ], 1)

def shipped():
    return VV.spins().astype(np.float32)
