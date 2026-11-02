"""Own JAX re-implementation of the Rende/Viteritti 10x10 J1-J2 ViT NQS (HF nqs-models/j1j2_square_10x10).
Reference code was read as TEXT only; weights come from safetensors. sigma in +-1, shape (B,100), site = 10*x+y.
Defaults: flax LayerNorm eps=1e-6 and nn.gelu approximate=True (the reference calls nn.gelu / nn.LayerNorm unchanged)."""
import os, numpy as np, jax, jax.numpy as jnp
from safetensors import safe_open
jax.config.update("jax_enable_x64", True)
HF = os.environ.get("VIT10_HF", "/scratch/a/A.Otaifi/ext/hf_vit10/")
PLAIN, SYMM = HF + "model.safetensors", HF + "symm_trxy_ising/model.safetensors"
IDX = HF + "symm_trxy_ising/symm_indexes.txt"
NL, H, D, LE, B = 8, 12, 72, 5, 2           # layers, heads, d_model, patches per side, patch side

def load(path):
    f = safe_open(path, "np")
    return {k[7:]: jnp.asarray(f.get_tensor(k), jnp.float64) for k in f.keys()}   # strip 'params.'

def circulant(J, transposed=False):
    """roll2d: Jfull[h, i*5+j, :] = flat(roll(roll(J_h as 5x5, i, axis -2), j, axis -1)); attn out[l]=sum_m Jfull[l,m] v[m]."""
    g = J.reshape(H, LE, LE)
    Jf = jnp.stack([jnp.roll(jnp.roll(g, i, axis=-2), j, axis=-1).reshape(H, -1)
                    for i in range(LE) for j in range(LE)], axis=1)               # (H, 25 (i,j), 25)
    return jnp.swapaxes(Jf, 1, 2) if transposed else Jf    # transposed=True: deliberately broken variant

def ln(x, p, eps=1e-6):                                      # flax LayerNorm (variance E[x^2]-E[x]^2)
    m = x.mean(-1, keepdims=True); v = (x * x).mean(-1, keepdims=True) - m * m
    return (x - m) * jax.lax.rsqrt(v + eps) * p["scale"] + p["bias"]

def log_cosh(x):
    x = x * (1 - 2 * jnp.signbit(x.real))
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)

def make_plain(P, gelu_approx=True, transposed=False):
    Jf = [circulant(P[f"encoder.layers_{l}.attn.J"], transposed) for l in range(NL)]
    gelu = lambda x: jax.nn.gelu(x, approximate=gelu_approx)
    lin = lambda x, n: x @ P[n + ".kernel"] + P[n + ".bias"]
    nrm = lambda x, n: ln(x, {"scale": P[n + ".scale"], "bias": P[n + ".bias"]})
    def logpsi(sigma):
        s = jnp.atleast_2d(sigma).astype(jnp.float64); n = s.shape[0]
        x = s.reshape(n, LE, B, LE, B).transpose(0, 1, 3, 2, 4).reshape(n, LE * LE, B * B)   # extract_patches2d
        x = lin(x, "patches_and_embed.embed")
        for l in range(NL):
            e = f"encoder.layers_{l}."
            v = lin(nrm(x, e + "layer_norm_1"), e + "attn.v").reshape(n, LE * LE, H, D // H).transpose(0, 2, 1, 3)
            x = x + lin(jnp.matmul(Jf[l], v).transpose(0, 2, 1, 3).reshape(n, LE * LE, D), e + "attn.W")
            x = x + lin(gelu(lin(nrm(x, e + "layer_norm_2"), e + "ff.layers_0")), e + "ff.layers_2")
        z = nrm(x.sum(1), "output.out_layer_norm")
        amp = nrm(lin(z, "output.output_layer0"), "output.norm2")
        sgn = nrm(lin(z, "output.output_layer1"), "output.norm3")
        return jnp.sum(log_cosh(amp + 1j * sgn), axis=-1)       # complex log psi
    return logpsi

def make_symm(P, idx=None, **kw):
    f = make_plain(P, **kw)
    idx = jnp.asarray(np.loadtxt(IDX, dtype=int) if idx is None else idx)   # (32,100) site permutations
    def sym(t):                                                              # log sum_g psi(t[:, g])
        out = f(t[:, idx].reshape(-1, 100)).reshape(t.shape[0], -1)         # (n,32)
        z0 = out[:, :1]
        return z0[:, 0] + jnp.log(jnp.sum(jnp.exp(out - z0), axis=1))
    def logpsi(sigma):
        s = jnp.atleast_2d(sigma).astype(jnp.float64)
        fp, fm = sym(s), sym(-s)                                             # sz-flip (singlet) term
        return fp + jnp.log(1.0 + jnp.exp(fm - fp))
    return logpsi

_c = {}
def logpsi_plain(sigma):
    _c.setdefault("p", jax.jit(make_plain(load(PLAIN))))
    return _c["p"](sigma)
def logpsi_symm(sigma):
    _c.setdefault("s", jax.jit(make_symm(load(SYMM))))
    return _c["s"](sigma)
