"""Dtype-configurable copy of the nqsmagic ViT (models/vit.py + attentions.py + utils/logpsi_transl.py).

Same module tree and parameter names as nqsmagic.models.ViT, so the 6x6 checkpoint
'vit_J2=0.50_N=6x6_k=0.mpack' loads unchanged.  The only change: every hard-coded jnp.float64 /
complex128 is replaced by the module-level dtypes set with set_dtype('float32'|'float64').
In float64 mode this is the original network (checked bit-for-bit in ll6_bench.py).
"""
from functools import partial
import math
import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange

DT = jnp.float64
CDT = jnp.complex128
HDT = jnp.float64          # dtype of the output head (LayerNorms on the token sum + log_cosh)
HCDT = jnp.complex128
LN_FAST = True             # flax default use_fast_variance (E[x^2]-E[x]^2); False = two-pass variance


def set_dtype(name, ln_fast=True, head64=False):
    global DT, CDT, HDT, HCDT, LN_FAST
    DT = jnp.float32 if name == 'float32' else jnp.float64
    CDT = jnp.complex64 if name == 'float32' else jnp.complex128
    HDT = jnp.float64 if head64 else DT
    HCDT = jnp.complex128 if head64 else CDT
    LN_FAST = ln_fast


def roll(J, shift, axis=-1):
    return jnp.roll(J, shift, axis=axis)


@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def roll2d(spins, i, j):
    side = int(spins.shape[-1] ** 0.5)
    spins = spins.reshape(spins.shape[0], side, side)
    spins = jnp.roll(jnp.roll(spins, i, axis=-2), j, axis=-1)
    return spins.reshape(spins.shape[0], -1)


class FMHA(nn.Module):
    d_model: int
    h: int
    L_eff: int
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        self.v = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=DT, dtype=DT)
        J = self.param("J", nn.initializers.xavier_uniform(), (self.h, self.L_eff), DT)
        sq = int(self.L_eff ** 0.5)
        J = roll2d(J, jnp.arange(sq), jnp.arange(sq))
        self.J = J.reshape(self.h, -1, self.L_eff)
        self.W = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=DT, dtype=DT)

    def __call__(self, x):
        v = self.v(x)
        v = rearrange(v, "batch L_eff (h d_eff) -> batch L_eff h d_eff", h=self.h)
        v = rearrange(v, "batch L_eff h d_eff -> batch h L_eff d_eff")
        x = jnp.matmul(self.J.astype(DT), v)
        x = rearrange(x, "batch h L_eff d_eff  -> batch L_eff h d_eff")
        x = rearrange(x, "batch L_eff h d_eff ->  batch L_eff (h d_eff)")
        return self.W(x)


def log_cosh(x):
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


def extract_patches2d(x, b):
    batch = x.shape[0]
    L_eff = int((x.shape[1] // b ** 2) ** 0.5)
    x = x.reshape(batch, L_eff, b, L_eff, b).transpose(0, 1, 3, 2, 4)
    return x.reshape(batch, L_eff, L_eff, -1).reshape(batch, L_eff * L_eff, -1)


class Embed(nn.Module):
    d_model: int
    b: int
    two_dimensional: bool = True

    def setup(self):
        self.embed = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=DT, dtype=DT)

    def __call__(self, x):
        return self.embed(extract_patches2d(x, self.b))


class EncoderBlock(nn.Module):
    d_model: int
    h: int
    L_eff: int
    transl_invariant: bool = True
    two_dimensional: bool = True

    def setup(self):
        self.attn = FMHA(d_model=self.d_model, h=self.h, L_eff=self.L_eff,
                         transl_invariant=self.transl_invariant, two_dimensional=self.two_dimensional)
        self.layer_norm_1 = nn.LayerNorm(dtype=DT, param_dtype=DT, use_fast_variance=LN_FAST)
        self.layer_norm_2 = nn.LayerNorm(dtype=DT, param_dtype=DT, use_fast_variance=LN_FAST)
        self.ff = nn.Sequential([
            nn.Dense(4 * self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=DT, dtype=DT),
            nn.gelu,
            nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=DT, dtype=DT)])

    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))
        return x + self.ff(self.layer_norm_2(x))


class Encoder(nn.Module):
    num_layers: int
    d_model: int
    h: int
    L_eff: int
    transl_invariant: bool = True
    two_dimensional: bool = True

    def setup(self):
        self.layers = [EncoderBlock(d_model=self.d_model, h=self.h, L_eff=self.L_eff,
                                    transl_invariant=self.transl_invariant, two_dimensional=self.two_dimensional)
                       for _ in range(self.num_layers)]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x


class OuputHead(nn.Module):
    d_model: int

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(dtype=HDT, param_dtype=DT, use_fast_variance=LN_FAST)
        self.norm2 = nn.LayerNorm(use_scale=True, use_bias=True, dtype=HDT, param_dtype=DT, use_fast_variance=LN_FAST)
        self.norm3 = nn.LayerNorm(use_scale=True, use_bias=True, dtype=HDT, param_dtype=DT, use_fast_variance=LN_FAST)
        self.output_layer0 = nn.Dense(self.d_model, param_dtype=DT, dtype=HDT,
                                      kernel_init=nn.initializers.xavier_uniform(), bias_init=jax.nn.initializers.zeros)
        self.output_layer1 = nn.Dense(self.d_model, param_dtype=DT, dtype=HDT,
                                      kernel_init=nn.initializers.xavier_uniform(), bias_init=jax.nn.initializers.zeros)

    def __call__(self, x):
        x = self.out_layer_norm(x.astype(HDT).sum(axis=1))
        amp = self.norm2(self.output_layer0(x))
        sign = self.norm3(self.output_layer1(x))
        return jnp.sum(log_cosh(amp + 1j * sign), axis=-1)


class ViT(nn.Module):
    num_layers: int
    d_model: int
    heads: int
    L_eff: int
    b: int
    transl_invariant: bool = True
    two_dimensional: bool = True

    def setup(self):
        self.patches_and_embed = Embed(self.d_model, self.b, two_dimensional=self.two_dimensional)
        self.encoder = Encoder(num_layers=self.num_layers, d_model=self.d_model, h=self.heads, L_eff=self.L_eff,
                               transl_invariant=self.transl_invariant, two_dimensional=self.two_dimensional)
        self.output = OuputHead(self.d_model)

    def __call__(self, x):
        return self.output(self.encoder(self.patches_and_embed(x)))


def logpsi_transl_2d(afun, b, variables, x):
    """Same as nqsmagic.utils._logpsi_transl_2d: logsumexp over the b*b patch-offset translations."""
    L = math.isqrt(x.shape[-1])
    shape = x.shape[:-1]
    x = x.reshape(shape + (L, L))
    outs = []
    for j in range(b):
        for i in range(b):
            xi = jnp.roll(jnp.roll(x, j, axis=-1), i, axis=-2).reshape(shape + (L * L,))
            outs.append(afun(variables, xi).astype(HCDT))
    out = jnp.stack(outs, 0)
    return jax.scipy.special.logsumexp(out, axis=0)


def make_model_6x6():
    return ViT(num_layers=4, d_model=60, heads=10, L_eff=9, b=2, transl_invariant=True, two_dimensional=True)
