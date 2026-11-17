"""consol_lib.py: pieces of the 6x6 consolidation test (STORAGE_PROBLEM.md 'CONSOLIDATION TEST 6x6 SPEC' and 'HOLD REVERSED').
Imported by consol_fit.py (= cx_c1fit.py + the options below) and by consol_tests.py (CPU tests, no sector needed).

1. per_sample_rel / solve_dir_rel: arm P, the frozen-relative loss.  cx_c1fit's residual e = S - lx - c (S = model log-amplitude, lx = target
   log-amplitude, c a fitted offset) is replaced by r = exp(e) - 1 in BOTH terms of the per-configuration loss
       L_x = r_x^2 + wb sum_k (r_x - r_yk)^2        (cx_c1fit: e_x^2 + wb sum_k (e_x - e_yk)^2)
   solve_dir_rel is cx_c1fit.solve_dir with the Gauss-Newton rows of r:  d r / d theta = (1 + r) G  (G = d log A / d theta),  d r / d c = -(1 + r);
   value rows  sv r_x      -> Jacobian sv (1 + r_x) G_x,  offset entry -sv (1 + r_x);
   bond rows   sb (r_x - r_y) -> sb [(1 + r_x) G_x - (1 + r_y) G_y],  offset entry -sb [(1 + r_x) - (1 + r_y)] = -sb (r_x - r_y)  (NOT zero: it is zero
   for the log loss only).  Same damping rule (lam_rel tr / R resp. / (P+1)), same trust radius (predicted rms change of log A), same two branches
   (sample space = minSR when R <= P + 1, parameter space otherwise).  drop1=True is a TEST HOOK that omits the (1 + r) factors (must fail the
   finite-difference test).
2. make_vit_w: the 'vit' architecture of cx_c1fit (vit_dt.ViT, make_model_6x6: 4 layers, d_model 60, 154980 parameters) with ONE extra parameter
   vector w_out (60) multiplying the channel sum of the output head, initialised to zero: log psi_g = sum_k w_out_k log cosh(amp_k + i sign_k).
   Reason (shown in consol_tests.py test e2): in the unmodified head log psi = sum_k log cosh(amp_k + i sign_k), and d log cosh(a + i s)/d(a, s) = tanh = 0
   at a = s = 0, so ANY zero-initialisation of the last layer that makes r = 0 (kernels, biases or norm scales set to 0) makes the whole Jacobian
   of r vanish at step 0 (a stationary point; Gauss-Newton cannot move).  With w_out = 0 the Jacobian is non-zero in the 60 columns of w_out and the
   body follows as w_out grows (ReZero-style gate).
"""
import jax
import jax.numpy as jnp
from functools import partial

f64 = jnp.float64


def per_sample_rel(S, lx, ly, wb, c):
    """cx_c1fit.per_sample with e -> r = exp(e) - 1 in the value and the bond terms; returns (L, ex, db) with ex, db in log units
    (the held-out diagnostics of cx_c1fit are in log units for every arm)."""
    M = lx.shape[0]
    ex = S[:M] - lx - c; ey = S[M:].reshape(M, -1) - ly - c
    rx = jnp.expm1(ex); ry = jnp.expm1(ey)
    dr = rx[:, None] - ry
    db = ex[:, None] - ey
    return rx * rx + wb * jnp.sum(dr * dr, 1), ex, db


def rel_terms(S, lx, ly, wb, c, drop1=False):
    """residual vector and the row structure of its Jacobian (all float64).
    res (R,)   = [sv r_x ; sb (r_x - r_yk)]  (R = M (1 + K))
    c1, c2 (R,), i1, i2 (R,) : row i of J = c1[i] G[i1[i]] + c2[i] G[i2[i]]       (G rows: [x ; y_(x,k)])
    cc (R,)    : d res / d c."""
    M = lx.shape[0]
    ex = S[:M] - lx - c; ey = S[M:].reshape(M, -1) - ly - c; Kb = ey.shape[1]
    rx = jnp.expm1(ex); ry = jnp.expm1(ey)
    ax = jnp.ones_like(rx) if drop1 else 1.0 + rx
    ay = jnp.ones_like(ry) if drop1 else 1.0 + ry
    sv = 1.0 / jnp.sqrt(M); sb = jnp.sqrt(wb / M)
    res = jnp.concatenate([sv * rx, (sb[:, None] * (rx[:, None] - ry)).reshape(-1)])
    rows = jnp.arange(M * Kb); xs = rows // Kb
    i1 = jnp.concatenate([jnp.arange(M), xs]); c1 = jnp.concatenate([sv * ax, (sb * ax)[xs]])
    i2 = jnp.concatenate([jnp.zeros(M, jnp.int32), M + rows]); c2 = jnp.concatenate([jnp.zeros(M, f64), -(sb[xs] * ay.reshape(-1))])
    cc = jnp.concatenate([-sv * ax, (-sb[:, None] * (ax[:, None] - ay)).reshape(-1)])
    return res, c1, c2, i1, i2, cc, ex, ey, sv, sb, ax, ay


@partial(jax.jit, static_argnames=('drop1',))
def solve_dir_rel(G, S, lx, ly, wb, c, lam_rel, tau, drop1=False):
    """cx_c1fit.solve_dir for the relative loss (see the module docstring).  Returns (dP, dc, prms, scale, sum res^2)."""
    M = lx.shape[0]; P = G.shape[1]; dt = G.dtype
    r, c1, c2, i1, i2, cc, ex, ey, sv, sb, ax, ay = rel_terms(S, lx, ly, wb, c, drop1)
    Kb = ey.shape[1]; R = M * (1 + Kb)
    Gx = G[:M]
    if R <= P + 1:                                        # minSR, sample space: J never materialised
        GG = (G @ G.T).astype(f64)
        H = c1[:, None] * GG[i1] + c2[:, None] * GG[i2]; del GG
        T = H[:, i1] * c1[None, :] + H[:, i2] * c2[None, :] + jnp.outer(cc, cc); del H
        lam = lam_rel * jnp.trace(T) / R
        T = T.at[jnp.arange(R), jnp.arange(R)].add(lam)
        a = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(T, lower=True), r); del T
        w = jnp.zeros(R, f64).at[i1].add(c1 * a).at[i2].add(c2 * a)        # A^T a
        dP = -(G.T @ w.astype(dt)).astype(f64); dc = -(cc @ a)
    else:                                                 # the same LM step in parameter space
        Jv = Gx * (sv * ax).astype(dt)[:, None]
        Jb = ((Gx[:, None, :] * (sb[:, None] * ax[:, None]).astype(dt)[:, :, None] if False else
               Gx[:, None, :] * (sb * ax).astype(dt)[:, None, None]) -
              G[M:].reshape(M, Kb, P) * (sb[:, None] * ay).astype(dt)[:, :, None]).reshape(M * Kb, P)
        Jf = jnp.concatenate([jnp.concatenate([Jv, Jb]).astype(f64), cc[:, None]], 1); del Jv, Jb
        Sm = Jf.T @ Jf; g = Jf.T @ r
        lam = lam_rel * jnp.trace(Sm) / (P + 1)
        d = -jnp.linalg.solve(Sm + lam * jnp.eye(P + 1, dtype=f64), g)
        dP = d[:P]; dc = d[P]
    pc = (Gx.astype(f64) @ dP); pc = pc - jnp.mean(pc)
    prms = jnp.sqrt(jnp.mean(pc * pc))
    sc = jnp.minimum(1.0, tau / jnp.maximum(prms, 1e-30))
    return dP * sc, dc * sc, prms, sc, jnp.sum(r * r)


# ============================================================================================ arm H: the ViT with a zero-initialised output gate
def make_vit_w(vit_dt):
    """ViT of vit_dt (same module tree and parameter names) with params['output']['w_out'] (d_model,), init 0, multiplying the channel sum of the
    output head.  Call after vit_dt.set_dtype('float32')."""
    import flax.linen as nn

    class OuputHeadW(vit_dt.OuputHead):
        def setup(self):
            super().setup()
            self.w_out = self.param('w_out', nn.initializers.zeros, (self.d_model,), vit_dt.DT)

        def __call__(self, x):
            x = self.out_layer_norm(x.astype(vit_dt.HDT).sum(axis=1))
            amp = self.norm2(self.output_layer0(x))
            sign = self.norm3(self.output_layer1(x))
            return jnp.sum(self.w_out.astype(vit_dt.HDT) * vit_dt.log_cosh(amp + 1j * sign), axis=-1)

    class ViTW(vit_dt.ViT):
        def setup(self):
            self.patches_and_embed = vit_dt.Embed(self.d_model, self.b, two_dimensional=self.two_dimensional)
            self.encoder = vit_dt.Encoder(num_layers=self.num_layers, d_model=self.d_model, h=self.heads, L_eff=self.L_eff,
                                          transl_invariant=self.transl_invariant, two_dimensional=self.two_dimensional)
            self.output = OuputHeadW(self.d_model)

    return ViTW(num_layers=4, d_model=60, heads=10, L_eff=9, b=2, transl_invariant=True, two_dimensional=True)
