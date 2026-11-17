"""consol_tests.py: CPU tests of consol_lib.py (no sector, no data).  usage: python consol_tests.py [a|e2|all]
 a   finite-difference check of the Jacobian rows of the relative residual r = exp(e) - 1 (value + bond rows + offset column), and of the two
     branches of solve_dir_rel against a dense Levenberg-Marquardt step built from the finite-difference Jacobian;
 b   the same with the (1 + r) factors dropped (drop1=True): must FAIL;
 e2  ViT head: with the unmodified head and every zeroed last-layer affine map the Jacobian of r is identically 0 (stationary point of log cosh);
     with the output gate w_out = 0 (consol_lib.make_vit_w) r = 0 and the Jacobian is non-zero exactly in the 60 columns of w_out.
Prints one JSON line per test."""
import sys, json
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import consol_lib as CL
f64 = jnp.float64


def synth(M=7, Kb=2, P=10, seed=0, scale=0.6):
    rng = np.random.default_rng(seed); R = M * (1 + Kb)
    X = jnp.asarray(rng.normal(size=(R, P)) / np.sqrt(P))
    th0 = jnp.asarray(rng.normal(size=P))
    model = lambda th: jnp.sin(X @ th) + 0.3 * (X @ th) ** 2               # S(theta) at the R = M(1+K) configurations [x ; y.reshape(-1)]
    G = jax.jacfwd(model)(th0)
    lx = jnp.asarray(scale * rng.normal(size=M)); ly = jnp.asarray(scale * rng.normal(size=(M, Kb)))
    wb = jnp.asarray(0.3 + rng.random(M)); c = 0.17
    return model, th0, G, lx, ly, wb, c


def residual(model, th, c, lx, ly, wb):
    return CL.rel_terms(model(th), lx, ly, wb, c)[0]


def fd_jac(model, th0, c, lx, ly, wb, h=1e-6):
    P = th0.size; cols = []
    for j in range(P):
        e = jnp.zeros(P).at[j].set(h)
        cols.append((residual(model, th0 + e, c, lx, ly, wb) - residual(model, th0 - e, c, lx, ly, wb)) / (2 * h))
    cols.append((residual(model, th0, c + h, lx, ly, wb) - residual(model, th0, c - h, lx, ly, wb)) / (2 * h))
    return jnp.stack(cols, 1)                                              # (R, P + 1)


def analytic_jac(G, S, lx, ly, wb, c, drop1=False):
    r, c1, c2, i1, i2, cc, *_ = CL.rel_terms(S, lx, ly, wb, c, drop1)
    return jnp.concatenate([c1[:, None] * G[i1] + c2[:, None] * G[i2], cc[:, None]], 1)


def test_ab(drop1):
    out = {}
    for tag, (M, Kb, P) in dict(sample_space=(7, 2, 30), param_space=(7, 2, 8)).items():
        model, th0, G, lx, ly, wb, c = synth(M, Kb, P)
        S = model(th0); Jfd = fd_jac(model, th0, c, lx, ly, wb)
        Ja = analytic_jac(G, S, lx, ly, wb, c, drop1)
        rel = float(jnp.linalg.norm(Ja - Jfd) / jnp.linalg.norm(Jfd))
        # solve_dir_rel against the dense LM step of the finite-difference Jacobian (same damping rule per branch)
        r = residual(model, th0, c, lx, ly, wb); R = M * (1 + Kb); lam_rel = 1e-2
        if R <= P + 1: lam = lam_rel * jnp.trace(Jfd @ Jfd.T) / R
        else: lam = lam_rel * jnp.trace(Jfd.T @ Jfd) / (P + 1)
        d_ref = -jnp.linalg.solve(Jfd.T @ Jfd + lam * jnp.eye(P + 1), Jfd.T @ r)
        dP, dc, *_ = CL.solve_dir_rel(G, S, lx, ly, wb, c, lam_rel, 1e9, drop1=drop1)
        d = jnp.concatenate([dP, jnp.atleast_1d(dc)])
        out[tag] = dict(R=R, P=P, jac_rel_err=rel, step_rel_err=float(jnp.linalg.norm(d - d_ref) / jnp.linalg.norm(d_ref)), r_rms=float(jnp.sqrt(jnp.mean(r ** 2))))
    return out


def test_e2():
    sys.path.insert(0, '/project/theorie/a/A.Otaifi/Chatty/runs/compress6/code')
    import flax
    import vit_dt
    from jax.flatten_util import ravel_pytree
    vit_dt.set_dtype('float32')
    N = 36; rng = np.random.default_rng(1); X = jnp.asarray(rng.choice([-1.0, 1.0], size=(5, N)).astype(np.float32))
    res = {}
    for tag in ('gate_w0', 'literal_zero_norms'):
        vit = CL.make_vit_w(vit_dt)
        tmpl = vit.init(jax.random.PRNGKey(3), jnp.zeros((1, N), jnp.float32)); p = flax.core.unfreeze(tmpl)['params']
        if tag == 'literal_zero_norms':
            for k in ('norm2', 'norm3'):
                p['output'][k]['scale'] = jnp.zeros_like(p['output'][k]['scale']); p['output'][k]['bias'] = jnp.zeros_like(p['output'][k]['bias'])
            p['output']['w_out'] = jnp.ones_like(p['output']['w_out'])
        th, unr = ravel_pytree(p)
        def r1(fl, x):
            zs = vit_dt.logpsi_transl_2d(vit.apply, 2, {'params': unr(fl)}, x[None])[0]
            return jnp.real(zs)
        G = jax.vmap(jax.grad(r1), in_axes=(None, 0))(th, X)
        nz = np.nonzero(np.max(np.abs(np.asarray(G)), 0) > 0)[0]
        # which leaf do the non-zero columns belong to
        leaves, names = [], []
        for path, v in jax.tree_util.tree_flatten_with_path(p)[0]:
            names += ['/'.join(str(k.key) for k in path)] * v.size
        res[tag] = dict(npar=int(th.size), r_max_abs=float(jnp.max(jnp.abs(jax.vmap(r1, in_axes=(None, 0))(th, X)))), n_nonzero_columns=int(nz.size),
                        nonzero_leaves=sorted(set(names[i] for i in nz)), G_max_abs=float(jnp.max(jnp.abs(G))))
    return res


if __name__ == '__main__':
    w = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if w in ('a', 'all'): print(json.dumps(dict(test='a', **test_ab(False))))
    if w in ('b', 'all'): print(json.dumps(dict(test='b_drop1', **test_ab(True))))
    if w in ('e2', 'all'): print(json.dumps(dict(test='e2', **test_e2())))
