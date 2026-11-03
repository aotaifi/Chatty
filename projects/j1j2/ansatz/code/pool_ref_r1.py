"""pool_ref_r1.py: reference numbers for pool_fit check (a), computed by cx_r1fit.py's OWN code (read-only: its source up to the "control 2" marker is
exec'd with a dry spec, so Lab, the pp2 model, Sampler, wbar_of, make_batch, bonds_of, per_sample and solve_dir are cx_r1fit's).  cx_r1fit.py is not modified.
  (a1) batches of the table fit (B = 72: minSR branch; B = 1800: parameter-space branch), theta = the round-5 pp2 warm start: mean loss, jax.grad of the mean loss
       through cx_r1fit's fwd, and solve_dir's (dP, dc).
  (a2) the table fit's POPULATION loss restricted to an orbit subset U (U_PER training orbits per target decade -5..-13):
       L = (1/9) sum_g sum_{x in U_g} p_x [e_x^2 + lam_b / Wbar_g sum_{bad y} w_xy (e_x - e_y)^2],  p_x = n_x / sum_{U_g} n,  Wbar_g = sum_{U_g} p_x W_x,
       w_xy = exp(lw) of bonds_of (|H_xy| phi_y / phi_x over all bad bonds of s_G), and its jax.grad.
usage: python pool_ref_r1.py OUT SPEC.json   (SPEC = a cx_r1fit dry spec, arm pp2, plus U_per, B_list)"""
import os, sys, json, time
import numpy as np
OUT, SPF = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
CODE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(CODE, 'cx_r1fit.py')
src = open(SRC).read(); cut = src.index('# ============================================================================================ control 2')
sys.argv = [SRC, os.path.join(OUT, 'r1head'), SPF]
G = dict(__name__='cx_r1fit_head', __file__=SRC); exec(compile(src[:cut], SRC, 'exec'), G)
jax, jnp = G['jax'], G['jnp']; SP = G['SPEC']
out = dict(cx_r1fit_sha256=G['hashlib'].sha256(open(SRC, 'rb').read()).hexdigest(), spec=SP)
th = G['theta_start']; out_np = dict(theta=np.asarray(th))


def grad_via_vjp(t, Xall, lossS):
    """jax.grad of lossS(fwd(t, Xall)) without holding every chunk's residuals: u = dL/dS by jax.grad, then sum over chunks of the VJP of cx_r1fit's fwd_j."""
    S = G['fwd'](t, Xall); u = jax.grad(lossS)(S); FCH = G['FCH']; g = jnp.zeros_like(t)
    for i in range(0, Xall.shape[0], FCH):
        s_, m = G['_pad'](Xall[i:i + FCH], FCH); uu = jnp.zeros(FCH).at[:m].set(u[i:i + m])
        _, vjp = jax.vjp(lambda tt: G['fwd_j'](tt, s_), t); g = g + vjp(uu)[0]
    return g
SEED = int(SP.get('seed', 1)); ho_h = G['ho_h']; Sampler = G['Sampler']
key = jax.random.PRNGKey(100 + SEED); key, k1 = jax.random.split(key)
WBAR = G['wbar_of'](Sampler(~ho_h, 3000 + SEED, True), k1, int(SP.get('n_wbar', 256))); out_np['wbar'] = WBAR
c = 0.3; out['c'] = c; lam_rel, tau = 1e-2, 0.05
for Bm in SP.get('B_list', [72, 1800]):
    t0 = time.time(); key, k1 = jax.random.split(key)
    bt = G['make_batch'](Sampler(~ho_h, 1000 + SEED + Bm, True), Bm, k1, WBAR)
    Lf = lambda t: jnp.mean(G['per_sample'](G['S_of'](t, bt), bt['lx'], bt['ly'], bt['wb'], c)[0])
    L = float(Lf(th)); Xall = jnp.concatenate([bt['x'], bt['y'].reshape(-1)])
    g = np.asarray(grad_via_vjp(th, Xall, lambda S_: jnp.mean(G['per_sample'](S_, bt['lx'], bt['ly'], bt['wb'], c)[0])))
    if Bm <= 100: out[f'a1_{Bm}_grad_full_jaxgrad_vs_vjp_rel'] = float(jnp.max(jnp.abs(jax.grad(Lf)(th) - g)) / jnp.max(jnp.abs(g)))
    S = G['fwd'](th, Xall); Gm = G['jac'](th, Xall)
    dP, dc, prms, sc, rr = G['solve_dir'](Gm, S, bt['lx'], bt['ly'], bt['wb'], c, lam_rel, tau)
    R = Bm * 3; branch = 'minSR' if R <= th.size + 1 else 'param'
    for k_, v_ in dict(idx=bt['idx'], g=bt['g'], x=bt['x'], y=bt['y'], lx=bt['lx'], ly=bt['ly'], wb=bt['wb'], grad=g, dP=dP, S=S).items(): out_np[f'a1_{Bm}_{k_}'] = np.asarray(v_)
    out[f'a1_{Bm}'] = dict(M=int(bt['x'].shape[0]), branch=branch, L=L, dc=float(dc), prms=float(prms), sc=float(sc), rr=float(rr), sec=time.time() - t0)
    print(f'a1 B={Bm} ({branch}) L {L:.12e} |grad| {np.linalg.norm(g):.6e} |dP| {float(jnp.linalg.norm(dP)):.6e} {time.time() - t0:.0f}s', flush=True)
# (a2) population loss on U
U_PER = int(SP.get('U_per', 27)); rng = np.random.default_rng(99); dec_h = G['dec_h']; n_h = G['n_h']; DECS = G['DECS']
_Ul = [np.sort(rng.choice(q, min(U_PER, q.size), replace=False)) for q in (np.nonzero((dec_h == d) & ~ho_h)[0] for d in DECS)]
U = np.concatenate(_Ul); gU = np.concatenate([np.full(u.size, k) for k, u in enumerate(_Ul)])
sec = G['sec']; x = sec.reps[jnp.asarray(U)]; v = G['valid_bonds'](x); y = jnp.where(v, x[:, None] ^ G['MASKS'][None, :], x[:, None])
iy = G['SS'].canon(sec.T, sec.reps, y.reshape(-1)).reshape(x.shape[0], G['NB'])
bad = v & (G['ST'][jnp.asarray(U)][:, None] * G['ST'][iy] > 0)
ls2 = G['ls2']; lw = jnp.where(bad, jnp.log(0.5 * G['JB'][None, :]) + (ls2[iy] - ls2[jnp.asarray(U)][:, None]), -jnp.inf)
w = np.asarray(jnp.exp(lw)); badn = np.asarray(bad); W = w.sum(1)
p = np.zeros(U.size); Wb = np.zeros(len(DECS))
for k in range(len(DECS)):
    m = gU == k; p[m] = n_h[U][m] / n_h[U][m].sum(); Wb[k] = np.sum(p[m] * W[m])
ii, bb = np.nonzero(badn); ycfg = np.asarray(y)[ii, bb]; wy = w[ii, bb]
LAMB = float(SP.get('lam_b', 1.0)); lx = ls2[jnp.asarray(U)]; ly = ls2[jnp.asarray(np.asarray(iy)[ii, bb])]


XA = jnp.concatenate([x, jnp.asarray(ycfg)]); NX = U.size


def LpopS(S_):
    ex = S_[:NX] - lx - c; ey = S_[NX:] - ly - c
    bond = jax.ops.segment_sum(jnp.asarray(wy) * (ex[jnp.asarray(ii)] - ey) ** 2, jnp.asarray(ii), num_segments=U.size)
    return jnp.sum(jnp.asarray(p) / len(DECS) * (ex * ex + LAMB / jnp.asarray(Wb)[jnp.asarray(gU)] * bond))
t0 = time.time(); L2 = float(LpopS(G['fwd'](th, XA))); g2 = np.asarray(grad_via_vjp(th, XA, LpopS))
out['a2'] = dict(U_per=U_PER, n_U=int(U.size), n_bad_pairs=int(ii.size), L=L2, Wbar=Wb.tolist(), sec=time.time() - t0)
out_np.update(a2_U=U, a2_g=gU, a2_grad=g2, a2_W=W, a2_nbad=badn.sum(1), a2_p=p)
print(f'a2 |U| {U.size} pairs {ii.size} L {L2:.12e} |grad| {np.linalg.norm(g2):.6e} {time.time() - t0:.0f}s', flush=True)
np.savez(os.path.join(OUT, 'ref_r1.npz'), **out_np); json.dump(out, open(os.path.join(OUT, 'ref_r1.json'), 'w'), indent=1, default=float)
print('DONE', flush=True)
