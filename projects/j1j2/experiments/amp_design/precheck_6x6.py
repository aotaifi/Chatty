"""Design pre-check (6x6, J2/J1 = 0.5, iteration-2 frozen FN problem of it2_amp.py, guide G1 = (a1, K1net + hop(a1, T1))).

For q_beta ~ a1^(2 beta), beta in --betas (bond-exchange Metropolis of it2_core.sample, full fp32 ViT):
  * acceptance, 2 tau_int (chain-mean variance vs per-sample variance), ESS/N of w_a = a^(2-2beta);
  * E_F[c] = sum w_c E_loc^F[c] / sum w_c, w_c = c^2/q_beta, for c = a1 and c = b (candidate: a2small params);
    paired D_hat = E_F[b] - E_F[a] on the same samples; jackknife over 16 chain groups (and over single chains);
  * FN wall term dfn - diag on samples (median / 99% / max), neighbour counts (total, sign-violating);
  * timing of ViT forward on all neighbours and of per-example gradients (vmap(grad)) on x, x+violating, x+all.
Usage: python precheck_6x6.py --spec spec_G1_hop.json --b params_a2small.npy --out precheck_6x6.json
"""
import argparse, json, time
import numpy as np
import jax, jax.numpy as jnp
import ll6_core as C
import it2_core as I

ap = argparse.ArgumentParser()
ap.add_argument('--spec', required=True)
ap.add_argument('--b', required=True)
ap.add_argument('--betas', default='1,0.5,0.25')
ap.add_argument('--nch', type=int, default=1024)
ap.add_argument('--nrounds', type=int, default=16)
ap.add_argument('--between', type=int, default=4)
ap.add_argument('--burn', type=int, default=100)
ap.add_argument('--chunk-local', type=int, default=1024)
ap.add_argument('--ngroups', type=int, default=16)
ap.add_argument('--njac', type=int, default=1024)
ap.add_argument('--jac-chunk', type=int, default=256)
ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--out', default='precheck_6x6.json')
args = ap.parse_args()
clk = I.GpuClock(); res = dict(args=vars(args), backend=jax.default_backend(),
                               device=str(jax.devices()[0].device_kind))
def save(): res['gpu_h'] = clk.hours(); json.dump(res, open(args.out, 'w'), indent=1)

net = C.Net(I.CKPT, dtype='float32', batch=16384, jac_chunk=args.jac_chunk)
spec, bd = I.load_spec(args.spec)
G = I.HGuide(spec, net, base_dir=bd)
thA = I.load_flat(net, I.resolve(spec['amp_g'], bd))
thB = jnp.asarray(np.load(args.b), net.DT)
res['npar'] = net.npar; res['spec'] = spec; res['b'] = args.b
I.log('backend', jax.default_backend(), res['device'], 'npar', net.npar)


def jk_groups(vals_fn, grp, G_):
    """jackknife over groups: vals_fn(mask) -> estimate."""
    full = vals_fn(np.ones_like(grp, bool))
    reps = np.array([vals_fn(grp != g) for g in range(G_)])
    return float(full), float(np.sqrt((G_ - 1) / G_ * np.sum((reps - reps.mean()) ** 2)))


def ratio(w, e, m):
    return np.sum(w[m] * e[m]) / np.sum(w[m])


def chain_jk(w, e, chain, nch):
    """fast leave-one-chain-out jackknife of sum(w e)/sum(w) (vector of per-chain sums)."""
    sw = np.bincount(chain, weights=w, minlength=nch); swe = np.bincount(chain, weights=w * e, minlength=nch)
    reps = (swe.sum() - swe) / (sw.sum() - sw)
    return float(np.sqrt((nch - 1) / nch * np.sum((reps - reps.mean()) ** 2)))


def chain_jk_diff(wb, eb, wa, ea, chain, nch):
    sb = np.bincount(chain, weights=wb, minlength=nch); sbe = np.bincount(chain, weights=wb * eb, minlength=nch)
    sa = np.bincount(chain, weights=wa, minlength=nch); sae = np.bincount(chain, weights=wa * ea, minlength=nch)
    reps = (sbe.sum() - sbe) / (sb.sum() - sb) - (sae.sum() - sae) / (sa.sum() - sa)
    return float(np.sqrt((nch - 1) / nch * np.sum((reps - reps.mean()) ** 2)))


def naive_se(w, e):
    """iid delta-method SE of the self-normalised mean."""
    wn = w / w.sum(); m = np.sum(wn * e)
    return float(np.sqrt(np.sum(wn ** 2 * (e - m) ** 2)))


def two_tau(v, nr, nch):
    """2 tau_int in rounds from chain-mean variance: var(chain mean) = var(v) 2tau / nr."""
    v = v.reshape(nr, nch); return float(nr * v.mean(0).var(ddof=1) / v.var())


def lag1(v, nr, nch):
    v = v.reshape(nr, nch); v = v - v.mean()
    return float(np.mean(v[1:] * v[:-1]) / np.mean(v * v))


q = lambda a: [float(np.median(a)), float(np.quantile(a, 0.99)), float(np.max(a))]
rows = {}; keep = None
for beta in [float(b) for b in args.betas.split(',')]:
    t0 = time.time()
    Xs, acc = I.sample(net, thA, beta, args.nch, args.nrounds, args.between, args.burn, args.seed + int(1000 * beta))
    X = Xs.reshape(-1); n = len(X); chain = np.tile(np.arange(args.nch), args.nrounds); grp = chain % args.ngroups
    jax.block_until_ready(thA); t1 = time.time()
    d = G.local_chunked(X, args.chunk_local); t2 = time.time()
    la = d['lax']
    lbx = net.logabs(thB, X); lbc = net.logabs(thB, d['child'])
    ELb, EHb = C.trial_elocs(d, lbx - la, lbc - d['lac'])
    ELa = d['eh']
    ELa0, _ = C.trial_elocs(d, np.zeros(n), np.zeros(len(d['child'])))
    assert np.allclose(ELa0, ELa)
    wall = d['dfn'] - d['diag']
    lwa = (2 - 2 * beta) * la; lwb = 2 * lbx - 2 * beta * la
    sh = max(lwa.max(), lwb.max()); wa = np.exp(lwa - sh); wb = np.exp(lwb - sh)
    ntot = np.bincount(d['own'], minlength=n); nvio = np.bincount(d['own'][~d['allowed']], minlength=n)
    Ea, Ea_se = jk_groups(lambda m: ratio(wa, ELa, m), grp, args.ngroups)
    Eb, Eb_se = jk_groups(lambda m: ratio(wb, ELb, m), grp, args.ngroups)
    D, D_se = jk_groups(lambda m: ratio(wb, ELb, m) - ratio(wa, ELa, m), grp, args.ngroups)
    r = dict(beta=beta, n=n, acceptance=acc, t_sample_s=t1 - t0, t_local_s=t2 - t1, t_b_s=time.time() - t2,
             ess_a=float(wa.sum() ** 2 / (n * np.sum(wa ** 2))), ess_b=float(wb.sum() ** 2 / (n * np.sum(wb ** 2))),
             EFa_site=[Ea / 36, Ea_se / 36], EFb_site=[Eb / 36, Eb_se / 36], D_site=[D / 36, D_se / 36],
             EFa_se_chainjk_site=chain_jk(wa, ELa, chain, args.nch) / 36,
             D_se_chainjk_site=chain_jk_diff(wb, ELb, wa, ELa, chain, args.nch) / 36,
             EFa_se_naive_site=naive_se(wa, ELa) / 36,
             twotau_rounds_ELa=two_tau(ELa, args.nrounds, args.nch), twotau_rounds_loga=two_tau(la, args.nrounds, args.nch),
             lag1_ELa=lag1(ELa, args.nrounds, args.nch), lag1_loga=lag1(la, args.nrounds, args.nch),
             wall_med_99_max=q(wall), wall_mean=float(wall.mean()), wall_frac_nonzero=float(np.mean(wall > 0)),
             wall_weighted_mean_a=float(np.sum(wa * wall) / wa.sum()),
             nbr_total_mean=float(ntot.mean()), nbr_viol_mean=float(nvio.mean()), nbr_viol_frac_samples=float(np.mean(nvio > 0)),
             nbr_viol_q=q(nvio), dlogb_rms=float(np.std(lbx - la)), loga_mean_sd=[float(la.mean()), float(la.std())],
             ELa_sd_site=float(np.std(ELa) / 36), EHb_mean_site=float(np.sum(wb * EHb) / wb.sum() / 36))
    # naive iid SE of the paired difference (delta method): influence z = wbn (ELb - Eb) - wan (ELa - Ea)
    z = wb / wb.sum() * (ELb - Eb) - wa / wa.sum() * (ELa - Ea)
    r['D_se_naive_site'] = float(np.sqrt(np.sum(z ** 2)) / 36)
    rows[beta] = r; res['rows'] = list(rows.values()); clk.tick(f'beta{beta}'); save()
    I.log('ROW', json.dumps(r))
    if beta == 1.0: keep = (X, d)

b1 = rows.get(1.0)
for r in rows.values():
    r['D_se_ratio_vs_beta1'] = r['D_site'][1] / b1['D_site'][1] if b1 else None
    r['D_se_chainjk_ratio_vs_beta1'] = r['D_se_chainjk_site'] / b1['D_se_chainjk_site'] if b1 else None
    r['EFa_se_ratio_vs_beta1'] = r['EFa_site'][1] / b1['EFa_site'][1] if b1 else None
res['rows'] = list(rows.values()); save()

# ---------------- Jacobian / forward timing on beta=1 samples
X, d = keep
m = args.njac; Xc = X[:m]
sel = d['own'] < m
ch_all = d['child'][sel]; ch_vio = d['child'][sel & ~d['allowed']]
grad = net._grad_bits; c = args.jac_chunk


def fwd(S):
    S = np.asarray(S, np.uint64); B = net.batch; outs = []
    for i in range(0, len(S), B):
        s = S[i:i + B]
        if len(s) < B: s = np.concatenate([s, np.repeat(s[:1], B - len(s))])
        outs.append(net._f_bits(thA, jnp.asarray(s)))
    jax.block_until_ready(outs)


def grads(S):
    S = np.asarray(S, np.uint64); npad = (-len(S)) % c
    if npad: S = np.concatenate([S, np.repeat(S[:1], npad)])
    acc = jnp.zeros((), net.DT)
    for i in range(0, len(S), c):
        g = grad(thA, jnp.asarray(S[i:i + c])); acc = acc + g[0, 0]   # keep per-example jac alive until done
    jax.block_until_ready(acc)


def timeit(f, S, reps=3):
    f(S[:c] if len(S) >= c else S)  # warm-up / compile
    ts = []
    for _ in range(reps):
        t = time.time(); f(S); ts.append(time.time() - t)
    return float(np.min(ts))


fwd(Xc)
tim = dict(n_central=m, n_children_all=int(len(ch_all)), n_children_viol=int(len(ch_vio)), jac_chunk=c, fwd_batch=net.batch)
tim['a_fwd_all_nbrs_ms_per_x'] = 1e3 * timeit(fwd, ch_all) / m
tim['b_grad_x_ms_per_x'] = 1e3 * timeit(grads, Xc) / m
tim['c_grad_x_plus_viol_ms_per_x'] = 1e3 * timeit(grads, np.r_[Xc, ch_vio]) / m
tim['d_grad_x_plus_all_ms_per_x'] = 1e3 * timeit(grads, np.r_[Xc, ch_all]) / m
tim['grad_ms_per_state'] = 1e3 * timeit(grads, np.r_[Xc, ch_all]) / (m + len(ch_all))
tim['fwd_ms_per_state'] = 1e3 * timeit(fwd, np.r_[Xc, ch_all]) / (m + len(ch_all))
res['timing'] = tim; clk.tick('timing'); save()
I.log('TIMING', json.dumps(tim)); I.log('DONE', json.dumps(res['gpu_h']))
