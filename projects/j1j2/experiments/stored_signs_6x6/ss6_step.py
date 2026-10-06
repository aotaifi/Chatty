"""One stored Krylov step on 6x6 at fixed |psi_ViT|:  given stored chain s_hat_{k-1}, build s_hat_k.

  T_{k-1}: k=1 -> T_FN; else energy-optimal (label-free) on beta=1 'thr' samples, E(T) of a*s_hat_{k-1}*sgn(T-r)
           with r computed from the STORED s_hat_{k-1} (one hop; curve needs the 2-hop shell, build time only).
  labels c_k(x) = sgn[T_{k-1} - r_{k-1}(x)] on tempered train samples (+ all one-hop neighbours).
  net_k trained (BCE + aux asinh(T-r)), s_hat_k = s_hat_{k-1} * sgn(logit_k).
Evaluation:
  held-out tempered samples (beta 0.5, 0.3, 1): disagreement stored vs one-hop recursion (s_hat_{k-1} c_k, the
  training target) and vs the EXACT recursive s^(k) (thresholds Ts, k hops).
  ED (x ~ |psi0|^2): w_s for stored, one-hop rec, exact recursive (subset), paired differences.
  Energy: defect-attributed Delta E vs ViT on beta=1 'energy' samples for stored s_hat_{k-1}, s_hat_k, one-hop rec,
  and exact recursive s^(k) on a subset (paired).
Usage: python ss6_step.py --k K --prev PREV_DIR|none --beta B --ch C --depth D --tag TAG [...]
"""
import argparse, json, time, os, shutil
import numpy as np
import jax
import ss6_core as S
import ll6_core as C

ap = argparse.ArgumentParser()
ap.add_argument('--k', type=int, required=True)
ap.add_argument('--prev', default='none')
ap.add_argument('--beta', type=float, default=0.5)
ap.add_argument('--ch', type=int, default=32)
ap.add_argument('--depth', type=int, default=4)
ap.add_argument('--d4', action='store_true')
ap.add_argument('--T', default='opt')
ap.add_argument('--ntrain', type=int, default=-1)
ap.add_argument('--nbr', type=int, default=1)
ap.add_argument('--epochs', type=int, default=12)
ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--batch', type=int, default=1024)
ap.add_argument('--lam', type=float, default=0.1)
ap.add_argument('--warm', type=int, default=0)          # warm start net_k from net_{k-1}
ap.add_argument('--nthr', type=int, default=-1)
ap.add_argument('--ned', type=int, default=400000)
ap.add_argument('--ned-exact', type=int, default=100000)
ap.add_argument('--nheld-exact', type=int, default=4096)
ap.add_argument('--nen', type=int, default=-1)            # energy samples (rows of 2048 chains)
ap.add_argument('--nen-exact-rows', type=int, default=1)  # rows of the energy set used for exact recursive K>=2
ap.add_argument('--skip-exact-energy', action='store_true')
ap.add_argument('--tag', required=True)
args = ap.parse_args()
t0 = time.time(); K = args.k
res = dict(args=vars(args)); tim = {}
def tick(name):
    tim[name] = round(time.time() - t0, 1); C.log('TIME', name, tim[name], 'vit_new', amp.nnew); res['time'] = tim
def save():
    json.dump(res, open(f'step_{args.tag}.json', 'w'), indent=1)

net = C.Net(S.CKPT, dtype='float32', batch=16384)
amp = S.AmpCache(net)
# ---- previous chain
if args.prev != 'none':
    meta = json.load(open(os.path.join(args.prev, 'chain.json')))
    nets = [S.SignNet.load(os.path.join(args.prev, p)) for p in meta['nets']]
    for p in meta['nets']:
        if not os.path.exists(p): shutil.copy(os.path.join(args.prev, p), p)
    prev = S.StoredChain(nets, meta['Ts']); prev_files = list(meta['nets'])
else:
    prev = S.StoredChain(); prev_files = []
assert prev.K == K - 1, (prev.K, K)
sprev = prev.signs
res['prev'] = args.prev; res['Ts_prev'] = prev.Ts

def load(beta):
    return dict(np.load(f'samples_b{beta}.npz'))
B1 = load(1.0)

# ---- threshold
if K == 1:
    T = C.T_FN; res['T'] = T
elif args.T != 'opt':
    T = float(args.T); res['T'] = T
else:
    Xt = B1['thr'].reshape(-1)
    if args.nthr > 0: Xt = Xt[:args.nthr]
    parts = [S.threshold_edges_stored(Xt[i:i + 512], amp, sprev) for i in range(0, len(Xt), 512)]
    E = C.merge_edges(parts)
    base, cand, curve = C.energy_curve(E)
    j = int(np.argmin(curve)); T = float(cand[j])
    half = {}
    for h in (0, 1):
        m = np.zeros(E['n'], bool); m[h::2] = True
        b2, c2, cu2 = C.energy_curve(E, m); half[h] = float(c2[int(np.argmin(cu2))])
    # T-curve on a coarse grid for the record
    grid = np.unique(np.r_[np.linspace(-20, 5, 101), T])
    curve_g = np.interp(grid, cand, curve)
    res['T'] = T; res['T_half'] = half; res['E_noflip_site'] = float(base / 36); res['E_opt_site'] = float(curve[j] / 36)
    res['flip_frac_thr_at_T'] = float(np.mean(E['rx_node'] > T))
    np.savez_compressed(f'thrcurve_{args.tag}.npz', grid=grid, curve=curve_g / 36)
    C.log('THRESH', T, half, res['E_noflip_site'], res['E_opt_site'])
tick('threshold'); save()

def onehop_rec(Sx):
    r, s = S.r_onehop_chunked(np.asarray(Sx, np.uint64), amp, sprev)
    return s * np.where(r <= T, 1., -1.)

Ts_all = prev.Ts + [T]
def exact_rec(Sx, chunk=None):
    Sx = np.asarray(Sx, np.uint64)
    g = C.Guide([amp] * K, Ts_all, amp)
    if chunk is None: chunk = {1: 4096, 2: 256, 3: 16}.get(K, 8)
    return np.concatenate([g.signs(Sx[i:i + chunk]) for i in range(0, len(Sx), chunk)])

# ---- labels on tempered training samples (+ one-hop neighbours)
Bt = load(args.beta)
Xs = Bt['train'].reshape(-1)
if args.ntrain > 0: Xs = Xs[:args.ntrain]
U = S.build_train_states(Xs, args.nbr)
NB = V = None
pre = S.load_label_shards(args.beta, K, U) if (K == 1 and args.ntrain <= 0) else None
if pre is not None:
    r = pre; res['labels_from_cpu_shards'] = True
else:
    r, s_ = S.r_onehop_chunked(U, amp, sprev)
y = np.where(r <= T, 1, -1).astype(np.int8)
aux = np.arcsinh(T - r)
res['labels'] = dict(n_samples=int(len(Xs)), n_unique_samples=int(len(np.unique(Xs))), n_train=int(len(U)),
                     flip_frac=float(np.mean(y < 0)), flip_frac_on_samples=float(np.mean(y[np.argsort(U)[np.searchsorted(np.sort(U), Xs)]] < 0)))
C.log('LABELS', res['labels'])
tick('labels'); save()

# held-out tempered sets: x and their neighbours
held = {}
for b in sorted({args.beta, 0.5, 0.3, 1.0}):
    Xh = (Bt if b == args.beta else load(b))['held'].reshape(-1)
    held[b] = Xh
Xv = held[args.beta][:8192]; yv = np.where(S.r_onehop_chunked(Xv, amp, sprev)[0] <= T, 1, -1)

# ---- train
snet = S.SignNet(args.ch, args.depth, d4=args.d4, seed=K)
if args.warm and prev.K and prev.nets[-1].cfg == snet.cfg:
    snet.params = prev.nets[-1].params
res['npar'] = snet.npar
hist = S.train_net(snet, U, y, aux, Xv, yv, args.epochs, args.lr, args.batch, lam=args.lam, seed=K)
res['train_hist'] = hist
fname = f'net{K}_{args.tag}.pkl'; snet.save(fname)
chain = S.StoredChain(prev.nets + [snet], Ts_all)
json.dump(dict(nets=prev_files + [fname], Ts=Ts_all, tag=args.tag), open('chain.json', 'w'), indent=1)
# training-set fit
lt = snet.logit(U); res['train_err'] = float(np.mean(np.where(lt >= 0, 1, -1) != y))
res['train_err_on_flips'] = float(np.mean(lt[y < 0] >= 0)) if (y < 0).any() else None
del NB, V
tick('train'); save()

# ---- eval timing
Z = held[1.0][:65536]
t1 = time.time(); chain.signs(Z); res['sec_per_stored_sign_eval'] = (time.time() - t1) / len(Z)
t1 = time.time(); onehop_rec(Z[:8192]); res['sec_per_onehop_rec_eval'] = (time.time() - t1) / 8192

# ---- held-out tempered: stored vs one-hop recursion vs exact recursion
H = {}
for b, Xh in held.items():
    st = chain.signs(Xh); oh = onehop_rec(Xh)
    d = dict(n=int(len(Xh)), stored_vs_onehop=S.disagree(st, oh),
             flipfrac_onehop_vs_prev=S.disagree(oh, sprev(Xh)))
    # neighbours of the first 2048 held states
    NBh, Vh = C.neighbors(Xh[:2048]); Yh = np.unique(NBh[Vh])
    d['nbr_stored_vs_onehop'] = S.disagree(chain.signs(Yh), onehop_rec(Yh)); d['n_nbr'] = int(len(Yh))
    if K >= 2:
        ne = min(len(Xh), args.nheld_exact if K == 2 else max(256, args.nheld_exact // 16))
        t1 = time.time(); ex = exact_rec(Xh[:ne]); d['exact_sec'] = time.time() - t1
        d['n_exact'] = int(ne)
        d['stored_vs_exact'] = S.disagree(st[:ne], ex); d['onehop_vs_exact'] = S.disagree(oh[:ne], ex)
        d['prevstored_vs_exactprev'] = None
    H[f'b{b}'] = d; C.log('HELD', b, d)
res['held'] = H
tick('held'); save()

# ---- ED scoring
Xe, strue = S.load_ed(args.ned)
st = chain.signs(Xe); sp = sprev(Xe)
oh = np.concatenate([onehop_rec(Xe[i:i + 50000]) for i in range(0, len(Xe), 50000)])
ED = dict(n=int(len(Xe)))
for nm, s in (('stored', st), ('onehop_rec', oh), ('prev_stored', sp)):
    ED[nm] = S.wrong_frac(s, strue)
ED['paired_stored_minus_onehop'] = S.paired_wrong(st, oh, strue, ED['stored']['g'], ED['onehop_rec']['g'])
ED['paired_stored_minus_prev'] = S.paired_wrong(st, sp, strue, ED['stored']['g'], ED['prev_stored']['g'])
ED['stored_vs_onehop_disagree'] = S.disagree(st, oh)
tick('ed_stored')
if K <= 2 or args.ned_exact > 0:
    ne = args.ned_exact if K <= 2 else min(args.ned_exact, 5000)
    if K == 1: ex = oh[:ne]
    else: ex = exact_rec(Xe[:ne])
    ED['n_exact'] = int(ne)
    ED['exact_rec'] = S.wrong_frac(ex, strue[:ne])
    ED['stored_on_exact_subset'] = S.wrong_frac(st[:ne], strue[:ne])
    ED['paired_stored_minus_exact'] = S.paired_wrong(st[:ne], ex, strue[:ne], ED['stored_on_exact_subset']['g'], ED['exact_rec']['g'])
    ED['stored_vs_exact_disagree'] = S.disagree(st[:ne], ex)
    np.savez_compressed(f'ed_signs_{args.tag}.npz', stored=st.astype(np.int8), onehop=oh.astype(np.int8),
                        exact=ex.astype(np.int8))
res['ED'] = ED; C.log('ED', json.dumps(ED))
tick('ed'); save()

# ---- energy (defect-attributed vs ViT)
Xen = B1['energy']; nch = Xen.shape[1]
if args.nen > 0: Xen = Xen[:max(1, args.nen // nch)]
fns = {'stored': chain.signs, 'prev_stored': sprev, 'onehop_rec': onehop_rec}
er = S.defect_energy(Xen, nch, amp, net, fns)
arr = er.pop('_dE_arrays'); res['phi'] = er.pop('_phi')
EN = dict(n=int(Xen.size), **er)
EN['paired_stored_minus_prev'] = S.paired_diff(arr['stored'], arr['prev_stored'], nch)
EN['paired_stored_minus_onehop'] = S.paired_diff(arr['stored'], arr['onehop_rec'], nch)
tick('energy_stored'); res['energy'] = EN; save()
if K == 2 and not args.skip_exact_energy:
    rows = args.nen_exact_rows if K == 2 else max(1, args.nen_exact_rows // 4)
    Xsub = Xen[:rows]
    if K >= 3: Xsub = Xsub[:, :512]
    ns = Xsub.shape[1]
    er2 = S.defect_energy(Xsub, ns, amp, net, {'exact_rec': exact_rec, 'stored': chain.signs,
                                                  'onehop_rec': onehop_rec}, chunk=256)
    a2 = er2.pop('_dE_arrays'); er2.pop('_phi')
    EN['exact_subset'] = dict(n=int(Xsub.size), **er2,
                              paired_stored_minus_exact=S.paired_diff(a2['stored'], a2['exact_rec'], ns),
                              paired_onehop_minus_exact=S.paired_diff(a2['onehop_rec'], a2['exact_rec'], ns))
res['energy'] = EN; C.log('ENERGY', json.dumps(EN))
res['vit_evals'] = int(net.neval); res['vit_new'] = int(amp.nnew)
tick('done'); save()
C.log('DONE')
