"""Distil a 'stored + one exact hop' sign into a fully stored sign (one more stacked sign net).

Input spec G (it2_core): s_G = s_st * sgn(T - r[a_hop, s_st]),  s_st = Marshall * prod sgn(logit_j).
Output: net_new with s_new = s_st * sgn(logit_new) ~ s_G;  spec G' = (nets + [net_new], hop = null, amp_g).
Training set (ss6 recipe, 'self-sealing' fix): tempered samples x ~ a_g^(2 beta) + ALL one-hop neighbours,
labels c = sgn(T - r[a_hop, s_st]) (one hop on the stored sign), BCE + 0.1 aux asinh(T - r).
Validation:
  held-out tempered (beta, 0.5, 1) disagreement stored vs hop;
  ED (x ~ psi0^2): w_s of stored s_new, of the hop sign s_G and (optional) of the RECURSIVE ll6 sign of --rec-state;
  variational energy on x ~ a_g^2: paired <H>_{a_g s_new} - <H>_{a_g s_G}; and vs the recursive sign on a subset.
Usage: python it2_distill.py --spec G.json --out TAG [--rec-state state.json] [...]
"""
import argparse, json, time, os, shutil
import numpy as np
import jax
import ll6_core as C
import ss6_core as S
import it2_core as I

ap = argparse.ArgumentParser()
ap.add_argument('--spec', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--rec-state', default=None)
ap.add_argument('--beta', type=float, default=0.3)
ap.add_argument('--nchains', type=int, default=1024)
ap.add_argument('--nrounds', type=int, default=40)
ap.add_argument('--between', type=int, default=2)
ap.add_argument('--burn', type=int, default=200)
ap.add_argument('--ch', type=int, default=48)
ap.add_argument('--depth', type=int, default=6)
ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--batch', type=int, default=1024)
ap.add_argument('--ned', type=int, default=400000)
ap.add_argument('--ned-rec', type=int, default=50000)
ap.add_argument('--nval', type=int, default=16384)
ap.add_argument('--nval-rec', type=int, default=2048)
ap.add_argument('--seed', type=int, default=5)
args = ap.parse_args()
clk = I.GpuClock(); res = dict(args=vars(args))
def save(): res['gpu_h'] = clk.hours(); json.dump(res, open(f'distill_{args.out}.json', 'w'), indent=1)

net = C.Net(I.CKPT, dtype='float32', batch=16384)
spec, bd = I.load_spec(args.spec); res['spec_in'] = spec
G = I.HGuide(spec, net, base_dir=bd)
assert G.hop, 'input spec must have a hop'
st_prev = G.st; amp_h = G.amp_h; T = G.T
flat_g = I.load_flat(net, I.resolve(spec['amp_g'], bd))
I.log('distill', spec, 'T', T)

# ---------------- tempered samples of a_g
def samp(beta, nch, nr, seed):
    X, acc = I.sample(net, flat_g, beta, nch, nr, args.between, args.burn, seed); return X, acc
Xtr, acc = samp(args.beta, args.nchains, args.nrounds, 1000 * args.seed + 1)
held = {args.beta: samp(args.beta, 512, 20, 1000 * args.seed + 2)[0].reshape(-1)}
for b in (0.5, 1.0):
    if b not in held: held[b] = samp(b, 512, 20, 1000 * args.seed + 3 + int(10 * b))[0].reshape(-1)
res['samples'] = dict(accept=acc, shape=list(Xtr.shape)); I.log('SAMPLES', res['samples'])
clk.tick('sample'); save()

# ---------------- labels
U = S.build_train_states(Xtr.reshape(-1), 1)
t0 = time.time()
r, _ = S.r_onehop_chunked(U, amp_h, st_prev, chunk=4096)
y = np.where(r <= T, 1, -1).astype(np.int8); aux = np.arcsinh(np.clip(T - r, -1e6, 1e6))
res['labels'] = dict(n_train=int(len(U)), flip_frac=float(np.mean(y < 0)), sec=time.time() - t0,
                     vit_new=int(amp_h.nnew), stored_new=int(st_prev.cache.nnew))
I.log('LABELS', res['labels']); clk.tick('labels'); save()
Xv = held[args.beta][:8192]; rv, _ = S.r_onehop_chunked(Xv, amp_h, st_prev); yv = np.where(rv <= T, 1, -1)

# ---------------- train
snet = S.SignNet(args.ch, args.depth, seed=len(spec['nets']) + 1)
res['npar'] = snet.npar
hist = S.train_net(snet, U, y, aux, Xv, yv, args.epochs, args.lr, args.batch, lam=0.1, seed=args.seed)
res['train_hist'] = hist
fname = f'net_{args.out}.pkl'; snet.save(fname)
lt = I.net_logit(snet, U); res['train_err'] = float(np.mean(np.where(lt >= 0, 1, -1) != y))
res['train_err_on_flips'] = float(np.mean(lt[y < 0] >= 0)) if (y < 0).any() else None
del U, y, aux, r, lt
clk.tick('train'); save()

nets_abs = [I.resolve(p, bd) for p in spec['nets']]
for p in nets_abs:
    if not os.path.exists(os.path.basename(p)): shutil.copy(p, os.path.basename(p))
spec_new = dict(nets=[os.path.basename(p) for p in nets_abs] + [fname],
                Ts_nets=list(spec.get('Ts_nets', [])) + [T], hop=None,
                amp_g=os.path.abspath(I.resolve(spec['amp_g'], bd)) if spec['amp_g'] != 'base' else 'base',
                name=f'{spec.get("name", "G")}_stored', distilled_from=os.path.abspath(args.spec))
I.spec_dump(spec_new, f'spec_{args.out}.json')
Gn = I.HGuide(spec_new, net, amps=G.amps, base_dir='.')

# ---------------- held-out tempered disagreement
H = {}
for b, Xh in held.items():
    a = Gn.signs(Xh); h = G.signs(Xh)
    NBh, Vh = C.neighbors(Xh[:2048]); Yh = np.unique(NBh[Vh])
    H[f'b{b}'] = dict(n=int(len(Xh)), stored_vs_hop=S.disagree(a, h), hop_vs_prev=S.disagree(h, st_prev(Xh)),
                      nbr_stored_vs_hop=S.disagree(Gn.signs(Yh), G.signs(Yh)), n_nbr=int(len(Yh)))
    I.log('HELD', b, H[f'b{b}'])
res['held'] = H; clk.tick('held'); save()

# ---------------- ED
Xe, strue = S.load_ed(args.ned)
s_new = Gn.signs(Xe); s_hop = G.signs(Xe); s_prev = st_prev(Xe)
ED = dict(n=int(len(Xe)), stored=I.wrong_frac(s_new, strue), hop=I.wrong_frac(s_hop, strue),
          prev_stored=I.wrong_frac(s_prev, strue),
          paired_stored_minus_hop=I.paired_wrong(s_new, s_hop, strue),
          stored_vs_hop_disagree=S.disagree(s_new, s_hop))
for n2 in (200000,):
    ED[f'stored_first{n2}'] = I.wrong_frac(s_new[:n2], strue[:n2]); ED[f'hop_first{n2}'] = I.wrong_frac(s_hop[:n2], strue[:n2])
rec_guide = None
if args.rec_state:
    st = json.load(open(args.rec_state))
    ampo = {}
    def amp_of(p):
        if p not in ampo: ampo[p] = C.Amp(net, I.load_flat(net, p), name=p)
        return ampo[p]
    rec_guide = C.Guide([amp_of(p) for p in st['chain_params']], st['Ts'], amp_of(st['amp_g']))
    if args.ned_rec > 0:
        ne = args.ned_rec; t0 = time.time()
        s_rec = np.concatenate([rec_guide.signs(Xe[i:min(i + 256, ne)]) for i in range(0, ne, 256)])
        ED['rec_n'] = ne; ED['rec_sec'] = time.time() - t0
        ED['rec'] = I.wrong_frac(s_rec, strue[:ne]); ED['stored_on_rec_subset'] = I.wrong_frac(s_new[:ne], strue[:ne])
        ED['hop_on_rec_subset'] = I.wrong_frac(s_hop[:ne], strue[:ne])
        ED['paired_stored_minus_rec'] = I.paired_wrong(s_new[:ne], s_rec, strue[:ne])
        ED['paired_hop_minus_rec'] = I.paired_wrong(s_hop[:ne], s_rec, strue[:ne])
np.savez_compressed(f'ed_signs_{args.out}.npz', stored=s_new.astype(np.int8), hop=s_hop.astype(np.int8))
res['ED'] = ED; I.log('ED', json.dumps(ED)); clk.tick('ed'); save()

# ---------------- variational energy on x ~ a_g^2 (paired)
nch = 2048
Xs, _ = I.sample(net, flat_g, 1.0, nch, max(1, args.nval // nch), 4, 200, 1000 * args.seed + 7)
Xs = Xs.reshape(-1)
dn = Gn.local_chunked(Xs, 512); dh = G.local_chunked(Xs, 256)
EN = dict(n=int(len(Xs)), H_stored=I.chain_mean_se(dn['eh'] / 36, nch), H_hop=I.chain_mean_se(dh['eh'] / 36, nch),
          paired_stored_minus_hop=I.chain_mean_se((dn['eh'] - dh['eh']) / 36, nch))
clk.tick('energy_stored_hop'); res['energy'] = EN; save()
if rec_guide is not None and args.nval_rec > 0:
    nr = args.nval_rec; nchr = min(nch, nr); Xr = Xs[:nr]   # first rows: nchr chains x rows
    t0 = time.time()
    dr = C.local_chunked(rec_guide, Xr, 16)
    EN['rec_n'] = nr; EN['rec_sec'] = time.time() - t0
    EN['H_rec_subset'] = I.chain_mean_se(dr['eh'] / 36, nchr)
    EN['paired_stored_minus_rec'] = I.chain_mean_se((dn['eh'][:nr] - dr['eh']) / 36, nchr)
    EN['paired_hop_minus_rec'] = I.chain_mean_se((dh['eh'][:nr] - dr['eh']) / 36, nchr)
    clk.tick('energy_rec')
res['energy'] = EN; I.log('ENERGY', json.dumps(EN))
res['vit_evals'] = int(net.neval)
save(); I.log('DONE', json.dumps(res['gpu_h']))
