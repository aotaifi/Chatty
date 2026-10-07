#!/usr/bin/env python3
"""T3: distil a 'stored + one exact hop' guide sign into ONE composite stored sign net (angle-2 rule).

Input spec G (it2_core): s_G = s_st * sgn(T - r[a_hop, s_st]).  Target: the COMPOSITE label y = s_G * Marshall
(never a factor relative to a previous net, never stacked).  Training set: tempered samples x ~ a^(2 beta) of
--amp (default: the guide amplitude) + ALL one-hop neighbours; BCE only (aux off); --steps Adam steps (default 8e4).
Output spec: nets = [new net], hop = null, amp_g = --amp (the amplitude for the next sign step).
Checks: held-out (beta 0.5, 1) disagreement stored vs s_G; ED w_s of stored and of s_G (x ~ psi0^2).
Usage: python t3_distill.py --spec G.json --amp params.npy --out TAG
"""
import argparse, json, os, shutil
import numpy as np
import ll6_core as C
import ss6_core as S
import it2_core as I

ap = argparse.ArgumentParser()
ap.add_argument('--spec', required=True)
ap.add_argument('--amp', default=None, help='amplitude params for sampling and for the output spec (default: spec amp_g)')
ap.add_argument('--out', required=True)
ap.add_argument('--beta', type=float, default=0.5)
ap.add_argument('--nchains', type=int, default=1024)
ap.add_argument('--nrounds', type=int, default=24)
ap.add_argument('--between', type=int, default=2)
ap.add_argument('--burn', type=int, default=200)
ap.add_argument('--ch', type=int, default=48)
ap.add_argument('--depth', type=int, default=6)
ap.add_argument('--steps', type=int, default=80000)
ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--batch', type=int, default=1024)
ap.add_argument('--ned', type=int, default=200000)
ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--energy-n', type=int, default=0, help='paired <H>_{a, s_net} - <H>_{a, s_G} on x ~ a^2')
ap.add_argument('--vs-vit', type=int, default=1)
args = ap.parse_args()
clk = I.GpuClock(); res = dict(args=vars(args))
def save(): res['gpu_h'] = clk.hours(); json.dump(res, open(f'distill_{args.out}.json', 'w'), indent=1)

net = C.Net(I.CKPT, dtype='float32', batch=16384)
spec, bd = I.load_spec(args.spec); res['spec_in'] = spec
for p in spec['nets']:
    q = I.resolve(p, bd)
    if not os.path.exists(os.path.basename(q)): shutil.copy(q, os.path.basename(q))
G = I.HGuide(spec, net, base_dir=bd)
amp_path = os.path.abspath(args.amp) if args.amp else (I.resolve(spec['amp_g'], bd) if spec['amp_g'] != 'base' else 'base')
flat = I.load_flat(net, amp_path)
I.log('distill composite', spec, 'amp', amp_path)

Xtr, acc = I.sample(net, flat, args.beta, args.nchains, args.nrounds, args.between, args.burn, 1000 * args.seed + 1)
held = {b: I.sample(net, flat, b, 512, 16, args.between, args.burn, 1000 * args.seed + 3 + int(10 * b))[0].reshape(-1) for b in (0.5, 1.0)}
res['samples'] = dict(accept=acc, shape=list(Xtr.shape)); clk.tick('sample'); save()

U = S.build_train_states(Xtr.reshape(-1), 1)
y = (G.signs(U) * C.marshall_vec(U)).astype(np.int8)
res['labels'] = dict(n_train=int(len(U)), frac_minus=float(np.mean(y < 0)))
I.log('LABELS', res['labels']); clk.tick('labels'); save()
Xv = held[args.beta][:8192]; yv = (G.signs(Xv) * C.marshall_vec(Xv)).astype(np.int8)

snet = S.SignNet(args.ch, args.depth, seed=args.seed)
epochs = max(1, int(round(args.steps * args.batch / len(U))))
res['npar'] = snet.npar; res['epochs'] = epochs
hist = S.train_net(snet, U, y.astype(np.float32), np.zeros(len(U), np.float32), Xv, yv, epochs, args.lr, args.batch,
                   lam=0.0, seed=args.seed, logevery=max(1, epochs // 10))
res['train_hist'] = hist
fname = f'net_{args.out}.pkl'; snet.save(fname)
clk.tick('train'); save()

spec_new = dict(nets=[fname], Ts_nets=[None], hop=None, amp_g=amp_path, name=f'{spec.get("name", "G")}_comp',
                distilled_from=os.path.abspath(args.spec))
I.spec_dump(spec_new, f'spec_{args.out}.json')
Gn = I.HGuide(spec_new, net, amps=G.amps, base_dir='.')
H = {}
for b, Xh in held.items():
    H[f'b{b}'] = dict(n=int(len(Xh)), stored_vs_G=S.disagree(Gn.signs(Xh), G.signs(Xh)))
res['held'] = H; I.log('HELD', json.dumps(H)); clk.tick('held'); save()
Xe, strue = S.load_ed(args.ned)
s_new = Gn.signs(Xe); s_G = G.signs(Xe)
res['ED'] = dict(n=int(len(Xe)), stored=I.wrong_frac(s_new, strue), G=I.wrong_frac(s_G, strue),
                 paired_stored_minus_G=I.paired_wrong(s_new, s_G, strue))
I.log('ED', json.dumps(res['ED'])); clk.tick('ed'); save()
if args.energy_n:
    nch = 2048
    Xs, _ = I.sample(net, flat, 1.0, nch, max(1, args.energy_n // nch), 4, 200, 1000 * args.seed + 9); Xs = Xs.reshape(-1)
    dn = Gn.local_chunked(Xs, 1024); dh = G.local_chunked(Xs, 256)
    EN = dict(n=int(len(Xs)), H_net=I.chain_mean_se(dn['eh'] / 36, nch), H_hop=I.chain_mean_se(dh['eh'] / 36, nch),
              paired_net_minus_hop=I.chain_mean_se((dn['eh'] - dh['eh']) / 36, nch))
    if args.vs_vit:
        zx = net.logpsi_c(net.flat0, Xs); NB, V = C.neighbors(Xs); own, col = np.nonzero(V)
        zy = net.logpsi_c(net.flat0, NB[own, col])
        ev = C.diag_vec(V).astype(complex); np.add.at(ev, own, 0.5 * C.JB[col] * np.exp(zy - zx[own])); ev = ev.real
        w = np.exp(2 * (zx.real - dn['lax'])); w /= w.mean()
        EN['net_minus_vit'] = [x / 36 for x in I.jk_ratio_diff(dn['eh'], ev, w, nch)]
        EN['hop_minus_vit'] = [x / 36 for x in I.jk_ratio_diff(dh['eh'], ev, w, nch)]
    res['energy'] = EN; I.log('ENERGY', json.dumps(EN)); clk.tick('energy')
res['vit_evals'] = int(net.neval); save()
I.log('DONE', json.dumps(res['gpu_h']))
