"""Sign step on top of a fully stored sign, label-free energy-optimal threshold with a ROBUST estimate of T.

Input spec G = (a_g, s_st) (hop = null).  New guide G' = (a_g, s_st * sgn(T - r[a_g, s_st])) (one exact hop).
T: energy curve E(T) of a_g s_st sgn(T - r) on --nthr fresh samples x ~ a_g^2 (2-hop shell, stored signs);
   candidates: raw argmin, argmin of the curve smoothed with a Gaussian of width --sigma in r; half-sample
   and chain-bootstrap spreads of both.  Default choice: smoothed.
Held-out (independent chains, --nheld): paired dH of the sign step for the chosen T (and the alternatives),
   <H>_{G'} - E_ViT paired by reweighting to |psi_ViT|^2 (complex ViT local energy, as ll6_vmcpair).
ED: w_s of s_st and of the new sign on --ned samples x ~ psi0^2 (paired), amplitude closeness std(log a - log|psi0|).
Writes spec_<out>.json (G'), sign_<out>.json.
Usage: python it2_sign.py --spec G.json --out TAG [--nthr 65536 --nheld 16384 --sigma 1.0 --choice smooth]
"""
import argparse, json, time, os, shutil
import numpy as np
import jax, jax.numpy as jnp
import ll6_core as C
import ss6_core as S
import it2_core as I

ap = argparse.ArgumentParser()
ap.add_argument('--spec', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--nthr', type=int, default=65536)
ap.add_argument('--nheld', type=int, default=16384)
ap.add_argument('--nch', type=int, default=2048)
ap.add_argument('--sigma', type=float, default=1.0)
ap.add_argument('--choice', default='smooth')
ap.add_argument('--extra-T', default='')          # comma list of extra thresholds to evaluate on the held-out set
ap.add_argument('--ned', type=int, default=400000)
ap.add_argument('--between', type=int, default=4)
ap.add_argument('--burn', type=int, default=200)
ap.add_argument('--seed', type=int, default=3)
ap.add_argument('--skip-vit', action='store_true')
args = ap.parse_args()
clk = I.GpuClock(); res = dict(args=vars(args))
def save(): res['gpu_h'] = clk.hours(); json.dump(res, open(f'sign_{args.out}.json', 'w'), indent=1)

net = C.Net(I.CKPT, dtype='float32', batch=16384)
spec, bd = I.load_spec(args.spec); res['spec_in'] = spec
assert not spec.get('hop')
G = I.HGuide(spec, net, base_dir=bd)
for p in spec['nets']:
    q = I.resolve(p, bd)
    if not os.path.exists(os.path.basename(q)): shutil.copy(q, os.path.basename(q))
amp_path = I.resolve(spec['amp_g'], bd)
flat = I.load_flat(net, amp_path); amp = G.amp_g; st = G.st
nch = args.nch

# ---------------- threshold curve
XT, acc = I.sample(net, flat, 1.0, nch, max(1, args.nthr // nch), args.between, args.burn, 1000 * args.seed + 1)
XT = XT.reshape(-1); clk.tick('sample_thr')
t0 = time.time()
E = I.threshold_edges_guide(XT, amp, st, chunk=256)
res['thr_edges_sec'] = time.time() - t0; clk.tick('thr_edges')
rt = I.robust_threshold(E, nch, sigma=args.sigma)
res['threshold'] = rt; I.log('THRESHOLD', json.dumps({k: v for k, v in rt.items() if not k.startswith('curve')}))
T = rt['T_smooth'] if args.choice == 'smooth' else rt['T_raw']
res['T'] = T
del E; clk.tick('thr_select'); save()

# ---------------- held-out
XB, _ = I.sample(net, flat, 1.0, nch, max(1, args.nheld // nch), args.between, args.burn, 1000 * args.seed + 2)
XB = XB.reshape(-1)
EB = I.threshold_edges_guide(XB, amp, st, chunk=256)
e_old = C.eloc_at_T(EB, 1e300); e_new = C.eloc_at_T(EB, T)
Ts_try = {'chosen': T, 'raw': rt['T_raw'], 'smooth': rt['T_smooth']}
for t in [x for x in args.extra_T.split(',') if x]: Ts_try[f'T{t}'] = float(t)
HO = dict(n=int(len(XB)), H_old_site=I.chain_mean_se(e_old / 36, nch), H_new_site=I.chain_mean_se(e_new / 36, nch))
for k, t in Ts_try.items():
    HO[f'dH_{k}_site'] = I.chain_mean_se((C.eloc_at_T(EB, t) - e_old) / 36, nch); HO[f'T_{k}'] = t
HO['flip_frac'] = float(np.mean(EB['rx_node'] > T))
bB, cB, cuB = C.energy_curve(EB); iB = int(np.argmin(cuB))
HO['T_heldout_optimal_biased'] = float(cB[iB]); HO['gain_heldout_optimal_biased_site'] = float((cuB[iB] - bB) / 36)
res['heldout'] = HO; I.log('HELDOUT', json.dumps(HO)); clk.tick('heldout'); save()
del EB

# <H>_{G'} - E_ViT (paired by reweighting on the same x ~ a_g^2)
if not args.skip_vit:
    zx = net.logpsi_c(net.flat0, XB)
    NB, V = C.neighbors(XB); own, col = np.nonzero(V)
    zy = net.logpsi_c(net.flat0, NB[own, col])
    ev = C.diag_vec(V).astype(complex); np.add.at(ev, own, 0.5 * C.JB[col] * np.exp(zy - zx[own])); ev = ev.real
    lax = amp(XB); w = np.exp(2 * (zx.real - lax)); w /= w.mean()
    d_new = [x / 36 for x in I.jk_ratio_diff(e_new, ev, w, nch)]
    d_old = [x / 36 for x in I.jk_ratio_diff(e_old, ev, w, nch)]
    res['vs_vit'] = dict(delta_new_site=d_new, delta_old_site=d_old,
                         H_new_via_ref=[I.E_VIT_SITE[0] + d_new[0], float(np.hypot(I.E_VIT_SITE[1], d_new[1]))],
                         H_old_via_ref=[I.E_VIT_SITE[0] + d_old[0], float(np.hypot(I.E_VIT_SITE[1], d_old[1]))],
                         ess=float(w.sum() ** 2 / np.sum(w * w) / len(w)))
    I.log('VSVIT', json.dumps(res['vs_vit'])); clk.tick('vs_vit'); save()

spec_new = dict(spec); spec_new['nets'] = [os.path.basename(I.resolve(p, bd)) for p in spec['nets']]
spec_new['amp_g'] = os.path.abspath(amp_path) if spec['amp_g'] != 'base' else 'base'
spec_new['hop'] = dict(amp=spec_new['amp_g'], T=float(T)); spec_new['name'] = f'{spec.get("name", "G")}_sign'
spec_new['sign_from'] = os.path.abspath(args.spec)
I.spec_dump(spec_new, f'spec_{args.out}.json')
Gn = I.HGuide(spec_new, net, amps=G.amps, base_dir='.')

# ---------------- ED
if args.ned > 0:
    Xe, strue = S.load_ed(args.ned)
    s_new = Gn.signs(Xe); s_old = st(Xe)
    ED = dict(n=int(len(Xe)), new=I.wrong_frac(s_new, strue), old=I.wrong_frac(s_old, strue),
              paired_new_minus_old=I.paired_wrong(s_new, s_old, strue))
    na = min(len(Xe), 100000)
    P = np.concatenate([np.load(f)['psi'] for f in ('samples_psi0sq_6x6.npz', 'samples_extra_psi0sq_6x6.npz')])[:na]
    la = amp(Xe[:na]); l0 = net.logabs(net.flat0, Xe[:na])
    ED['std_dlog_amp'] = float(np.std(la - np.log(np.abs(P)))); ED['std_dlog_vit'] = float(np.std(l0 - np.log(np.abs(P))))
    np.savez_compressed(f'ed_signs_{args.out}.npz', new=s_new.astype(np.int8), old=s_old.astype(np.int8))
    res['ED'] = ED; I.log('ED', json.dumps(ED)); clk.tick('ed')
res['vit_evals'] = int(net.neval); save(); I.log('DONE', json.dumps(res['gpu_h']))
