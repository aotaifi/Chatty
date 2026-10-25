"""Test L8 stage 2: refit the RBM+PP to the FN ground state phi_0 of g_0 from dumped walkers (relative fidelity R, SR ascent).

  python l8_refit.py OUT --params0 P.npz --consts consts.json --dumps D1,D2 [--L 8 --proj translations --steps 40 --lr 0.3 --shift 0.1
                     --maxn 0.02 --chains 1024 --holdout 0.2 --binb 2 --seed 1 --chunk 256 --gate_sigma 3]
Writes OUT/progress.txt (PROGRESS lines; health line WEIGHT_HEALTH; last line REFIT_DONE verdict ...), OUT/refit.json (curve per step:
R_train, R_held and errors, Q, Ninv, |d|_S, seconds), OUT/health.json, OUT/params_final.npz (same keys as params0; steps = 0 writes params0's
arrays unchanged), OUT/params_tNNN.npz every --save_every steps.
Held-out: default the whole second chain (--holdout_chain 1); params_final.npz = the step with the LARGEST held-out R (early stopping; refit.json
'final.selected_step', curves R_held and R_train per step); the gate applies to that step.
Exit codes: 0 refit done and held-out R rose by more than gate_sigma x its error (or --no_gate); 3 WEIGHT-HEALTH STOP (n_eff/n < 0.1 or a
single weight > 1%: design failure, nothing fitted); 4 held-out R did not rise (stage 3 must not run: afterok blocks it).
--sign_broken: weights v/|g_0| instead of v/g_0 (control iii).
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'rbmpp_6x6')):
    sys.path.insert(0, d)
from rbmpp_lib import Lattice
from gfmc_run import load_consts
import l8_lib as LL

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--params0', required=True); ap.add_argument('--consts', required=True)
    ap.add_argument('--dumps', required=True, help='comma-separated run dirs, each with dump/blk_*.npz (one per chain)')
    ap.add_argument('--L', type=int, default=8); ap.add_argument('--proj', default='translations')
    ap.add_argument('--steps', type=int, default=40); ap.add_argument('--lr', type=float, default=0.3)
    ap.add_argument('--shift', type=float, default=0.1); ap.add_argument('--maxn', type=float, default=0.02)
    ap.add_argument('--chains', type=int, default=1024); ap.add_argument('--therm', type=int, default=50)
    ap.add_argument('--sweeps_per_step', type=int, default=8, help='VMC sweeps pooled per step (the VMC term of the gradient is a main noise source)')
    ap.add_argument('--final_sweeps', type=int, default=8, help='extra VMC sweeps for the final held-out R')
    ap.add_argument('--holdout_chain', type=int, default=1, help='index (in --dumps) of the chain held out entirely (default 1: train on the first '
                    'chain, hold out the second); -1: old split, last --holdout fraction of every chain')
    ap.add_argument('--holdout', type=float, default=0.2); ap.add_argument('--binb', type=int, default=2)
    ap.add_argument('--first_block', type=int, default=None, help='ignore dump files of blocks below this index')
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--gate_sigma', type=float, default=3.0); ap.add_argument('--no_gate', action='store_true')
    ap.add_argument('--health_neff', type=float, default=0.1); ap.add_argument('--health_maxshare', type=float, default=0.01)
    ap.add_argument('--no_health_stop', action='store_true'); ap.add_argument('--sign_broken', action='store_true')
    ap.add_argument('--save_every', type=int, default=1); ap.add_argument('--select', default='best', choices=['best', 'last']); ap.add_argument('--maxsec', type=float, default=1e18)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prog = os.path.join(a.out, 'progress.txt')

    def P(msg):
        line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}'
        print(line, flush=True); open(prog, 'a').write(line + '\n')

    T0 = time.time()
    z = np.load(a.params0); use_rbm = bool(z['rbm']) if 'rbm' in z.files else True
    lat = Lattice(a.L, 0.5, a.proj)
    c = load_consts(a.consts, 1); alpha, E, w = c['alpha'], c['E'], c['w']
    model = LL.Model(lat, z, use_rbm, a.chunk); flat0 = np.asarray(model.flat0)
    P(f'l8_refit L {a.L} proj {a.proj} params0 {a.params0} consts {a.consts} alpha {alpha:.8f} E {E:.8f} w {w:.6f} nparams {model.P} '
      f'steps {a.steps} lr {a.lr} shift {a.shift} maxn {a.maxn} chains {a.chains} sign_broken {a.sign_broken}')
    dirs = [d for d in a.dumps.split(',') if d]
    D = LL.load_dumps(dirs, a.holdout, a.first_block, None if a.holdout_chain < 0 else a.holdout_chain)
    if a.holdout_chain >= 0 and len(dirs) < 2:
        raise SystemExit('--holdout_chain needs >= 2 chains in --dumps (or --holdout_chain -1)')
    t = time.time(); G = LL.guide_at(model, flat0, D['X'], alpha, E, w); tg = time.time() - t
    tr = LL.WalkerSet(D, G, ~D['held'], w, a.binb, a.sign_broken); ho = LL.WalkerSet(D, G, D['held'], w, a.binb, a.sign_broken)
    al = LL.WalkerSet(D, G, np.ones(len(D['p']), bool), w, a.binb, a.sign_broken)
    # consistency: the dumped FN local energies reproduce the mixed estimator per block (proves the dump is the walk's state)
    P(f'walkers {len(D["p"])} (train {len(tr.pw)} in {tr.nblocks} blocks, held-out {len(ho.pw)} in {ho.nblocks} blocks, chains {len(dirs)}); '
      f'guide values {tg:.0f}s ({tg / len(D["p"]) * 1e3:.3f} ms/walker); h<0 fraction {np.mean(G["h"] < 0):.2e}; min|h| {np.min(np.abs(G["h"])):.2e}; '
      f'mixed E_L/N from dump {np.sum(al.pw * D["EL"]) / lat.N:.7f}')
    H = dict(all=al.health(al.r0), train=tr.health(tr.r0), held=ho.health(ho.r0))
    # the same per 20-block window (the production sets are larger; the health of a single block window is the worst case)
    json.dump(H, open(os.path.join(a.out, 'health.json'), 'w'), indent=1)
    h = H['all']
    bad = (h['neff_over_n'] < a.health_neff) or (h['max_share'] > a.health_maxshare)
    P(f'WEIGHT_HEALTH n {h["n"]} neff/n {h["neff_over_n"]:.4f} (plain walker weights {h["plain_neff_over_n"]:.4f}) max_share {h["max_share"]:.2e} '
      f'neg_weight_frac {h["neg_frac_weight"]:.2e} -> {"STOP" if bad else "OK"} (limits neff/n >= {a.health_neff}, max_share <= {a.health_maxshare})')
    if bad and not a.no_health_stop:
        P('REFIT_DONE verdict HEALTH_STOP (design failure: weights x v/g too heavy; nothing fitted)'); sys.exit(3)
    vmc = LL.VMC(model, a.chains, a.seed, therm=a.therm); rng = np.random.default_rng(1000 + a.seed)
    flat = flat0.copy(); curve = []; flats = [flat0.copy()]
    for it in range(a.steps + 1):
        ts = time.time()
        S = np.concatenate([vmc.sample(flat) for _ in range(a.sweeps_per_step)]); ni, ni_err, l_s, vt_s = LL.ninv(model, flat, flat0, S, w)
        if it == 0:
            r_tr, r_ho = tr.r0, ho.r0
        else:
            r_tr = tr.ratios(model.logpsi(flat, tr.X)); r_ho = ho.ratios(model.logpsi(flat, ho.X))
        qt, qt_e = tr.Q(r_tr); qh, qh_e = ho.Q(r_ho)
        Rt = qt ** 2 * ni; Rh = qh ** 2 * ni
        Rh_e = Rh * np.sqrt((2 * qh_e / qh) ** 2 + (ni_err / ni) ** 2); Rt_e = Rt * np.sqrt((2 * qt_e / qt) ** 2 + (ni_err / ni) ** 2)
        rec = dict(it=it, R_train=Rt, R_train_err=Rt_e, R_held=Rh, R_held_err=Rh_e, Q_train=qt, Q_train_err=qt_e, Q_held=qh, Q_held_err=qh_e,
                   Ninv=ni, Ninv_err=ni_err, acc=vmc.acc, health_train=tr.health(r_tr))
        if it < a.steps and time.time() - T0 < a.maxsec:
            dl, info = LL.sr_step(model, flat, S, vt_s, tr, r_tr, w, a.lr, a.shift, a.maxn, rng)
            flat = flat + dl; rec.update(info); flats.append(flat.copy())
        rec['sec'] = time.time() - ts; curve.append(rec)
        P(f'step {it}/{a.steps} R_held {Rh:.6f} +- {Rh_e:.1e} R_train {Rt:.6f} +- {Rt_e:.1e} Q_held {qh:.6f} Ninv {ni:.6f} +- {ni_err:.1e} '
          f'neff/n_train {rec["health_train"]["neff_over_n"]:.3f} |d|_S {rec.get("dnorm", 0):.3f} rms_dlogv {rec.get("rms_dlogv", 0):.3f} halv {rec.get("halvings", 0)} acc {vmc.acc:.2f} sec {rec["sec"]:.1f}')
        json.dump(dict(args=vars(a), curve=curve, health=H), open(os.path.join(a.out, 'refit.json'), 'w'), indent=1, default=float)
        if a.save_every and it and it % a.save_every == 0:
            LL.save_params(os.path.join(a.out, f'params_t{it:03d}.npz'), model, flats[it], a.params0)
        if time.time() - T0 > a.maxsec:
            P(f'MAXSEC reached after step {it}'); break
    # early stopping: the step with the largest held-out R (step 0 = params0, R = 1 exactly)
    best = int(np.argmax([c_['R_held'] for c_ in curve])) if a.select == 'best' else len(curve) - 1; fbest = flats[best]
    if best == 0:
        np.savez(os.path.join(a.out, 'params_final.npz'), **{k: z[k] for k in z.files})       # params0 unchanged, bitwise
        Rh, Rh_e = curve[0]['R_held'], curve[0]['R_held_err']
    else:
        LL.save_params(os.path.join(a.out, 'params_final.npz'), model, fbest, a.params0)
        # held-out R at the selected step with a fresh, thermalised VMC set for the norm ratio (the walker part Q is the one used to select)
        vf = LL.VMC(model, a.chains, a.seed + 500, therm=a.therm); ns = []
        for k in range(a.final_sweeps):
            S = vf.sample(fbest); ns.append(LL.ninv(model, fbest, flat0, S, w))
        num = np.mean([x[0] for x in ns]); err = np.std([x[0] for x in ns], ddof=1) / np.sqrt(len(ns)) if len(ns) > 1 else ns[0][1]
        r_ho = ho.ratios(model.logpsi(fbest, ho.X)); qh, qh_e = ho.Q(r_ho)
        Rh = qh ** 2 * num; Rh_e = Rh * np.sqrt((2 * qh_e / qh) ** 2 + (err / num) ** 2)
    P(f'SELECTED step {best} of {len(curve) - 1} (largest held-out R {curve[best]["R_held"]:.6f}); curve turned down after it: '
      f'{best < len(curve) - 1}')
    fin = dict(selected_step=best, R_held=Rh, R_held_err=Rh_e, z=(Rh - 1) / Rh_e if Rh_e > 0 else 0.0, steps_done=len(curve) - 1, sec=time.time() - T0)
    ok = a.no_gate or fin['z'] > a.gate_sigma
    json.dump(dict(args=vars(a), curve=curve, health=H, final=fin), open(os.path.join(a.out, 'refit.json'), 'w'), indent=1, default=float)
    P(f'REFIT_DONE verdict {"PASS" if ok else "NO_RISE"} held-out R at selected step {best} {Rh:.6f} +- {Rh_e:.1e} ({fin["z"]:+.1f} sigma, gate {a.gate_sigma}) '
      f'steps {fin["steps_done"]} {fin["sec"]:.0f}s')
    sys.exit(0 if ok else 4)
