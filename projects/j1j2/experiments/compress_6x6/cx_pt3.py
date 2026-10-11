"""Round-3 pre-tests (README): per-bond base input for (a) the re-base and (b) the V/W predictor, and (c) the
rolling-base arm (re-base features = psi_P's one-hop scalars + the one-hop scalars of an earlier re-based guide G_3,
which is itself a psi_P-one-hop net, so its features cost two hops of psi_P at every k).
  python cx_pt3.py OUT SPEC.json     SPEC: mode 'rebase' | 'vw', ...
"""
import os, sys, time
import numpy as np
from cx_lib import *
from cx_bond import *
import optax

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_pt3.json'); T00 = time.time()
lab = Lab(); sec = lab.sec
FEAT_BASE = base_features(lab)
res = dict(spec=SPEC, arms=[])
MODE = SPEC['mode']

if MODE == 'rebase':
    RD, IT = SPEC['run_dir'], SPEC['it']
    chop = json.load(open(os.path.join(RD, 'cx_chop.json')))['iters'][IT - 1]
    la_w = jnp.asarray(np.load(os.path.join(RD, f'la_w_{IT}.npy')))
    s = jnp.asarray(np.load(os.path.join(RD, f's_{IT}.npy')).astype(np.float32))
    H_w = lab.H_site(la_w, s); H_prev = chop['H_G_newsign']
    res.update(H_w=H_w, H_prev=H_prev, H_chop=chop['H'], retention_chop=chop['retention_step'])
    D = la_w - lab.lP0
    CDF, LIW, DLR, _ = tail_proposal(lab, D, la_w, s); del D
    for arm in SPEC['arms']:
        t0 = time.time(); tag = arm['tag']
        if arm['net'] == 'bond':
            m = BondFeatNet(lab, FEAT_BASE, s, seed=50 + IT, C=arm.get('C', 32))
            sec.offload()
            fl, curve = fit_edge_bond(lab, m, la_w, s, CDF, LIW, DLR, arm['steps'], 50 + IT, lr=arm.get('lr', 1e-3), tag=tag)
            sec.reload(); g = m.table(fl)[:, 0]
        elif arm['net'] == 'inc':                               # residual re-base: G = G_ref + h, h fit to la_w - G_ref
            G_ref = lab.lP0 + FeatNet(FEAT_BASE, 0).table(jnp.asarray(np.load(arm['G_ref_params']), jnp.float32), sec.reps, lab.TIDX)
            TF = FEAT_BASE
            if arm.get('with_ref_feats'):
                lanr, Vr, Wr = lab.VW(G_ref, s)
                Egr = FrozenOp(lab, G_ref, s).energy(G_ref)
                _, _, measr = lab.proposal_q(lab.T_of(Vr, Wr, Egr), G_ref, s)
                TF = jnp.concatenate([FEAT_BASE, lab.feat_table(lanr, Vr, Wr, measr)], 1); del lanr, Vr, Wr, measr
            inc = la_w - G_ref
            CDFi, LIWi, DLRi, _ = tail_proposal(lab, inc, la_w, s); del inc   # sampling targeted at the increment
            m = FeatNet(TF, seed=50 + IT)
            sec.offload()
            fl, curve = fit_edge(lab, m, la_w, s, CDFi, LIWi, DLRi, arm['steps'], 50 + IT, lr=arm.get('lr', 1e-3), tag=tag)
            sec.reload(); g = G_ref - lab.lP0 + m.table(fl, sec.reps, lab.TIDX); del TF, CDFi, LIWi, DLRi, G_ref
        else:                                                   # 'roll': scalar features of psi_P and of G_ref
            if arm.get('G_ref_params'):                       # rebuild a re-based guide from its stored params
                G_ref = lab.lP0 + FeatNet(FEAT_BASE, 0).table(jnp.asarray(np.load(arm['G_ref_params']), jnp.float32), sec.reps, lab.TIDX)
            else:
                G_ref = jnp.asarray(np.load(arm['G_ref']))
            s_ref = jnp.asarray(np.load(arm['s_ref']).astype(np.float32))
            lanr, Vr, Wr = lab.VW(G_ref, s_ref)
            Egr = FrozenOp(lab, G_ref, s_ref).energy(G_ref)
            _, _, measr = lab.proposal_q(lab.T_of(Vr, Wr, Egr), G_ref, s_ref)
            TF = jnp.concatenate([FEAT_BASE, lab.feat_table(lanr, Vr, Wr, measr)], 1); del lanr, Vr, Wr, measr, G_ref
            m = FeatNet(TF, seed=50 + IT)
            sec.offload()
            fl, curve = fit_edge(lab, m, la_w, s, CDF, LIW, DLR, arm['steps'], 50 + IT, lr=arm.get('lr', 1e-3), tag=tag)
            sec.reload(); g = m.table(fl, sec.reps, lab.TIDX); del TF
        G = lab.lP0 + g; del g
        H = lab.H_site(G, s)
        a = dict(tag=tag, npar=m.npar, H=H, curve=curve, retention_step=(H_prev - H) / (H_prev - H_w),
                 retention_acc=(H_START - H) / (H_START - H_w), bond_rms=bond_rms(lab, la_w, G - la_w), sec=time.time() - t0)
        res['arms'].append(a); dump(F, res)
        log(f'== {tag}: <H> {H:.5e} (la_w {H_w:.5e}, prev {H_prev:.5e})  retention step {a["retention_step"]:.3f} '
            f'acc {a["retention_acc"]:.3f}  bond rms {a["bond_rms"]:.4f}  npar {m.npar}  ({a["sec"]:.0f}s)')
        if SPEC.get('save_G'): np.save(os.path.join(OUT, f'G_{tag}.npy'), np.asarray(G))
        del m, fl, G

elif MODE == 'vw':
    REFS = json.load(open(os.path.join(TABDIR, 'cx_replay.json')))['refs']
    STEPS = SPEC.get('steps', 20000); B = SPEC.get('B', 512)
    for k in SPEC['ks']:
        t0 = time.time(); rec = dict(k=k)
        la, s = load_guide(k)
        ctx = GuideCtx(lab, la, s, load_loop_params(k + 1), sec.E0 + N * REFS[str(k)]['E_FN_stack'])
        rec['frac_exact'], _ = ctx.step()
        uV = jnp.log(jnp.maximum(ctx.V, 0) + 1e-6); uW = jnp.log(jnp.maximum(ctx.W, 0) + 1e-6)
        mm = ctx.meas
        st = [(float(jnp.sum(mm * q)), float(jnp.sqrt(jnp.sum(mm * (q - jnp.sum(mm * q)) ** 2)))) for q in (uV, uW)]
        TGT = jnp.stack([(uV - st[0][0]) / st[0][1], (uW - st[1][0]) / st[1][1]], 1).astype(jnp.float32)
        Q, _, _ = lab.proposal_q(lab.T_of(ctx.V, ctx.W, ctx.Eg), la, s)
        Q = jnp.where(lab.HO, 0.0, Q); Q = Q / jnp.sum(Q); CDF = jnp.cumsum(Q)
        LW = jnp.where(Q > 0, jnp.log(jnp.maximum(mm, 1e-300)) - jnp.log(jnp.maximum(Q, 1e-300)), -jnp.inf); del Q
        m = BondFeatNet(lab, FEAT_BASE, s, seed=10 * k, nout=2, zero_head=True)
        sched = optax.warmup_cosine_decay_schedule(0.0, SPEC.get('lr', 3e-3), 200, STEPS, SPEC.get('lr', 3e-3) * 0.02)
        opt = optax.adam(sched); flat = m.flat0; ost = opt.init(flat)

        @jax.jit
        def batch(key):
            idx = jnp.clip(jnp.searchsorted(CDF, jax.random.uniform(key, (B,), f64) * CDF[-1]), 0, sec.D - 1)
            x = sec.reps[idx]; v, r, kp = rfeat(lab, x, s)
            lw = LW[idx]
            return x, v, r, kp, FEAT_BASE[idx], TGT[idx], jnp.exp(lw - jnp.max(lw))

        def loss(fl, x, v, r, kp, t, tg, iw, G):
            pr = m.f_aug(fl, bits(x), r, v, kp, t, G) + t[:, 1:3]   # residual on the base's own (log V_P, log W_P)
            iw = iw / jnp.mean(iw)
            return jnp.mean(iw[:, None] * (pr - tg) ** 2)

        @jax.jit
        def step(fl, ost, key):
            k1, k2 = jax.random.split(key)
            bd = batch(k1); G = jax.random.randint(k2, (B,), 0, 16)
            l, g = jax.value_and_grad(loss)(fl, *bd, G)
            upd, ost = opt.update(g, ost, fl)
            return optax.apply_updates(fl, upd), ost, l
        vb = [batch(jax.random.PRNGKey(5000 + i)) for i in range(8)]
        vloss = jax.jit(lambda fl, *b: loss(fl, *b, jnp.zeros(B, jnp.int32)))
        curve = []; key = jax.random.PRNGKey(300 + k); sec.offload()
        for it in range(0, STEPS + 1):
            if it % 2000 == 0:
                curve.append((it, float(np.mean([float(vloss(flat, *b)) for b in vb])))); log(f'  [vw-bond k={k}] step {it} val {curve[-1][1]:.4e}')
            if it == STEPS: break
            key, kk = jax.random.split(key)
            flat, ost, l = step(flat, ost, kk)
        sec.reload(); rec['curve'] = curve
        P = m.table(flat) + FEAT_BASE[:, 1:3].astype(f64)
        Vh = jnp.maximum(jnp.exp(P[:, 0] * st[0][1] + st[0][0]) - 1e-6, 0.0)
        Wh = jnp.maximum(jnp.exp(P[:, 1] * st[1][1] + st[1][0]) - 1e-6, 0.0); del P
        rec['err_hat'] = ctx.vw_err(Vh, Wh)
        for split, msk in (('train', ~lab.HO), ('test', lab.HO)):
            pa = jnp.where(msk, ctx.PA, 0.0)
            ev = jnp.log(Vh + 1e-6) - uV; ew = jnp.log(Wh + 1e-6) - uW
            rec[f'u_rms_{split}'] = (float(jnp.sqrt(jnp.sum(pa * ev * ev) / jnp.sum(pa))), float(jnp.sqrt(jnp.sum(pa * ew * ew) / jnp.sum(pa))))
        rec['frac_hat'], rec['SI_hat'] = ctx.step((ctx.lan, Vh, Wh))
        rec['capture_rel'] = rec['frac_hat'] / rec['frac_exact']; rec['npar'] = m.npar; rec['sec'] = time.time() - t0
        log(f'== k={k}: exact {rec["frac_exact"]:.4f} amortised(bond) {rec["frac_hat"]:.4f} rel {rec["capture_rel"]:.3f} '
            f'err {rec["err_hat"]} u-rms train/test {rec["u_rms_train"]} {rec["u_rms_test"]}')
        res['arms'].append(rec); dump(F, res)
        del ctx, Vh, Wh, uV, uW, TGT, CDF, LW, m
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
