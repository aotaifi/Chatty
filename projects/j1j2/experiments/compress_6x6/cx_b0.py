"""Test B0 readout (README "Test B0"): the VMC-trained PP + 3x3 backflow network, read out exactly on the 15.8M orbits.
  python cx_b0.py OUT SPEC.json
SPEC: refresh (run dir with s2.npy), pp_table (sym_tables_PP.npz: lP, sP of the unmodified start), pp_params (rpT1_pp params_final.npz),
  params (trained params_final.npz of rbmpp_bf_train.py, or a theta .npy; omit with control_only), control_only [false],
  code_sha256 {cx_b0.py, cx_lib.py, rbmpp_bf_lib.py, rbmpp_lib.py} [optional guard], table_batch [512], ckpt_sec [1200], n_check [4096].
Stages
  control (test d): on the unmodified start table (sym_tables_PP.npz), E_FN with sign s_2 at zero hop (5.3677e-5, T0) and one hop
    alpha_V (2.3692e-5 = O2a E_FN(g_0)), rel 1e-3, else exit 2; the same two numbers with the PP's OWN sign (sP of the table) and
    the |psi0|^2-weighted fraction of orbits where the own sign disagrees with s0 and with s_2.  Network check: the rbmpp_bf network at
    the start params (u = 0) against the table on n_check random orbits (log|psi| up to a constant, sign).
  table: |psi| and sign of the trained network on all orbit representatives (whole space group x spin flip, 576 images, float64),
    resumable (OUT/table_part.npz).
  readout: E_FN with sign s_2 at zero hop and at one hop with alpha_V (O1 rule; the cx_o2b.hop code), and with the network's own sign
    at zero hop and one hop (alpha_V of the own-sign state); disagreement weights of the own sign with s0 and s_2.
Disagreement weights are |psi0|^2 (exact ground state) weights; a global sign of a sign field is a convention, so both the raw weight
w and min(w, 1 - w) are recorded.  Output OUT/cx_b0.json, OUT/progress.txt.
"""
import os, sys, time, hashlib
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
HERE_ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE_, '..', 'rbmpp_6x6'))
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_b0.json'); T00 = time.time()
CONTROL_ONLY = bool(SPEC.get('control_only', False))
REF = dict(EFN_zero_PP=5.3677e-5, EFN_g0=2.369190165069707e-05)     # T0; O2a (o2a_17430739), same as cx_o2b.py
res = dict(spec=SPEC, done=False)


def progress(msg):
    log(msg); open(os.path.join(OUT, 'progress.txt'), 'a').write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}\n')


CODE_FILES = ('cx_b0.py', 'cx_lib.py', 'rbmpp_bf_lib.py', 'rbmpp_lib.py')
SHA_RUN = {fn: (hashlib.sha256(open(os.path.join(HERE_, fn), 'rb').read()).hexdigest() if os.path.exists(os.path.join(HERE_, fn)) else None) for fn in CODE_FILES}
SHA_EXP = SPEC.get('code_sha256'); res['code_sha256'] = dict(expected=SHA_EXP, running=SHA_RUN, dir=HERE_)
if SHA_EXP is not None:
    bad = [fn for fn in CODE_FILES if SHA_EXP.get(fn) != SHA_RUN[fn]]
    if bad:
        res['code_sha256']['mismatch'] = bad; dump(F, res); progress(f'CODE DRIFT: sha256 mismatch for {bad}; exit 3'); sys.exit(3)
dump(F, res)
from rbmpp_bf_lib import Lattice, make_bf_fns, theta_from_f, NTH

lab = Lab(); sec = lab.sec


def norm_la(la):
    a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))


def fn(la, s, maxit=400):
    _, u, info = sec.fn_solve(la, s, maxit=maxit); del u
    return info['dE_FN_site'], info


RF = SPEC['refresh']
s2 = jnp.asarray(np.load(os.path.join(RF, 's2.npy')).astype(np.float32))
z = np.load(TABLE); amp = z['amp']; del z
la0 = norm_la(jnp.log(jnp.maximum(jnp.abs(jnp.asarray(amp)), 1e-300)))
s0 = jnp.where(jnp.asarray(amp) >= 0, 1.0, -1.0).astype(jnp.float32); del amp
W0 = sec.n * jnp.exp(2 * la0); W0 = W0 / jnp.sum(W0); del la0


def disagree(s):
    """|psi0|^2 weight of the orbits where sign field s differs from s0 / s_2 (raw w and min(w, 1 - w): global sign is a convention)."""
    w0 = float(jnp.sum(jnp.where(s != s0, W0, 0.0))); w2 = float(jnp.sum(jnp.where(s != s2, W0, 0.0)))
    return dict(w_ne_s0=w0, w_ne_s0_modsign=min(w0, 1 - w0), w_ne_s2=w2, w_ne_s2_modsign=min(w2, 1 - w2))


def hop(la, s, tag):
    """alpha_V (bounded argmin of the exact <H> of |a||1 + alpha (E_L - E)| with sign s) and E_FN of that state (cx_o2b.hop)."""
    t0 = time.time(); lb = norm_la(la)
    _, V, W = lab.VW(lb, s); v = sec.vec(lb, s); Eb = float(v @ sec.H(v)); del v
    ELb = lab.D0 + V - W; del V, W
    ok_fin = bool(jnp.all(jnp.isfinite(ELb)))
    sdE = float(jnp.sqrt(jnp.sum(sec.vec(lb, s) ** 2 * (ELb - Eb) ** 2)))

    def la_c(a): return lb + jnp.log(jnp.maximum(jnp.abs(1.0 + a * (ELb - Eb)), 1e-30))
    lo, hi = -3.0 / sdE, 3.0 / sdE
    r_ = minimize_scalar(lambda a: float(lab.H_site(la_c(a), s)), bounds=(lo, hi), method='bounded', options=dict(maxiter=100, xatol=1e-4 / sdE))
    aV = float(r_.x); HV = float(r_.fun)
    g = norm_la(la_c(aV)); del ELb
    e, info = fn(g, s)
    rec = dict(tag=tag, alpha_V=aV, H_V=HV, EFN=e, sdE=sdE, E_base=Eb, bound=hi, at_bound=bool(abs(abs(aV) - hi) < 1e-3 * hi), nfev=r_.nfev,
               EL_finite=ok_fin, lobpcg_it=info['lobpcg_it'], fn_res=info['res'], neg_weight=info['neg_weight'], sec=time.time() - t0)
    progress(f'[{tag}] alpha_V {aV:.5f} (nfev {r_.nfev}) <H>-E0 {HV:.4e} E_FN {e:.5e} (lobpcg {info["lobpcg_it"]}, res {info["res"]:.1e}, EL finite {ok_fin})')
    return rec


def readout(la, s, name):
    """zero hop and one hop E_FN with sign s; the disagreement weights of s."""
    out = dict(sign=name, disagree=disagree(s))
    e0, info = fn(norm_la(la), s)
    out['EFN_zero_hop'] = e0; out['zero_hop_info'] = dict(lobpcg_it=info['lobpcg_it'], res=info['res'], neg_weight=info['neg_weight'])
    progress(f'[{name}] zero-hop E_FN {e0:.5e}  disagreement {out["disagree"]}')
    out['one_hop'] = hop(la, s, f'{name} one hop'); out['EFN_one_hop'] = out['one_hop']['EFN']
    return out


# ================================================================================================ control (test d), unmodified start
z = np.load(SPEC['pp_table']); lpp = jnp.asarray(z['lP'], jnp.float64); sP = jnp.asarray(z['sP'].astype(np.float32)); del z
ctl = dict(s2=readout(lpp, s2, 'ctl s2'))
ctl['rel_zero_s2'] = abs(ctl['s2']['EFN_zero_hop'] - REF['EFN_zero_PP']) / REF['EFN_zero_PP']
ctl['rel_one_s2'] = abs(ctl['s2']['EFN_one_hop'] - REF['EFN_g0']) / REF['EFN_g0']
ctl['ref'] = REF; ctl['ok'] = bool(ctl['rel_zero_s2'] < 1e-3 and ctl['rel_one_s2'] < 1e-3)
res['control'] = ctl; dump(F, res)
progress(f'CONTROL s2: zero {ctl["s2"]["EFN_zero_hop"]:.5e} vs {REF["EFN_zero_PP"]:.4e} (rel {ctl["rel_zero_s2"]:.1e}); '
         f'one hop {ctl["s2"]["EFN_one_hop"]:.5e} vs {REF["EFN_g0"]:.5e} (rel {ctl["rel_one_s2"]:.1e}); ok={ctl["ok"]}')
ctl['own'] = readout(lpp, sP, 'ctl own'); dump(F, res)
# network (u = 0, start params) against the table
lat = Lattice(6, 0.5, 'full'); fb = make_bf_fns(lat, jnp.float64)
net_v = jax.jit(jax.vmap(lambda th, x: (fb['logpsi'](th, ((((x >> AR) & jnp.uint64(1)).astype(jnp.float64)) * 2 - 1)),
                                       fb['sign'](th, ((((x >> AR) & jnp.uint64(1)).astype(jnp.float64)) * 2 - 1))), in_axes=(None, 0)))
pz = np.load(SPEC['pp_params']); th_start = jnp.asarray(theta_from_f(pz['f']))
rng = np.random.default_rng(0); idx = np.sort(rng.choice(int(sec.D), int(SPEC.get('n_check', 4096)), replace=False))
xs = sec.reps[jnp.asarray(idx)]
def net_batched(th_, x_, b=256):
    o = [net_v(th_, x_[i:i + b]) for i in range(0, x_.shape[0], b)]
    return jnp.concatenate([q[0] for q in o]), jnp.concatenate([q[1] for q in o])


ln, sn = net_batched(th_start, xs); d_ = np.asarray(ln - lpp[jnp.asarray(idx)])
ctl['net_vs_table'] = dict(n=int(idx.size), logabs_rms_after_offset=float(np.std(d_)), logabs_max_after_offset=float(np.max(np.abs(d_ - d_.mean()))),
                           sign_mismatch_fraction_mod_global=float(min(np.mean(np.asarray(sn) != np.asarray(sP[jnp.asarray(idx)])),
                                                                       np.mean(np.asarray(sn) == np.asarray(sP[jnp.asarray(idx)])))))
progress(f'[control] network(start) vs table: {ctl["net_vs_table"]}'); dump(F, res)
if not ctl['ok']:
    res['stop'] = 'CONTROL FAILED'; res['sec'] = time.time() - T00; res['done'] = True; dump(F, res); progress('CONTROL FAILED: exit 2'); sys.exit(2)
if CONTROL_ONLY:
    res['stop'] = 'control_only'; res['sec'] = time.time() - T00; res['done'] = True; dump(F, res); progress(f'DONE control_only {res["sec"]:.0f}s'); sys.exit(0)

# ================================================================================================ table of the trained network
PZ = SPEC['params']
th = jnp.asarray(np.load(PZ) if PZ.endswith('.npy') else np.load(PZ)['theta'], jnp.float64); assert th.size == NTH
res['params'] = PZ; res['params_sha256'] = hashlib.sha256(open(PZ, 'rb').read()).hexdigest()
th_np = np.asarray(th); res['theta_summary'] = dict(max_abs_u=float(np.max(np.abs(th_np[36 * 36:]))), f_dist_from_start=float(np.linalg.norm(th_np[:1296] - np.asarray(th_start)[:1296])))
PART = os.path.join(OUT, 'table_part.npz'); TB = int(SPEC.get('table_batch', 512)); D = int(sec.D)
if os.path.exists(PART):
    zz = np.load(PART); LA = zz['la']; SG = zz['sg']; i0 = int(zz['i0'])
    assert zz['th_sha'] == res['params_sha256']; progress(f'table resumed at orbit {i0}/{D}')
else:
    LA = np.zeros(D, np.float64); SG = np.zeros(D, np.int8); i0 = 0
sec.offload(); t0 = time.time(); tl = t0; reps = sec.reps
DRY_N = int(SPEC.get('dry_orbits', 0))                  # plumbing test only: the network on the first dry_orbits orbits, the start table elsewhere
ABORT = int(SPEC.get('dry_abort_at', 0))                # plumbing test only: exit(9) after this many orbits (to test the resume)
if DRY_N and i0 == 0:
    LA[:] = np.asarray(lpp); SG[:] = np.asarray(sP).astype(np.int8)
for i in range(i0, DRY_N if DRY_N else D, TB):
    x = reps[i:i + TB]; m = x.shape[0]
    if m < TB: x = jnp.concatenate([x, jnp.repeat(x[:1], TB - m)])
    l_, s_ = net_v(th, x); LA[i:i + m] = np.asarray(l_)[:m]; SG[i:i + m] = np.asarray(s_)[:m].astype(np.int8)
    if ABORT and i + TB >= ABORT:
        np.savez(PART + '.tmp.npz', la=LA, sg=SG, i0=i + TB, th_sha=res['params_sha256']); os.replace(PART + '.tmp.npz', PART); progress(f'dry abort at {i + TB}'); sys.exit(9)
    if time.time() - tl > float(SPEC.get('ckpt_sec', 1200)):
        tl = time.time(); np.savez(PART + '.tmp.npz', la=LA, sg=SG, i0=i + TB, th_sha=res['params_sha256']); os.replace(PART + '.tmp.npz', PART)
        progress(f'  table {min(i + TB, D)}/{D}  {tl - t0:.0f}s')
res['table_sec'] = time.time() - t0; sec.reload(); res['dry_orbits'] = DRY_N
np.savez(os.path.join(OUT, 'table_b0.npz'), lP=LA, sP=SG)
if os.path.exists(PART): os.remove(PART)
progress(f'table done {res["table_sec"]:.0f}s ({((D if not DRY_N else DRY_N) - i0) / max(res["table_sec"], 1e-9):.0f} orbits/s this run)'); dump(F, res)
if DRY_N:
    SG[:DRY_N] = SG[:DRY_N] * (1 if np.mean(SG[:DRY_N] == np.asarray(sP)[:DRY_N].astype(np.int8)) > 0.5 else -1)
    LA[:DRY_N] -= np.mean(LA[:DRY_N] - np.asarray(lpp)[:DRY_N])     # plumbing test: align the log offset of the spliced part
la_b = jnp.asarray(LA); sN = jnp.asarray(SG.astype(np.float32)); del LA, SG
if float(sN[0]) < 0: sN = -sN                                  # global sign convention (as rbmpp_table.py)
res['table_check'] = dict(finite=bool(jnp.all(jnp.isfinite(la_b))), n_zero_sign=int(jnp.sum(sN == 0)))

# ================================================================================================ readout
res['b0_s2'] = readout(la_b, s2, 'B0 s2'); dump(F, res)
res['b0_own'] = readout(la_b, sN, 'B0 own'); dump(F, res)
res['sec'] = time.time() - T00; res['done'] = True; dump(F, res)
progress(f'DONE {res["sec"]:.0f}s  E_FN(s2): zero {res["b0_s2"]["EFN_zero_hop"]:.4e} one hop {res["b0_s2"]["EFN_one_hop"]:.4e}; '
         f'own sign: zero {res["b0_own"]["EFN_zero_hop"]:.4e} one hop {res["b0_own"]["EFN_one_hop"]:.4e}')
