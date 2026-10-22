"""Tests a, b, c of Test B0 for rbmpp_bf_lib.py (README compress_6x6, "Test B0").  python rbmpp_bf_test.py [OUT.json]
 a. u = 0: log psi equals rbmpp_lib's PP log psi (200 random configs, <= 1e-12) and the local energy equals rbmpp_lib's fast
    eloc (<= 1e-10).
 b. log|psi| equals the cx_t1s 'ppbf' network (code copied VERBATIM below, REF_*) at o2b_17431886/round_1/theta.npy on 200 configs
    (<= 1e-12); deliberately broken reference variants must FAIL the same comparison.
 c. local energy with nonzero u equals <x|H|psi>/psi(x) by explicit neighbour enumeration with the sector code's bond list
    (st6_sector BI, BJ, JB), psi(y) from the verbatim cx_t1s reference.
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'compress_6x6'))
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import rbmpp_lib as RL
import rbmpp_bf_lib as BF
from cx_lib import NBR as CX_NBR
import st6_sector as SS

PP_PARAMS = '/project/theorie/a/A.Otaifi/chatty_rbmpp6/runs/rpT1_pp_17163493/params_final.npz'
THETA_R1 = '/project/theorie/a/A.Otaifi/chatty_compress6/runs/o2b_17431886/round_1/theta.npy'
N = 36; NH = 18
OUT = {}


def rep(name, val, tol, ok=None):
    ok = (val <= tol) if ok is None else ok
    OUT[name] = dict(value=float(val), tol=tol, ok=bool(ok)); print(f'{name}: {val:.3e} (tol {tol:g}) {"PASS" if ok else "FAIL"}', flush=True)
    return ok


lat = BF.Lattice(6, 0.5, 'full')
z = np.load(PP_PARAMS); f0 = np.asarray(z['f'], np.float64)
S200 = np.asarray(BF.random_configs(N, 200, 7), np.int8)
fb = BF.make_bf_fns(lat, jnp.float64)
lp_bf = jax.jit(jax.vmap(fb['logpsi'], in_axes=(None, 0)))
sg_bf = jax.jit(jax.vmap(fb['sign'], in_axes=(None, 0)))
el_bf = jax.jit(jax.vmap(fb['eloc'], in_axes=(None, 0)))

# ------------------------------------------------------------------------------------------------ test a
th0 = jnp.asarray(BF.theta_from_f(f0))
fr = RL.make_fns(lat, jnp.float64, use_rbm=False)
pr = dict(f=jnp.asarray(f0), bre=jnp.zeros(16), bim=jnp.zeros(16), Wre=jnp.zeros((N, 16)), Wim=jnp.zeros((N, 16)))
lr_ = jax.jit(jax.vmap(fr['logpsi'], in_axes=(None, 0)))(pr, jnp.asarray(S200))
er_, lPr = jax.jit(jax.vmap(fr['eloc'], in_axes=(None, 0)))(pr, jnp.asarray(S200))
lb_ = lp_bf(th0, jnp.asarray(S200)); sb_ = sg_bf(th0, jnp.asarray(S200))
d1 = float(jnp.max(jnp.abs(jnp.real(lr_) - lb_)))
sgn_r = jnp.where(jnp.cos(jnp.imag(lr_)) >= 0, 1.0, -1.0)
d1s = float(jnp.max(jnp.abs(sgn_r - sb_)))
rep('a1_logpsi_max_abs_diff_vs_rbmpp_lib', d1, 1e-12)
rep('a1_sign_mismatches_vs_rbmpp_lib', d1s, 0.0)
eb_, _ = el_bf(th0, jnp.asarray(S200))
d2 = float(jnp.max(jnp.abs(jnp.real(er_) - eb_))); rep('a2_eloc_max_abs_diff_vs_rbmpp_lib_fast', d2, 1e-10)
ebr = jax.jit(jax.vmap(lambda p, s: fr['eloc_brute'](p, s)[0], in_axes=(None, 0)))(pr, jnp.asarray(S200[:40]))
rep('a2b_rbmpp_lib_FAST_eloc_vs_rbmpp_lib_BRUTE (40 configs; isolates the error of the fast update itself)', float(jnp.max(jnp.abs(er_[:40] - ebr))), 1e-10)
rep('a2c_rbmpp_bf_eloc_vs_rbmpp_lib_BRUTE (40 configs)', float(jnp.max(jnp.abs(eb_[:40] - jnp.real(ebr)))), 1e-10)
rep('a2_eloc_imag_rbmpp_lib_max', float(jnp.max(jnp.abs(jnp.imag(er_)))), 1e-10)
ebk, _ = jax.jit(jax.vmap(lambda t, s: fb['eloc_bonds'](t, s, 144), in_axes=(None, 0)))(th0, jnp.asarray(S200[:20]))
rep('a3_eloc_bonds(kmax=144=NB)_vs_eloc', float(jnp.max(jnp.abs(ebk - eb_[:20]))), 1e-12)
# the compressed-bond variant with kmax = max number of anti-aligned bonds in the sample
Sx = S200[:20].astype(np.float64); nanti = int(np.max(np.sum(Sx[:, lat.BI] != Sx[:, lat.BJ], 1)))
ebk2, _ = jax.jit(jax.vmap(lambda t, s: fb['eloc_bonds'](t, s, nanti), in_axes=(None, 0)))(th0, jnp.asarray(S200[:20]))
rep(f'a4_eloc_bonds(kmax={nanti})_vs_eloc', float(jnp.max(jnp.abs(ebk2 - eb_[:20]))), 1e-12)
rep('a5_NBR_equals_cx_lib', float(np.sum(BF.NBR != np.asarray(CX_NBR))), 0.0)

# ------------------------------------------------------------------------------------------------ verbatim cx_t1s.py (ppbf branch)
FLP = lat.FL.astype(jnp.float64); IMG = lat.IMG
_xy = np.array([(i % 6, i // 6) for i in range(N)])
DISP = jnp.asarray(np.array([[((_xy[j, 0] - _xy[i, 0]) % 6) + 6 * ((_xy[j, 1] - _xy[i, 1]) % 6) for j in range(N)] for i in range(N)], np.int32))
NBR = CX_NBR


def make_ref(BF_=True, brk=None):
    def log_one(f, u1, u2, zz):                       # --- cx_t1s.py, lines 154-167 (verbatim body, `brk` hooks for the break test)
        order = jnp.argsort(-zz, stable=True); Uo = order[:NH]; Do = order[NH:]
        Fm = f[Uo][:, Do]
        if BF_:
            zn = zz[NBR]
            M1 = zn[Uo] @ u1
            M2 = zn[Do] @ u2
            if brk == 'u2_displacement_swapped':
                Fm = Fm + jnp.take_along_axis(M1, DISP[Uo][:, Do], 1) + jnp.take_along_axis(M2, DISP[Uo][:, Do].T, 1).T
            elif brk == 'u2_sign':
                Fm = Fm + jnp.take_along_axis(M1, DISP[Uo][:, Do], 1) - jnp.take_along_axis(M2, DISP[Do][:, Uo], 1).T
            elif brk == 'u1_window_shift':
                Fm = Fm + jnp.take_along_axis(M1, DISP[Uo][:, Do], 1) * 1.0000001 + jnp.take_along_axis(M2, DISP[Do][:, Uo], 1).T
            else:
                Fm = Fm + jnp.take_along_axis(M1, DISP[Uo][:, Do], 1) + jnp.take_along_axis(M2, DISP[Do][:, Uo], 1).T
        sg, ld = jnp.linalg.slogdet(Fm)
        up = (zz > 0).astype(jnp.int32); dn = 1 - up
        inv = jnp.sum(up * (jnp.cumsum(dn) - dn))
        par = (jnp.where(sg < 0, 1, 0) + inv) % 2
        return ld, 1.0 - 2.0 * par

    def logA_one(th, sigma):                          # --- cx_t1s.py, lines 169-179
        f = th[:N * N].reshape(N, N)
        if BF_:
            u = th[N * N:].reshape(2, 9, N); u1, u2 = u[0], u[1]
        else:
            u1 = u2 = None
        Z = FLP[:, None] * sigma.astype(jnp.float64)[IMG]
        ld, sgn = jax.vmap(lambda zz: log_one(f, u1, u2, zz))(Z)
        m = jax.lax.stop_gradient(jnp.max(ld))
        return m + jnp.log(jnp.maximum(jnp.abs(jnp.sum(sgn * jnp.exp(ld - m))), 1e-300)), jnp.sum(sgn * jnp.exp(ld - m))
    return logA_one


# ------------------------------------------------------------------------------------------------ test b
thR = jnp.asarray(np.load(THETA_R1)); assert thR.size == BF.NTH, thR.size
ref = make_ref()
refv = jax.jit(jax.vmap(lambda s: ref(thR, s), in_axes=0))
lref, vref = refv(jnp.asarray(S200))
lmy = lp_bf(thR, jnp.asarray(S200))
smy = sg_bf(thR, jnp.asarray(S200))
rep('b1_logabspsi_vs_cx_t1s_ppbf_round1_theta', float(jnp.max(jnp.abs(lref - lmy))), 1e-12)
rep('b1_sign_mismatches', float(jnp.sum(jnp.where(vref >= 0, 1.0, -1.0) != smy)), 0.0)
rep('b0_backflow_is_nonzero_here(max|u|, so the test is not vacuous)', float(jnp.max(jnp.abs(thR[N * N:]))), 0.0, ok=bool(jnp.max(jnp.abs(thR[N * N:])) > 1e-3))
f_only = jnp.concatenate([thR[:N * N], jnp.zeros(2 * 9 * N)])
rep('b0_effect_of_backflow_on_logpsi(max |log psi(theta) - log psi(f only)|)', float(jnp.max(jnp.abs(lmy - lp_bf(f_only, jnp.asarray(S200))))), 0.0,
    ok=bool(jnp.max(jnp.abs(lmy - lp_bf(f_only, jnp.asarray(S200)))) > 1e-6))
for brk in ('u2_displacement_swapped', 'u2_sign', 'u1_window_shift'):
    rb = make_ref(brk=brk); lb2, _ = jax.jit(jax.vmap(lambda s: rb(thR, s), in_axes=0))(jnp.asarray(S200))
    dd = float(jnp.max(jnp.abs(lb2 - lmy)))
    rep(f'b2_BROKEN_{brk}: diff to rbmpp_bf_lib (must exceed 1e-12 => comparison fails => test discriminates)', dd, 1e-12, ok=bool(dd > 1e-12))
# a second theta with random backflow (larger than round_1's) as an extra check
rng = np.random.default_rng(3)
thX = jnp.concatenate([thR[:N * N], jnp.asarray(0.1 * rng.normal(size=2 * 9 * N))])
lX = jax.vmap(lambda s: ref(thX, s)[0])(jnp.asarray(S200))
rep('b3_random_u_logabspsi_vs_cx_t1s_ref', float(jnp.max(jnp.abs(lX - lp_bf(thX, jnp.asarray(S200))))), 1e-12)

# ------------------------------------------------------------------------------------------------ test c
th_c = jnp.concatenate([jnp.asarray(f0).reshape(-1), jnp.asarray(0.05 * rng.normal(size=2 * 9 * N))])
Sc = np.asarray(BF.random_configs(N, 6, 11), np.int8)
ec, _ = el_bf(th_c, jnp.asarray(Sc))
ec_b = np.asarray(jax.jit(jax.vmap(lambda t, s: fb['eloc_bonds'](t, s, 144)[0], in_axes=(None, 0)))(th_c, jnp.asarray(Sc)))
BI_, BJ_, JB_ = np.asarray(SS.BI), np.asarray(SS.BJ), np.asarray(SS.JB)
dmax = 0.0
for k in range(Sc.shape[0]):
    x = Sc[k].astype(np.float64)
    lx, vx = ref(th_c, jnp.asarray(x))      # ref returns (log|psi|, signed scaled sum): psi = sign * exp(log|psi|)
    sx = np.sign(float(vx)); Ex = 0.0
    for b in range(len(BI_)):
        i, j = BI_[b], BJ_[b]
        Ex += 0.25 * JB_[b] * x[i] * x[j]
        if x[i] != x[j]:
            y = x.copy(); y[i], y[j] = x[j], x[i]
            ly, vy = ref(th_c, jnp.asarray(y))
            Ex += 0.5 * JB_[b] * np.sign(float(vy)) * sx * np.exp(float(ly) - float(lx))
    dmax = max(dmax, abs(Ex - float(ec[k])), abs(Ex - float(ec_b[k])))
    print(f'  c: config {k}: enumeration {Ex:.12f}  rbmpp_bf eloc {float(ec[k]):.12f}  eloc_bonds {ec_b[k]:.12f}', flush=True)
rep('c1_eloc_nonzero_u_vs_explicit_enumeration(sector-code bond list)', dmax, 1e-10)
rep('c0_sector_bond_list_matches_Lattice(NB, sum J)', abs(len(BI_) - lat.NB) + abs(JB_.sum() - lat.JB.sum()), 0.0)
OUT['all_pass'] = all(v['ok'] for v in OUT.values() if isinstance(v, dict))
print('ALL PASS' if OUT['all_pass'] else 'SOME FAIL')
if len(sys.argv) > 1: json.dump(OUT, open(sys.argv[1], 'w'), indent=1)
