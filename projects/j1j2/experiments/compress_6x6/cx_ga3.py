"""Gate G-A3 (README compress_6x6 "M1 + gate for the loop"): fit-free two-hop chain from RBM+PP A with its own sign.  No training.
  A = sym_tables_A.npz of rpTabA_17168764 (rbmpp_table.py on rpT4_sr2_17164867/params_final.npz; Re(e^{-i theta} psi), c = [1, 0], im_weight 7.7e-27): (lP, sP).
  K_2(v) = lowest Ritz vector in span{v, Hv, H^2 v} (exact signed two-step polynomial: its coefficients minimise <H> of the signed state; = lanczos_ritz p = 2).
  g_0 = K_2(A);  phi_k = FN solution of g_k (Perron vector of H_FN[|g_k|, sign g_k]; sign of g_k kept);  g_{k+1} = K_2(sign(g_k) phi_k), sign = sign of the
  signed polynomial.  Per round: E_FN(g_k) (= <phi_k|H|phi_k>), <H>(g_k), variance, w_s of the sign vs the exact ground state (sec.score, s0), gain.
Controls (before the chain): A: dE 4.742047e-5, w_s 1.09482e-5, E_FN 3.524944e-5 (table_A.json); one hop p = 1 then FN: 5.573833e-6 (table_A.json lanczos p1 dE_FN);
  lanczos p = 2 energy 2.485107e-6.
  python cx_ga3.py OUT SPEC.json   SPEC: table_A (sym_tables_A.npz), rounds (6)
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
import st6_sector as SS
from lanczos_lib import lanczos_ritz
import json
jax.config.update('jax_enable_x64', True)
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
ROOT = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6'
CSR = os.environ.get('ST6_CSR', ROOT + '/csr'); TABLE = os.environ.get('ST6_TABLE', ROOT + '/data/psi0_6x6_table.npz')
N = 36; T00 = time.time(); F = os.path.join(OUT, 'cx_ga3.json'); res = dict(spec=SPEC, done=False)
def dump(): json.dump(res, open(F, 'w'), indent=1, default=float)
def progress(msg):
    SS.log(msg); open(os.path.join(OUT, 'progress.txt'), 'a').write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}\n')
sec = SS.Sector(CSR, TABLE); E0 = sec.E0
z = np.load(SPEC['table_A']); la = jnp.asarray(z['lP']); s = jnp.asarray(z['sP'].astype(np.float32)); del z
def la_s(psi):
    la_ = jnp.log(jnp.maximum(jnp.abs(psi) / sec.sqrt_n, 1e-300)); s_ = jnp.where(psi >= 0, 1.0, -1.0).astype(jnp.float32); return la_, s_
def Kp(v, p):
    rl = lanczos_ritz(lambda x: sec.Hm(x), v, p); return rl
def ctl(name, val, ref, tol):
    rel = abs(val - ref) / ref; ok = rel < tol; res.setdefault('control', {})[name] = dict(val=val, ref=ref, rel=rel, ok=bool(ok)); dump()
    progress(f'CONTROL {name}: {val:.7e} vs {ref:.7e} rel {rel:.2e} ok={ok}'); return ok
sc = sec.score(la, s); ok = ctl('A_dE', sc['dE_site'], 4.7420471760675457e-05, 1e-6) & ctl('A_ws', sc['w_s'], 1.0948228243157221e-05, 1e-3)
v = sec.vec(la, s)
_, u, info = sec.fn_solve(la, s); ok &= ctl('A_EFN', info['dE_FN_site'], 3.524943767886491e-05, 1e-6)
rl = Kp(v, 2)
ok &= ctl('lanczos_p1_dE', (rl[1]['E'] - E0) / N, 7.3797957444752014e-06, 1e-5); ok &= ctl('lanczos_p2_dE', (rl[2]['E'] - E0) / N, 2.4851069732415945e-06, 1e-5)
la1, s1 = la_s(rl[1]['psi']); _, _, i1 = sec.fn_solve(la1, s1); ok &= ctl('one_hop_EFN', i1['dE_FN_site'], 5.573833232593491e-06, 1e-6)
res['control']['ok'] = bool(ok); dump()
if not ok: res['stop'] = 'CONTROL FAILED'; res['done'] = True; dump(); progress('CONTROL FAILED: stop'); sys.exit(2)
rounds = int(SPEC.get('rounds', 6)); chain = []
psi = rl[2]['psi']; del rl, u
ref = v; s_prev = s
res['chain'] = chain
for k in range(rounds + 1):
    t0 = time.time(); psi = psi / jnp.linalg.norm(psi)
    if float(psi @ ref) < 0: psi = -psi                      # global sign aligned with the previous signed state
    la_k, s_k = la_s(psi)
    Hp = sec.H(psi); Ek = float(psi @ Hp); var = float(Hp @ Hp) - Ek ** 2; del Hp
    scr = sec.score(la_k, s_k)
    _, u, info = sec.fn_solve(la_k, s_k)
    rec = dict(k=k, dE_g=(Ek - E0) / N, var_site=var / N, EFN=info['dE_FN_site'], fn_res=info['res'], fn_it=info['lobpcg_it'], w_s=scr['w_s'], amp_infid_g=scr['amp_infid'])
    if chain: rec['gain_EFN'] = chain[-1]['EFN'] / rec['EFN']
    rec['sign_flip_weight_vs_prev'] = float(jnp.sum(jnp.where(s_k != s_prev, psi * psi, 0.0)))
    chain.append(rec); s_prev = s_k; dump()
    progress(f'g_{k}: dE(g) {rec["dE_g"]:.5e} E_FN(g) {rec["EFN"]:.6e} gain {rec.get("gain_EFN", float("nan")):.4f} w_s {rec["w_s"]:.5e} var {rec["var_site"]:.3e} ({time.time() - t0:.0f}s)')
    if k == rounds: break
    phi = u * s_k                                                  # signed FN state phi_k = FN amplitude with the sign of g_k
    ref = phi; rl = Kp(phi, 2); psi = rl[2]['psi']; del rl, phi, u
res['sec'] = time.time() - T00; res['done'] = True; dump()
progress('VERDICT DONE ' + ' '.join(f'g{r["k"]}:{r["EFN"]:.4e}' for r in chain) + ' gains ' + ' '.join(f'{r["gain_EFN"]:.3f}' for r in chain if 'gain_EFN' in r))
