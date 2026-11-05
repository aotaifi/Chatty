"""preflight8.py: CPU checks of the 8x8 adapter (gfmc_aug8.py) on tiny M.  usage: python preflight8.py OUT.json
 A1  c0 provider (AugGuide8 fast0) == AugGuide(mode 'old', c = 0) on the OLD path == brute force (explicit swaps, logB, PhaseSign direct sign); broken: marshall-less sign
 A2  AugGuide8 c = 4 'fix' dedup == base AugGuide (old path) ratios; broken: c = 3.9
 A3  DirectSG ratios == brute force with decades8.make_sG8 signs; its sign == M1's own s_G (decades8.m1_sign) on every config; broken: alpha = 0
 A4  PhaseSign guard counters, dmax floor and shapes of AugGuide8
"""
import sys, json, time, numpy as np
sys.path.insert(0, '/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code')
import gfmc_aug8 as G8, gfmc_aug as GA, pp2_lemma as PL, decades8 as D8
OUT = sys.argv[1]; I = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8build/h100bench/inputs'; TH = '/scratch/a/A.Otaifi/ansatz_runs/sign_scale/r8/ppw_s1/theta.npy'
res = {}; T0 = time.time()
def log(m): print(f'[{time.time() - T0:6.0f}s] {m}', flush=True)
X = np.load(f'{I}/X0_a_M16.npy')[:6]; p = PL.PP2.from_rbmpp_npz(G8.PROD8); lat = p.lat; th = np.load(TH)
ps = PL.PhaseSign.from_theta(th, 8); psb = PL.PhaseSign.from_theta(th, 8, broken='marshall'); EP = G8.EP8; f0 = G8.F0_8
def rel(a, b): a = np.asarray(a); b = np.asarray(b); m = (b != 0) | (a != 0); return float(np.max(np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-300))) if m.any() else 0.0
def run(g, X): return np.asarray(g.rows(X.copy()) [0])      # rate (B, NB)
# ---------------- brute force ratios G(y_b)/G(x) = s(x) s(y) |B(y)/B(x)| from explicit swapped configurations
def brute(X, sign_fn):
    B = X.shape[0]; out = np.zeros((B, lat.NB)); lbx = np.real(np.asarray(PL.logB(p, X.astype(float))))
    for w in range(B):
        for b in range(lat.NB):
            i, j = lat.BI[b], lat.BJ[b]
            if X[w, i] == X[w, j]: continue
            y = X[w].copy(); y[i], y[j] = y[j], y[i]
            out[w, b] = sign_fn(X[w:w + 1].astype(float))[0] * sign_fn(y[None].astype(float))[0] * np.exp(np.real(np.asarray(PL.logB(p, y[None].astype(float)))[0]) - lbx[w])
    return out
def ratios_of(g, X):
    """AugGuide-style providers: ratio (B, NB) via _aug"""
    Xf = X.astype(np.float64); wc = min(g.wchunk, len(X)) if hasattr(g, 'wchunk') else len(X)
    return np.asarray(g._aug(Xf)['ratio'])
t = time.time()
g0 = G8.AugGuide8(p, ps, EP, 0.0, f0, mode='old', eval_path='dedup', wchunk=8, chunk=8, fast0=True)
gold = GA.AugGuide(p, ps, EP, 0.0, f0, mode='old', chunk=8); gold.dmax = 168                 # base class, old path, as in c3 (rows() sets dmax; _aug is called directly here)
r_fast = ratios_of(g0, X); r_old = np.asarray(gold._aug(X.astype(float))['ratio']); r_bf = brute(X, ps)
res['A1'] = dict(fast_vs_oldaug=rel(r_fast, r_old), fast_vs_brute=rel(r_fast, r_bf), oldaug_vs_brute=rel(r_old, r_bf), nonzero=int((r_bf != 0).sum()), sec=time.time() - t)
gb = G8.AugGuide8(p, psb, EP, 0.0, f0, mode='old', eval_path='dedup', wchunk=8, chunk=8, fast0=True)
res['A1']['broken_marshall_vs_brute'] = rel(ratios_of(gb, X), r_bf)
res['A1']['PASS'] = bool(res['A1']['fast_vs_oldaug'] <= 1e-12 and res['A1']['fast_vs_brute'] <= 1e-9 and res['A1']['broken_marshall_vs_brute'] > 1e-3)
log(f'A1 {res["A1"]}')
# ---------------- A2
t = time.time()
g4 = G8.AugGuide8(p, ps, EP, 4.0, f0, mode='fix', eval_path='dedup', wchunk=8, chunk=8, dmax0=168)
b4 = GA.AugGuide(p, ps, EP, 4.0, f0, mode='fix', chunk=8); b4.dmax = 168
r_d = ratios_of(g4, X); r_b = np.asarray(b4._aug(X.astype(float))['ratio'])
g39 = G8.AugGuide8(p, ps, EP, 3.9, f0, mode='fix', eval_path='dedup', wchunk=8, chunk=8, dmax0=168)
res['A2'] = dict(dedup_vs_old=rel(r_d, r_b), broken_c39_vs_old=rel(ratios_of(g39, X), r_b), sec=time.time() - t)
res['A2']['PASS'] = bool(res['A2']['dedup_vs_old'] <= 1e-12 and res['A2']['broken_c39_vs_old'] > 1e-4)
log(f'A2 {res["A2"]}')
# ---------------- A3
t = time.time(); al, E, w = G8.m1_consts()
sg = G8.DirectSG(p, al, E, w, chunk=16)
Xs = np.load(f'{I}/X0_a_M16.npy'); Ys = []
for x in Xs[:4]:
    for b in range(lat.NB):
        i, j = lat.BI[b], lat.BJ[b]
        if x[i] != x[j]:
            y = x.copy(); y[i], y[j] = y[j], y[i]; Ys.append(y)
Ys = np.array(Ys[:80]); Zall = np.concatenate([Xs, Ys]).astype(float)
s_mine = sg.sign_at(Zall); s_own = D8.make_sG8(p, al, E, w)(Zall); s_m1 = D8.m1_sign(Zall, al, E, w)[0]
sg0 = G8.DirectSG(p, 0.0, E, w, chunk=16); s_brk = sg0.sign_at(Zall)
res['A3'] = dict(n=int(len(Zall)), mine_vs_m1_disagree=int(np.sum(s_mine != s_m1)), own_vs_m1_disagree=int(np.sum(s_own != s_m1)), broken_alpha0_vs_m1_disagree=int(np.sum(s_brk != s_m1)), frac_minus=float(np.mean(s_m1 < 0)))
Xt = Xs[:3]; rt = np.asarray(sg.rows(Xt.copy())[0])                                        # rate = -K r for kept; compare the G ratios through finish
brute_sg = brute(Xt, lambda z: D8.make_sG8(p, al, E, w)(np.asarray(z, float)))
# rebuild rate from brute ratios with the same finish rule
anti = Xt[:, lat.BI] != Xt[:, lat.BJ]; wi, bi = np.nonzero(anti); rate_b, d_b = G8.GuideBase.finish(sg, Xt, anti, wi, bi, brute_sg[wi, bi])
rate_m, d_m = sg.rows(Xt.copy())
res['A3'].update(rate_vs_brute=rel(rate_m, rate_b), dFN_vs_brute=rel(d_m, d_b), sec=time.time() - t)
res['A3']['PASS'] = bool(res['A3']['mine_vs_m1_disagree'] == 0 and res['A3']['broken_alpha0_vs_m1_disagree'] > 0 and res['A3']['rate_vs_brute'] <= 1e-9 and res['A3']['dFN_vs_brute'] <= 1e-9)
log(f'A3 {res["A3"]}')
# ---------------- A4
res['A4'] = dict(guard_checked=int(ps.n_checked), guard_fallback=int(ps.n_fallback), fallback_rate=float(ps.fallback_rate), shapes=sorted(map(list, g4.shapes)), dmax=int(g4.dmax), dmax_floor=168)
res['A4']['PASS'] = bool(ps.n_checked > 0 and ps.n_fallback == 0)
res['PASS_all'] = bool(all(res[k]['PASS'] for k in ('A1', 'A2', 'A3', 'A4')))
json.dump(res, open(OUT, 'w'), indent=1, default=float); log(f'PASS_all {res["PASS_all"]}')
