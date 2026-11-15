"""T-BAD10 round 1, stage LABELS: K[Marshall] on tempered root packets at 10x10 (the 8x8 layout: chains x nsamp samples, held-out iff chain id % 5 == 0).
Roots: Metropolis chains at p ~ |ViT|^(2 beta), beta = 1/2 (single antiparallel-bond swap proposal, Hastings n_a(x)/n_a(y)), starts = distinct shipped spins (|ViT|^2 draws), burn_sweeps sweeps
(1 sweep = 100 proposals), nsamp samples thinned by thin_sweeps sweeps.  Per root: complete neighbourhood L1 = {x} + all valid one-swap neighbours (bond order), the TWO-HOP |ViT| set (deduplicated),
r = |psi(z_b)/psi(z)| for every antiparallel bond of every z in L1 (float32, stored: later rounds label any sign function from it plus n on the two-hop set),
  E_L(|ViT| n; z) = H_zz + n(z) sum_b (J_b/2) n(z_b) r_b ;  K[n](z) = n(z) sgn[1 + alpha (E_L - E)],  stored relative to Marshall: y = sgn[1 + alpha (E_L(|ViT| M; z) - E)]  (K[M] = M y).
alpha, E are READ from the sha'd consts json (--consts, --consts_sha) and never recomputed.
Controls (gate, after the first chunk and again at the end; exit 3 on failure): C1 20 states vs direct recomputation (plain-python Marshall, own ViT call on z and its neighbours) + a deliberately corrupted
copy must be detected; C2 wrong-alpha arms (1.5, 0.5, 0) change labels; C3 'Marshall in place': the generic path (sign function evaluated on the explicit two-hop states, Net10 marshall-mode) reproduces y on all states of 4 roots.
usage: python labels10.py OUT --consts C.json --consts_sha C.json.sha256 --chains 1000 --nsamp 4 --seed 11"""
import sys, os, time, json, argparse, hashlib, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--consts', required=True); ap.add_argument('--consts_sha', required=True)
ap.add_argument('--chains', type=int, default=1000); ap.add_argument('--nsamp', type=int, default=4); ap.add_argument('--burn_sweeps', type=int, default=40); ap.add_argument('--thin_sweeps', type=int, default=2)
ap.add_argument('--beta', type=float, default=0.5); ap.add_argument('--seed', type=int, default=11); ap.add_argument('--chunk', type=int, default=128); ap.add_argument('--maxroots', type=int, default=0)
ap.add_argument('--controls_only', action='store_true')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
P = lambda *s: print(time.strftime('%H:%M:%S'), 'PROGRESS', *s, flush=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
import vitlib as VL, vmc10 as VM
from vitlib import BI, BJ, HW, JB, N, NB
cs = sha(a.consts); want = open(a.consts_sha).read().split()[0]
if cs != want: sys.exit(f'LABELS: consts sha256 mismatch {cs} != {want}')
C0 = json.load(open(a.consts)); ALPHA = float(C0['alpha']); EREF = float(C0['E']); P('consts', a.consts, cs, 'alpha', ALPHA, 'E', EREF)
ev = VL.make_vit('f32h'); S = VL.shipped(); CH = 1024
def pack(Z): return np.packbits(np.asarray(Z) > 0, axis=1)
def unpack(B): return np.where(np.unpackbits(B, axis=1)[:, :N] > 0, 1.0, -1.0).astype(np.float32)
def lab(ELM, alpha=ALPHA, E=EREF):
    f = 1.0 + alpha * (ELM - E); return np.where(f >= 0, 1, -1).astype(np.int8), f

# ---------------------------------------------------------------------------------------------- sampler
fs = a.out + '/samples.npz'
if os.path.exists(fs):
    d = np.load(fs); roots = d['X']; chain = d['chain']; P('samples reloaded', len(roots))
elif not a.controls_only:
    rng = np.random.default_rng(a.seed); C = a.chains; assert C <= len(S)
    X = S[rng.choice(len(S), C, replace=False)].copy(); lp = ev(X, CH).real; rec = []; lpm = []; t0 = time.time(); acc = 0.0
    times = [a.burn_sweeps * N + k * a.thin_sweeps * N for k in range(1, a.nsamp + 1)]; step = 0; ar = np.arange(C)
    while step < times[-1]:
        anti = X[:, BI] != X[:, BJ]; na = anti.sum(1); b = np.argmax(rng.random(anti.shape) * anti, 1)
        Y = X.copy(); Y[ar, BI[b]], Y[ar, BJ[b]] = X[ar, BJ[b]], X[ar, BI[b]]
        lq = ev(Y, CH).real; nay = (Y[:, BI] != Y[:, BJ]).sum(1)
        ok = np.log(rng.random(C)) < 2 * a.beta * (lq - lp) + np.log(na / nay); X[ok] = Y[ok]; lp = np.where(ok, lq, lp); acc += ok.mean(); step += 1
        if step in times: rec.append(X.copy()); lpm.append(float(lp.mean())); P('sample time', step, 'mean log|psi|', lpm[-1], 'acc', acc / step)
        if step % 200 == 0: P('sampler step', step, '/', times[-1], 'sec', f'{time.time() - t0:.0f}')
    roots = np.concatenate(rec).astype(np.int8); chain = np.tile(np.arange(C), a.nsamp)
    np.savez(fs, X=roots, chain=chain, times=np.array(times), mean_lp=np.array(lpm), acc=acc / step)
    json.dump(dict(mean_lp_by_sample=lpm, acc=acc / step, sec=time.time() - t0, args=vars(a), mref10=float(np.median(2 * ev(S[:2048], CH).real))), open(a.out + '/sampler_meta.json', 'w'))
    P('sampler done', len(roots), 'roots; mean log|psi| per sample time', lpm)
if a.maxroots: roots = roots[:a.maxroots]; chain = chain[:a.maxroots]

# ---------------------------------------------------------------------------------------------- one root
def label_root(x):
    """returns dict for root x (100,): L1 (n1,100), lp1, r (pairs), ELM (n1,), f, y, plus the explicit two-hop objects for controls (optional)."""
    x = np.asarray(x, np.float32)[None]; wi1, bi1, Y1 = VL.neighbours(x); L1 = np.concatenate([x, Y1]); n1 = len(L1)
    wi2, bi2, Z = VL.neighbours(L1); allS = np.concatenate([L1, Z]); k = VL.keys_of(allS); _, first, inv = np.unique(k, return_index=True, return_inverse=True)
    lpU = ev(allS[first], CH).real; lp = lpU[inv]; lp1 = lp[:n1]; r = np.exp(lp[n1:] - lp1[wi2])
    M = VL.marshall(allS).astype(float); M1 = M[:n1]; MZ = M[n1:]
    ELM = VL.diag0(L1) + M1 * np.bincount(wi2, weights=HW[bi2] * MZ * r, minlength=n1)
    y, f = lab(ELM); return dict(L1=L1, lp1=lp1, r=r, wi2=wi2, bi2=bi2, Z=Z, ELM=ELM, f=f, y=y)

def chunk_file(k): return f'{a.out}/chunk_{k:04d}.npz'
def load_chunks():
    out = []
    for f in sorted(glob.glob(a.out + '/chunk_*.npz')): out.append(dict(np.load(f)))
    return out

# ---------------------------------------------------------------------------------------------- controls
def controls(nroots_generic=4, n_direct=20, tol=2e-4):
    ch = load_chunks(); assert ch, 'no chunks'; res = {}
    L1all = np.concatenate([unpack(c['L1b']) for c in ch]); ELall = np.concatenate([c['ELM'] for c in ch]); yall = np.concatenate([c['y'] for c in ch]); fall = np.concatenate([c['f'] for c in ch])
    rng = np.random.default_rng(7); pick = rng.choice(len(L1all), n_direct, replace=False); dev = []; mism = 0; edge = 0; rows = []
    def Mpy(z): return -1 if sum(1 for i in range(N) if z[i] > 0 and ((i % 10) + (i // 10)) % 2 == 0) % 2 else 1
    for i in pick:
        z = L1all[i].astype(np.float64); nb = [b for b in range(NB) if z[BI[b]] != z[BJ[b]]]; Zs = [z]
        for b in nb:
            y_ = z.copy(); y_[BI[b]], y_[BJ[b]] = z[BJ[b]], z[BI[b]]; Zs.append(y_)
        lz = ev(np.array(Zs, np.float32), 256).real; Mz = Mpy(z); EL = sum(0.25 * JB[b] * z[BI[b]] * z[BJ[b]] for b in range(NB))
        for j, b in enumerate(nb): EL += 0.5 * JB[b] * Mz * Mpy(Zs[j + 1]) * np.exp(lz[j + 1] - lz[0])
        f = 1 + ALPHA * (EL - EREF); yd = 1 if f >= 0 else -1; dev.append(abs(EL - ELall[i])); rows.append((len(nb), float(EL), float(ELall[i])))
        if abs(f) < 1e-6: edge += 1
        elif yd != int(yall[i]): mism += 1
    res['C1'] = dict(n=n_direct, max_abs_dEL=float(max(dev)), sign_mismatches=mism, edge_states=edge, n_neighbours=[r_[0] for r_ in rows])
    # corrupted copy: one stored ratio of the first root of chunk 0 doubled -> recomputed E_L must differ by > tol (detection works)
    c0 = ch[0]; n1_0 = int(c0['n1'][0]); L0 = unpack(c0['L1b'][:n1_0]); z = L0[0].astype(np.float64); nb = [b for b in range(NB) if z[BI[b]] != z[BJ[b]]]
    rr = c0['r'][:len(nb)].astype(np.float64).copy(); Mz = Mpy(z); Mn = np.array([Mpy(np.where(np.arange(N) == BI[b], z[BJ[b]], np.where(np.arange(N) == BJ[b], z[BI[b]], z))) for b in nb])
    el_ok = float(sum(0.25 * JB[b] * z[BI[b]] * z[BJ[b]] for b in range(NB)) + Mz * np.sum(HW[nb] * Mn * rr)); rr[0] *= 2.0
    el_bad = float(sum(0.25 * JB[b] * z[BI[b]] * z[BJ[b]] for b in range(NB)) + Mz * np.sum(HW[nb] * Mn * rr))
    res['C1_broken'] = dict(EL_from_stored=el_ok, EL_stored=float(c0['ELM'][0]), EL_corrupted=el_bad, detected=bool(abs(el_bad - c0['ELM'][0]) > tol and abs(el_ok - c0['ELM'][0]) < tol))
    ch2 = {}
    for nm, al in (('1.5alpha', 1.5 * ALPHA), ('0.5alpha', 0.5 * ALPHA), ('alpha0', 0.0)):
        l2, _ = lab(ELall, alpha=al); ch2[nm] = float((l2 != yall).mean())
    res['C2_wrong_alpha_label_fraction_changed'] = ch2; res['minus_share'] = float((yall < 0).mean()); res['states'] = int(len(yall))
    # C3: Marshall in place, generic path: n evaluated on the explicit two-hop states through Net10 (marshall mode: n == Marshall by construction, every layer still computed)
    import s10_net as SN; net = SN.Net10(64, 10, seed=0, mode='marshall'); mism3 = 0; ns3 = 0; maxdE = 0.0; cnt = 0
    for c in ch:
        off = 0; po = 0; ro = 0
        for ri in range(len(c['n1'])):
            n1 = int(c['n1'][ri]); L1 = unpack(c['L1b'][off:off + n1]); lp1 = c['lp1'][off:off + n1]
            if cnt < nroots_generic:
                wi2, bi2, Z = VL.neighbours(L1); npairs = len(wi2); rr = c['r'][po:po + npairs]
                nz = net.sign(L1).astype(float); nZ = net.sign(Z).astype(float)
                EL = VL.diag0(L1) + nz * np.bincount(wi2, weights=HW[bi2] * nZ * rr, minlength=n1); yy, _ = lab(EL)
                mism3 += int((yy != c['y'][off:off + n1]).sum()); ns3 += n1; maxdE = max(maxdE, float(np.abs(EL - c['ELM'][off:off + n1]).max())); cnt += 1
            po += int(VL_npairs(L1)); off += n1
            if cnt >= nroots_generic: break
        if cnt >= nroots_generic: break
    res['C3_marshall_in_place'] = dict(roots=cnt, states=ns3, label_mismatches=mism3, max_abs_dEL=maxdE)
    ok = res['C1']['max_abs_dEL'] < tol and mism == 0 and res['C1_broken']['detected'] and all(v > 0.005 for v in ch2.values()) and mism3 == 0 and maxdE < tol
    res['verdict'] = 'PASS' if ok else 'FAIL'; print(json.dumps(res, indent=1), flush=True); P('R1_LABEL_CONTROLS', res['verdict']); json.dump(res, open(a.out + '/controls.json', 'w'), indent=1)
    return ok
def VL_npairs(L1): return int((L1[:, BI] != L1[:, BJ]).sum())

if a.controls_only:
    sys.exit(0 if controls() else 3)
# ---------------------------------------------------------------------------------------------- label chunks
nroots = len(roots); t0 = time.time(); done_states = 0; ev0 = 0
for k, i in enumerate(range(0, nroots, a.chunk)):
    fk = chunk_file(k)
    if os.path.exists(fk): continue
    L1b = []; lp1 = []; rr = []; ELM = []; ff = []; yy = []; n1s = []
    for x in roots[i:i + a.chunk]:
        d = label_root(x); L1b.append(pack(d['L1'])); lp1.append(d['lp1']); rr.append(d['r'].astype(np.float32)); ELM.append(d['ELM']); ff.append(d['f']); yy.append(d['y']); n1s.append(len(d['L1']))
    sl = slice(i, i + a.chunk)
    np.savez(fk + '.tmp.npz', roots_b=pack(roots[sl]), chain=chain[sl], n1=np.array(n1s), L1b=np.concatenate(L1b), lp1=np.concatenate(lp1), r=np.concatenate(rr), ELM=np.concatenate(ELM), f=np.concatenate(ff), y=np.concatenate(yy)); os.replace(fk + '.tmp.npz', fk)
    done_states += sum(n1s); el = time.time() - t0; P(f'chunk {k} roots {min(i + a.chunk, nroots)}/{nroots} states {sum(n1s)} {el:.0f}s ({el / min(i + a.chunk, nroots) * 1.0:.2f} s/root cumulative)')
    if k == 0 and not os.path.exists(a.out + '/controls_chunk0.ok'):
        if not controls(): P('R1_LABELS_ABORT controls failed after chunk 0'); sys.exit(3)
        open(a.out + '/controls_chunk0.ok', 'w').write('ok\n')
ok = controls()
json.dump(dict(roots=int(nroots), consts_sha256=cs, alpha=ALPHA, E=EREF, sec=time.time() - t0, controls=('PASS' if ok else 'FAIL'), args=vars(a)), open(a.out + '/labels_done.json', 'w'), indent=1)
P('R1_LABELS_DONE', 'roots', nroots, 'controls', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 3)
