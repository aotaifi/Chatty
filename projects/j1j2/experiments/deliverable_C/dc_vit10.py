"""Deliverable C, E4: the public 10x10 ViT (nqs-models/j1j2_square_10x10_05) in our JAX code (vit_dt.py = the same
nqsmagic ViT: Embed / EncoderBlock / FMHA / OuputHead), fp32 with full-precision matmuls.

psi(x) = sum_{rows p of symm_indexes} [psi_net(x[p]) + flip * psi_net(-x[p])]   (as the card's vitnqs_symm_ising_model.py:
log-sum-exp of the complex network output over the permutations, then log(e^f(s) + e^f(-s)) for spin inversion).
Revision main: no permutations (identity), no flip.

Modes:
  probe:  throughput with random parameters (no checkpoint needed): python dc_vit10.py OUT probe [--nperm 32 --flip]
  vmc:    reproduction gate (PREREG E4): python dc_vit10.py OUT vmc --ckpt model.safetensors [--symm symm_indexes.txt --flip]
          --nchains 1024 --burn 20 --nsamp 4 --thin 1
"""
import os, sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p_ in (HERE, os.path.join(HERE, '..', 'learned_loop_6x6')):
    if p_ not in sys.path: sys.path.append(p_)
import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp
import flax
import vit_dt
import dc_fn as F

L = 10; N = 100


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


def load_safetensors(path):
    """minimal safetensors reader (8-byte little-endian header length, JSON header, raw little-endian buffers)."""
    raw = open(path, 'rb').read(); n = int.from_bytes(raw[:8], 'little'); hdr = json.loads(raw[8:8 + n]); base = 8 + n
    dt = {'F64': np.float64, 'F32': np.float32, 'F16': np.float16, 'I64': np.int64, 'I32': np.int32}
    out = {}
    for k, v in hdr.items():
        if k == '__metadata__': continue
        a, b = v['data_offsets']
        out[k] = np.frombuffer(raw[base + a:base + b], dtype=dt[v['dtype']]).reshape(v['shape'])
    return out


def nest(flat):
    tree = {}
    for k, v in flat.items():
        parts = k.split('.'); d = tree
        for p in parts[:-1]: d = d.setdefault(p, {})
        d[parts[-1]] = v
    return tree


class Guide10:
    def __init__(self, ckpt=None, symm=None, flip=False, nperm=1, batch=512, seed=0):
        vit_dt.set_dtype('float32')
        self.model = vit_dt.ViT(num_layers=8, d_model=72, heads=12, L_eff=25, b=2, transl_invariant=True, two_dimensional=True)
        tmpl = self.model.init(jax.random.PRNGKey(seed), jnp.zeros((1, N), jnp.float32))
        if ckpt:
            tree = nest(load_safetensors(ckpt))
            if 'params' in tree: tree = tree['params']
            params = flax.serialization.from_state_dict(tmpl['params'], tree)
            n_t = len(jax.tree_util.tree_leaves(tmpl['params'])); n_l = len(jax.tree_util.tree_leaves(params))
            assert n_t == n_l, (n_t, n_l)
            for a, b in zip(jax.tree_util.tree_leaves(tmpl['params']), jax.tree_util.tree_leaves(params)):
                assert np.shape(a) == np.shape(b), (np.shape(a), np.shape(b))
        else:
            params = tmpl['params']
        self.params = jax.tree_util.tree_map(lambda p: jnp.asarray(p, jnp.float32), params)
        self.npar = int(sum(np.size(p) for p in jax.tree_util.tree_leaves(self.params)))
        if symm:
            P = np.loadtxt(symm, dtype=np.int64).reshape(-1, N)
        else:                                   # identity, or nperm random permutations for timing only
            rg = np.random.default_rng(1); P = np.stack([np.arange(N)] + [rg.permutation(N) for _ in range(nperm - 1)])
        self.P = jnp.asarray(P); self.flip = flip; self.batch = batch; self.neval = 0
        model = self.model; Pj = self.P; sg = [1.0, -1.0] if flip else [1.0]

        def lz(params, X):
            zs = []
            for s in sg:
                zs.append(jax.lax.map(lambda p: model.apply({'params': params}, s * X[:, p]), Pj))
            z = jnp.concatenate(zs, 0)
            m = jnp.max(jnp.real(z), 0)
            Ssum = jnp.sum(jnp.exp(z - m[None, :]), 0)
            return m + jnp.log(jnp.abs(Ssum)), jnp.angle(Ssum)
        self.lz = jax.jit(lz)

    def eval_spins(self, X):
        n = len(X); B = self.batch; la = np.empty(n); ph = np.empty(n)
        for i in range(0, n, B):
            x = X[i:i + B]; m = len(x)
            if m < B: x = np.concatenate([x, np.repeat(x[:1], B - m, 0)])
            a, p = self.lz(self.params, jnp.asarray(x, jnp.float32)); la[i:i + m] = np.asarray(a)[:m]; ph[i:i + m] = np.asarray(p)[:m]
        self.neval += n
        return la, ph

    def eval(self, S):
        S = np.asarray(S, np.uint64)
        X = 2.0 * ((S[:, None] >> np.arange(N, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.float32) - 1.0
        return self.eval_spins(X)


def local_energy(lat, g, S):
    """complex local energy with the network's own phase, and the binarised-sign energy (phi = 0)."""
    S = np.asarray(S, np.uint64); n = len(S)
    NBs, V = lat.neighbors(S)
    st = np.concatenate([S, NBs[V]]); u, inv = np.unique(st, return_inverse=True)
    la_u, ph_u = g.eval(u); la = la_u[inv]; ph = ph_u[inv]
    lan = np.zeros((n, lat.NB)); phn = np.zeros((n, lat.NB)); lan[V] = la[n:]; phn[V] = ph[n:]
    r = np.where(V, 0.5 * lat.JB[None, :] * np.exp(np.clip(lan - la[:n, None], -60, 60)), 0.0)
    D = lat.diag(V)
    elc = D + np.sum(r * np.exp(1j * (phn - ph[:n, None])), 1)
    return elc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('mode')
    ap.add_argument('--ckpt'); ap.add_argument('--symm'); ap.add_argument('--flip', action='store_true')
    ap.add_argument('--nperm', type=int, default=1); ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--nchains', type=int, default=1024); ap.add_argument('--burn', type=int, default=20)
    ap.add_argument('--nsamp', type=int, default=4); ap.add_argument('--thin', type=int, default=1); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T0 = time.time()
    g = Guide10(a.ckpt, a.symm, a.flip, a.nperm, a.batch)
    lat = F.Lattice(L); log('params', g.npar, 'images', int(g.P.shape[0]) * (2 if a.flip else 1), jax.devices())
    rg = np.random.default_rng(a.seed)
    if a.mode == 'probe':
        S = np.array([np.sum(np.uint64(1) << rg.choice(N, N // 2, replace=False).astype(np.uint64)) for _ in range(4 * a.batch)], np.uint64)
        g.eval(S[:a.batch]); t = time.time(); g.eval(S); dt = time.time() - t
        res = dict(mode='probe', images=int(g.P.shape[0]) * (2 if a.flip else 1), batch=a.batch, evals_per_s=len(S) / dt, device=str(jax.devices()))
    else:
        nch = a.nchains
        S = np.array([np.sum(np.uint64(1) << rg.choice(N, N // 2, replace=False).astype(np.uint64)) for _ in range(nch)], np.uint64)
        la, _ = g.eval(S); out = []; acc = 0; prop = 0
        for sw in range(a.burn + a.nsamp * a.thin):
            for _ in range(N):
                b = rg.integers(lat.NB, size=nch)
                ok = (((S >> lat.BI[b]) ^ (S >> lat.BJ[b])) & np.uint64(1)).astype(bool)
                cand = S ^ lat.MASKS[b]; lc = np.full(nch, -np.inf)
                if ok.any(): lc[ok] = g.eval(cand[ok])[0]
                A = ok & (np.log(rg.random(nch)) < 2 * (lc - la)); S = np.where(A, cand, S); la = np.where(A, lc, la)
                acc += A.sum(); prop += nch
            if sw >= a.burn and (sw - a.burn + 1) % a.thin == 0: out.append(S.copy())
            log(f'sweep {sw + 1} accept {acc / prop:.3f} {time.time() - T0:.0f}s')
        X = np.stack(out).reshape(-1); CH = np.tile(np.arange(nch), len(out))
        E = np.concatenate([local_energy(lat, g, X[i:i + 256]) for i in range(0, len(X), 256)])
        m = np.array([E.real[CH == c].mean() for c in range(nch)])
        res = dict(mode='vmc', n=len(X), E_site=float(E.real.mean() / N), se=float(m.std(ddof=1) / np.sqrt(nch) / N),
                   imag_site=float(E.imag.mean() / N), accept=acc / prop)
        np.savez(os.path.join(a.out, 'vmc10.npz'), X=X, chain=CH, ELr=E.real, ELi=E.imag)
    res['sec'] = time.time() - T0; res['args'] = vars(a)
    json.dump(res, open(os.path.join(a.out, 'vit10.json'), 'w'), indent=1); log('DONE', json.dumps(res))


if __name__ == '__main__':
    main()
