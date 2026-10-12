"""Post-processing of s8_twohop.py chunks.
  python s8_post.py train RUN_DIR OUT.npz
      training/held-out states = samples x and ALL their valid neighbours (L1 per sample) with r_V (ViT sign) and r_M
      (Marshall) Krylov inputs, log a, the ViT sign, Marshall, origin chain, is-sample flag.  CPU only.
  python s8_post.py ref RUN_DIR OUT.npz [name=net.pkl ...]
      per-sample and per-edge quantities (s8_q.onehop_from_chunk) for the bases V (ViT sign), M (Marshall) and any
      stored sign nets given (evaluated on the full two-hop set U2; needs jax for nets).
"""
import sys, os, glob, time, json
import numpy as np
import s8_q as Q

mode, RUN, OUT = sys.argv[1:4]
nets = dict(a.split('=') for a in sys.argv[4:])
files = sorted(glob.glob(os.path.join(RUN, 'chunk_*.npz')))
t0 = time.time()
if nets:
    import s8_net
    nets = {k: s8_net.SignNet.load(v) for k, v in nets.items()}

if mode == 'train':
    parts = []
    for f in files:
        d = np.load(f); S = d['X']; U2 = d['U2']
        signs = dict(V=d['s'], M=Q.marshall(U2))
        L1, r1, la1, s1, x, e = Q.onehop_from_chunk(S, U2, d['la'], signs)
        # origin chain of every L1 state: chain of the first sample whose neighbourhood contains it
        NBs, V = Q.neighbors(S)
        own, col = np.nonzero(V)
        st = np.concatenate([S, NBs[own, col]]); org = np.concatenate([np.arange(len(S)), own])
        u, first = np.unique(st, return_index=True)
        assert np.array_equal(u, L1)
        parts.append(dict(S=L1, r_V=r1['V'], r_M=r1['M'], la=la1, s_V=s1['V'], M=s1['M'],
                          chain=d['chain'][org[first]], is_x=np.isin(L1, S)))
        print(f, len(L1), f'{time.time() - t0:.0f}s', flush=True)
    P = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    u, first = np.unique(P['S'], return_index=True)
    P = {k: v[first] for k, v in P.items()}
    np.savez(OUT, **P)
    print('states', len(P['S']), 'samples', int(P['is_x'].sum()))
else:
    qs = []; tim = dict(net=0.0, net_states=0)
    for f in files:
        d = np.load(f); S = d['X']; U2 = d['U2']
        signs = dict(V=d['s'], M=Q.marshall(U2))
        for k, n in nets.items():
            t = time.time(); signs[k] = n.sign(U2); tim['net'] += time.time() - t; tim['net_states'] += len(U2)
        L1, r1, la1, s1, x, e = Q.onehop_from_chunk(S, U2, d['la'], signs)
        q = dict(x); q.update(e); q['chain'] = d['chain']; q['X'] = S
        q['nU2'] = np.full(len(S), len(U2) / len(S)); q['nL1'] = np.full(len(S), len(L1) / len(S))
        qs.append(q)
        print(f, len(U2), f'{time.time() - t0:.0f}s', json.dumps(tim), flush=True)
    Qa = Q.concat_q(qs)
    np.savez(OUT, **Qa, net_sec=tim['net'], net_states=tim['net_states'])
    print('samples', len(Qa['D']), 'edges', len(Qa['own']), json.dumps(tim))
