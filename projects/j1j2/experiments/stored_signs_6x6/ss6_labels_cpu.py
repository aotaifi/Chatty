"""CPU shard of the K1 training labels r0(x) (Marshall, |psi_ViT|) on the stored-sign training states.
Usage: python ss6_labels_cpu.py BETA SHARD NSHARD   (writes labels_k1_b{BETA}_s{i}of{n}.npz to $SS6_DATA)."""
import sys, os, time
import numpy as np
import ss6_core as S
import ll6_core as C
beta, i, n = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
net = C.Net(S.CKPT, dtype='float32', batch=4096)
amp = S.AmpCache(net, cap=10_000_000)
Xs = np.load(f'samples_b{beta}.npz')['train'].reshape(-1)
U = S.build_train_states(Xs, 1)
lo = len(U) * i // n; hi = len(U) * (i + 1) // n; Ui = U[lo:hi]
t0 = time.time(); r = np.empty(len(Ui))
for j in range(0, len(Ui), 4096):
    r[j:j + 4096], _ = S.r_onehop(Ui[j:j + 4096], amp, C.marshall_vec)
    if (j // 4096) % 20 == 0: S.log('SHARD', i, j, len(Ui), round(time.time() - t0), amp.nnew)
out = os.path.join(os.environ.get('SS6_DATA', '.'), S.label_shard_name(beta, 1, i, n))
np.savez(out, U=Ui, r=r, T=C.T_FN); S.log('DONE', out, len(Ui), time.time() - t0, amp.nnew)
