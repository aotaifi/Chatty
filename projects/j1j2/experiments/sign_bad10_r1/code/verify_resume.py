"""verify_resume.py OLD_E_MIX.npy NEW_RUN_DIR ORIG_NBLOCKS BURN : proof that a resume read its checkpoint: (1) 'RESUMED at block N' with N = ORIG_NBLOCKS in progress.txt, (2) block numbers continue N+1.. (no restart at 1),
(3) the new E_mix.npy has the old post-burn series as an exact prefix (checkpointed rec restored), (4) new blocks differ from a cold replay (first new block != first block of the run)."""
import sys, re, numpy as np
old = np.load(sys.argv[1]); d = sys.argv[2]; n0 = int(sys.argv[3]); burn = int(sys.argv[4]); txt = open(d + '/progress.txt').read(); new = np.load(d + '/E_mix.npy')
m = re.search(r'RESUMED at block (\d+)', txt); ok1 = bool(m) and int(m.group(1)) == n0
tail = txt.split('RESUMED at block')[-1]; bl = [int(x) for x in re.findall(r'PROGRESS block (\d+)/', tail)]; ok2 = len(bl) > 0 and bl[0] == n0 + 1
ok3 = len(new) > len(old) and np.array_equal(new[:len(old)], old); ok4 = ok3 and abs(new[len(old)] - old[0]) > 0
print(f'RESUME_CHECK resumed_line={ok1} (block {m.group(1) if m else None}, expect {n0}) first_new_block={bl[:1]} (expect {n0 + 1}) prefix_equal={ok3} old_len={len(old)} new_len={len(new)} not_cold_replay={ok4}')
print('RESUME_PROOF', 'PASS' if (ok1 and ok2 and ok3 and ok4) else 'FAIL'); sys.exit(0 if (ok1 and ok2 and ok3 and ok4) else 3)
