# rb8_pgprobe analysis brief: cost probe for retraining the 8x8 RBM+PP (M1) with the published point-group projection

Question: warm-started from rb8_prod params_final.npz (sha 0520c5be; translations x flip, VMC -0.498671(3)), what is (i) the
energy per site of the SAME parameters under the 'simple' projection (translations x {E, C4} x spin flip, 256 images; the
published 8x8 benchmark projection of Nomura-Imada 2005.14142 App. B) and under 'full' (whole space group x flip, 1024 images),
(ii) the H100 seconds per SR step at 8192 samples for each, (iii) the energy trend over 100 SR steps (lr 0.02, shift 0.01) under 'simple'.
Not a production number: the brief forbids quoting anything from this as converged.

Files (run dir, local copy of Paderborn .../rbmpp8/runs/rb8_pgprobe): a0/train.json (simple, lr 0, 20 steps), a/train.json
(simple, SR, 100 steps), b/train.json (full, lr 0, 12 steps); hist[it] = {E per site, err, var, t_step, ...}. run_metadata.txt
has code sha256 and per-stage rc; VERDICT_RUN last line "DONE_RBMPP8 exit=0 job=... stages: a0=rc a=rc b=rc" (rc 124 = the
stage's hard timeout fired; the steps written before it are still valid). The job's .out (rb8pg-<job>.out) has PROGRESS lines.

Gate (no quotable.py: one chain per stage, not independent chains; label every number "probe, single chain"):
python3 -c "import json,numpy as np,sys
for s in 'a0','a','b':
  try: h=json.load(open(s+'/train.json'))['hist']
  except Exception as e: print(s,'missing',e); continue
  E=np.array([x['E'] for x in h]); t=np.array([x['t_step'] for x in h])
  print(s,'steps',len(h),'E mean',E.mean(),'+-',E.std(ddof=1)/np.sqrt(len(E)),'first10',E[:10].mean(),'last20',E[-20:].mean(),'t_step median(skip 3)',np.median(t[3:]))"
Next step per verdict (the worker that submitted it reports to its lead; no further runs from this brief):
- a0 E below -0.498671 by more than 3 sigma -> the projection alone gains; report the gain and s/step into the proposal.
- a0 not below -> the point group gains nothing without retraining; the trend in stage a decides.
- any stage missing or rc != 0/124 -> report which, from the .out.
