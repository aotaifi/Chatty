#!/bin/bash
# Gate for the A40 pair s307 'full' seeds 55 56 (same checks and quotable.py call as eval_combine.sh; the original is untouched).
LB=/scratch/a/A.Otaifi/rbmpp8/rb8_retrain; Q=/project/theorie/a/A.Otaifi/tools/quotable.py; F=""; ok=1
for s in 55 56; do d=$LB/s307/ev_full_$s; grep -q "eval_rc=0" $d/VERDICT_RUN 2>/dev/null && [ -f $d/eval/E_s$s.npy ] || ok=0; F="$F $d/eval/E_s$s.npy"; done
[ $ok = 1 ] || { echo "s307 full 55/56: INCOMPLETE (eval_rc != 0 or E_s file missing)"; exit 1; }
python3 $Q series $F --name "E/N 8x8 RBM+PP retrained s307, 'full' projection (A40 pair 55/56)" --plot $LB/s307/quotable_full_a40.png | tail -3
