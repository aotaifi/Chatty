#!/bin/bash
# Gate over the evaluation chains (run after all chains of a group have their VERDICT): per parameter set and projection,
# quotable.py series on the two independent chains; refuses a set whose VERDICT_RUN lacks eval_rc=0 or whose E_s file is missing.
LB=/scratch/a/A.Otaifi/rbmpp8/rb8_retrain; Q=/project/theorie/a/A.Otaifi/tools/quotable.py
for G in "s306 simple 41 42" "s307 simple 43 44" "s306 full 51 52" "s307 full 53 54"; do set -- $G
  F=""; ok=1
  for s in $3 $4; do d=$LB/$1/ev_$2_$s; grep -q "eval_rc=0" $d/VERDICT_RUN 2>/dev/null && [ -f $d/eval/E_s$s.npy ] || ok=0; F="$F $d/eval/E_s$s.npy"; done
  [ $ok = 1 ] || { echo "$1 $2: INCOMPLETE (eval_rc != 0 or E_s file missing)"; continue; }
  python3 $Q series $F --name "E/N 8x8 RBM+PP retrained $1, '$2' projection" --plot $LB/$1/quotable_$2.png | tail -3
done
