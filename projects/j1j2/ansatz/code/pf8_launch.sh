#!/bin/bash
# Stage + submit the pool_fit8 CPU jobs and arm one durable setsid watcher per job (cluster-jobs skill).  usage: pf8_launch.sh   (prints the job ids it got from sbatch)
set -u
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code; R=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8build/poolfit8
DELTA=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md
mkdir -p $R; sha256sum $C/pool_fit.py $C/pool_fit8_test.py $C/pool_fit8_gate.py $C/pool_fit8_job.sbatch $C/pp2_lemma.py $C/pool_checks.py $C/pf8_watch.sh > $R/code_sha256.txt
# name  cpus  mem  time  stall(s)  question
JOBS="t8abcd:16:32G:03:00:00:2400:1.5h:8x8_checks_a_b_c_d t8e:16:32G:01:30:00:2400:0.5h:8x8_GN_step_timing t8e2:16:100G:01:00:00:1800:0.3h:param_space_solve_timing c6a:8:16G:01:30:00:3600:0.4h:6x6_check_a c6b:8:16G:02:00:00:3600:0.6h:6x6_check_b c6c:8:16G:03:00:00:3600:1.2h:6x6_check_c_pert"
for j in $JOBS; do
  IFS=: read name cpus mem h m s stall exp q <<< "$j"; D=$R/$name; mkdir -p $D
  cat > $D/ANALYSIS_BRIEF.md <<EOB
# pool_fit8 job $name ($q) (build piece (1b), PLAN_pfaffian.md "8x8 BUILD, route 2"; worker-owned, CPU only)
Question: $q.  Sizing (derived from the 2-thread dev runs / the 2026-11-03 baseline logs): about $exp wall against the ${h}:${m} limit, $cpus cpus, $mem (MaxRSS to be read from sacct afterwards).  Code: $C (sha256 in $R/code_sha256.txt).
Files: $D/run.log (stdout, /usr/bin/time -v at the end of each command), $D/test.log (8x8 tests), $D/check_*.json, $D/VERDICT_LINE (the run's own verdict, written last).
Gate: python3 $C/pool_fit8_gate.py $D $name
Next: t8abcd ALL PASS -> report numbers to the lead; FAIL -> the worker diagnoses (no more jobs without a new sizing).  c6a/b/c: pass and deviation from the baseline ~0 -> '6x6 unchanged'; otherwise diagnose.
EOB
  jid=$(sbatch --parsable --job-name=pf8_$name --cpus-per-task=$cpus --mem=$mem --time=$h:$m:00 --export=ALL,WHAT=$name,OUTD=$D $C/pool_fit8_job.sbatch)
  echo "$name $jid" | tee -a $R/job_ids.txt
  STALL=$stall setsid nohup $C/pf8_watch.sh $jid $D > $D/watcher.log 2>&1 < /dev/null &
  echo $! > $D/watcher_pid.txt
done
