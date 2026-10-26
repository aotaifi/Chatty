#!/bin/bash
# driver for one class: opt, then the three loop variants in parallel.  usage: loop4x4_driver.sh RUNDIR MASK
D=$1; export MASK=$2; S=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments/gfmc_fn_lattice/l8/l8_loop4x4.py
source /usr/share/Modules/init/bash 2>/dev/null || source /usr/share/modules/init/bash; module load python/3.12-2024.10
export JAX_PLATFORMS=cpu MAXEV_OPT=${MAXEV_OPT:-100000} MAXEV_FIT=${MAXEV_FIT:-50000} TOL_OPT=${TOL_OPT:-1e-13} TOL_FIT=${TOL_FIT:-1e-13}
mkdir -p $D; sha256sum $S > $D/code_sha256.txt; echo "host $(hostname) start $(date -Is) MASK $MASK" > $D/run_metadata.txt
python3 $S $D opt > $D/opt.log 2>&1 || { echo "opt exit $?" >> $D/run_metadata.txt; echo FAILED > $D/DRIVER_DONE; exit 1; }
for v in loopi loopii loopiii; do python3 $S $D $v > $D/$v.log 2>&1 & done; wait
echo "end $(date -Is)" >> $D/run_metadata.txt
n=$(grep -c "VERDICT LOOP" $D/progress.txt); [ "$n" = 3 ] && echo COMPLETED > $D/DRIVER_DONE || echo FAILED > $D/DRIVER_DONE
