#!/bin/bash
# hop-aware training: stages hopa, hopa_pert, hopb in parallel for one class.  usage: hop4x4_driver.sh RUNDIR MASK SRCDIR
D=$1; export MASK=$2; SRC=$3; S=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments/gfmc_fn_lattice/l8/l8_loop4x4.py
source /usr/share/Modules/init/bash 2>/dev/null || source /usr/share/modules/init/bash; module load python/3.12-2024.10
export JAX_PLATFORMS=cpu MAXEV_HOP=${MAXEV_HOP:-20000}
mkdir -p $D; cp $SRC/flat_opt.npy $SRC/flat_ii_5.npy $D/; sha256sum $S > $D/code_sha256.txt; echo "host $(hostname) start $(date -Is)" > $D/run_metadata.txt
for v in hopa hopa_pert hopb; do python3 $S $D $v > $D/$v.log 2>&1 & done; wait
echo "end $(date -Is)" >> $D/run_metadata.txt
[ $(grep -c "VERDICT hop" $D/progress.txt) = 3 ] && echo COMPLETED > $D/DRIVER_DONE || echo FAILED > $D/DRIVER_DONE
