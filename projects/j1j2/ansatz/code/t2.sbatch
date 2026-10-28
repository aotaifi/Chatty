#!/bin/bash
#SBATCH --job-name=t2_cplx
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6/runs/t2_cplx/slurm-%j.out
# Test T2 (PLAN_pfaffian.md): cx_h1.py (sha b555f9ba, codeH1, unchanged) three times in sequence, one per sub-dir s1,s2,s3, as h1.sbatch. Default QoS (no express).
set -uo pipefail
RUN=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6/runs/t2_cplx
export ST6_CSR=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6/data/csr ST6_TABLE=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6/data/psi0_6x6_table.npz ST6_SYM=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6/data/sym_tables.npz CX_LOOP8=/unused
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=8
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; } >> $RUN/run_metadata.txt
cd /scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6/codeH1
rcall=0
for s in s1 s2 s3; do
  CX_TABDIR=$RUN/$s/tables /scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6/../venv_jax_0.8.2/bin/python cx_h1.py $RUN/$s $RUN/$s/spec_h1.json; rc=$?
  echo "exit[$s]=$rc finished=$(date -Is)" >> $RUN/run_metadata.txt
  [ $rc -ne 0 ] && rcall=$rc
done
exit $rcall
