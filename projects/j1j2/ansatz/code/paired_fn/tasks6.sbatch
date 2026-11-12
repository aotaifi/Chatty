#!/bin/bash
#SBATCH --job-name=pfn-tasks6
#SBATCH --partition=cluster
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
# run a task file of run6.py argument lines (one 6x6 ensemble per line), PAR single-threaded processes sharing the mmap'd neighbour tables (page cache).
# usage: sbatch --cpus-per-task=P --mem=.. --time=.. --output RUN/tasks.log tasks6.sbatch RUN TASKFILE   (line: "SUBDIR run6-args...")
set -u
RUN=$1; TF=$2; C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code/paired_fn; P=${SLURM_CPUS_PER_TASK}
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
echo "host $(hostname) job $SLURM_JOB_ID $(date -Is) tasks $(wc -l < $TF) par $P"; sha256sum $C/*.py $TF
cat /scratch/a/A.Otaifi/ansatz_runs/c2_t0b_6x6/tables/nb_J1.npy /scratch/a/A.Otaifi/ansatz_runs/c2_t0b_6x6/tables/nb_J2.npy > /dev/null; echo "PROGRESS tables warm $(date -Is)"
export PY C RUN
grep -v '^\s*$' $TF | xargs -P $P -L 1 bash -c 'd=$0; mkdir -p $RUN/$d; $PY $C/run6.py $RUN/$d "$@" > $RUN/$d/run6.log 2>&1; echo "PROGRESS task $d exit $? $(date -Is) $(tail -1 $RUN/$d/run6.log | cut -c1-200)"'
n=$(grep -l RUN6_DONE $RUN/*/run6.log 2>/dev/null | wc -l)
echo "TASKS6_DONE ok=$n of $(grep -vc '^\s*$' $TF) $(date -Is)"; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P
