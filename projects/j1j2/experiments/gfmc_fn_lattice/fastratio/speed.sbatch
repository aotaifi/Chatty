#!/bin/bash
#SBATCH --job-name=fastratio
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:24:00
#SBATCH --no-requeue
# fast-vs-ref speed + equivalence on one full theorie A40 (Paderborn H100 est. start > 1 day).  usage: sbatch speed.sbatch RUN
bash $1/code/job_speed.sh $1
