#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --partition=epyc2
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-40
declare -i year=SLURM_ARRAY_TASK_ID+2059

source ~/.bashrc
conda activate env12
srun python old_cesm.py ${year} $1