#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --partition=epyc2
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-11

declare -i year=SLURM_ARRAY_TASK_ID+2069
srun python download_cesm.py ${year} 
