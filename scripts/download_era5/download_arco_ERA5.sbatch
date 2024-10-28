#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --partition=epyc2
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-63

declare -i year=SLURM_ARRAY_TASK_ID+1959
srun python download_arco_ERA5.py ${year} 
