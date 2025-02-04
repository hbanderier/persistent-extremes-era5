#!/bin/bash
#SBATCH --job-name="download_cesm"
#SBATCH --time=90:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=5

source ~/.bashrc
conda activate env12
srun python new_cesm.py $1 $2 5
