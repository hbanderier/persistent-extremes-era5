#!/bin/bash
#SBATCH --job-name="download_cesm"
#SBATCH --time=90:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=20

source ~/.bashrc
conda activate env11_2
srun python new_cesm.py $1 $2 20
