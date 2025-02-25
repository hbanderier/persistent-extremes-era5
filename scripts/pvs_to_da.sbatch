#!/bin/bash
#SBATCH --job-name="pvs_to_da"
#SBATCH --time=10:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem-per-cpu=50G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate env11_2
srun python pvs_to_da.py 
