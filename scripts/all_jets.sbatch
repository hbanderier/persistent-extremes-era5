#!/bin/bash
#SBATCH --job-name="all_jets"
#SBATCH --time=4:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem-per-cpu=49G
#SBATCH --cpus-per-task=20

mamba activate env11_2
srun python all_jets.py 
