#!/bin/bash
#SBATCH --job-name="all_jets"
#SBATCH --time=10:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem-per-cpu=49G
#SBATCH --cpus-per-task=20

source ~/.bashrc
conda activate env12
srun python nh_jets.py 
