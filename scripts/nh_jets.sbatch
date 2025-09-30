#!/bin/bash
#SBATCH --job-name="all_jets"
#SBATCH --time=00:20:00
#SBATCH --partition="epyc2"
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10

source ~/.bashrc
conda activate env12
srun python nh_jets.py 
