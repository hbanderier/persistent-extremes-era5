#!/bin/bash
#SBATCH --job-name="all_jets_cesm"
#SBATCH --time=14:00:00
#SBATCH --partition=epyc2
#SBATCH --mem-per-cpu=30G
#SBATCH --cpus-per-task=33

mamba activate env11_2
srun python all_jets_cesm.py 
