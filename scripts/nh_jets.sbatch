#!/bin/bash
#SBATCH --job-name="all_jets"
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem=150G
#SBATCH --cpus-per-task=10

srun uv run nh_jets.py 
