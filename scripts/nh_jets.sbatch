#!/bin/bash
#SBATCH --job-name="all_jets"
#SBATCH --time=06:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10

srun uv run nh_jets.py 
