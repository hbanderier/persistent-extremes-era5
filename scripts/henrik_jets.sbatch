#!/bin/bash
#SBATCH --job-name="henrik_jets"
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem=100G
#SBATCH --cpus-per-task=5

srun uv run henrik_jets.py 
