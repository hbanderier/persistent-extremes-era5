#!/bin/bash
#SBATCH --job-name="add_temp_future"
#SBATCH --time=90:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem-per-cpu=50G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate env12
srun python add_temp_future.py
