#!/bin/bash
#SBATCH --job-name="download_cesm"
#SBATCH --time=90:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem-per-cpu=30G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate env12
srun python download_for_cra.py
