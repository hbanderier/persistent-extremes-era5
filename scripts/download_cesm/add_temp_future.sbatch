#!/bin/bash
#SBATCH --job-name="add_temp_future"
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=1

srun uv run add_temp_future.py
