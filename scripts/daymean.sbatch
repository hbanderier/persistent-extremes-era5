#!/bin/bash
#SBATCH --job-name="Daymean"
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=16

# Your code below this line
srun bash daymean.sh $1
