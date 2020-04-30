#!/bin/bash
#
#SBATCH --job-name=bl-n25600-ln1600-it100-np256
#SBATCH --nodes=16
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=1GB
#SBATCH --output=bl-n25600-ln1600-it100-np256.out

module purge
module load openmpi/gnu/4.0.2

mpiexec ../jacobi 25600 100
