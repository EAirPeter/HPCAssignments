#!/bin/bash
#
#SBATCH --job-name=bl-n100-ln100-it10000-np1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=1GB
#SBATCH --output=bl-n100-ln100-it10000-np1.out

module purge
module load openmpi/gnu/4.0.2

mpiexec ../jacobi 100 10000
