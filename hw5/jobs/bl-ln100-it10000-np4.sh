#!/bin/bash
#
#SBATCH --job-name=bl-ln100-it10000-np4
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=1GB
#SBATCH --output=bl-ln100-it10000-np4.out

module purge
module load openmpi/gnu/4.0.2

mpiexec ../jacobi 200 10000
