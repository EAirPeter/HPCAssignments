#!/bin/bash
#
#SBATCH --job-name=nb-ln100-it10000-np256
#SBATCH --nodes=16
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=1GB
#SBATCH --output=nb-ln100-it10000-np256.out

module purge
module load openmpi/gnu/4.0.2

mpiexec ../jacobi-nb 1600 10000
