#!/bin/bash
#
#SBATCH --job-name=nb-n100-ln100-it10000-np1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=10GB
#SBATCH --output=nb-n100-ln100-it10000-np1.out

module purge
module load openmpi/gnu/4.0.2

mpiexec ../jacobi-nb 100 10000
