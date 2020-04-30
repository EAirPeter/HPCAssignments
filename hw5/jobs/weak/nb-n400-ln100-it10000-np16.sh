#!/bin/bash
#
#SBATCH --job-name=nb-n400-ln100-it10000-np16
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=1GB
#SBATCH --output=nb-n400-ln100-it10000-np16.out

module purge
module load openmpi/gnu/4.0.2

mpiexec ../jacobi-nb 400 10000
