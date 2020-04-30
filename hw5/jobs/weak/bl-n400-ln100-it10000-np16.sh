#!/bin/bash
#
#SBATCH --job-name=bl-n400-ln100-it10000-np16
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=10GB
#SBATCH --output=bl-n400-ln100-it10000-np16.out

module purge
module load openmpi/gnu/4.0.2

mpiexec ../../jacobi 400 10000
