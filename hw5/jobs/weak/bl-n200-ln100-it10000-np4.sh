#!/bin/bash
#
#SBATCH --job-name=bl-n200-ln100-it10000-np4
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=10GB
#SBATCH --output=bl-n200-ln100-it10000-np4.out

module purge
module load openmpi/gnu/4.0.2

mpiexec ../../jacobi 200 10000
