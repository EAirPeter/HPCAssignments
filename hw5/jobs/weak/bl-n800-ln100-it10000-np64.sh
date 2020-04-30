#!/bin/bash
#
#SBATCH --job-name=bl-n800-ln100-it10000-np64
#SBATCH --nodes=4
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=10GB
#SBATCH --output=bl-n800-ln100-it10000-np64.out

module purge
module load openmpi/gnu/4.0.2

mpiexec ../jacobi 800 10000
