#!/bin/bash
#
#SBATCH --job-name=ssort-n1000000-p256
#SBATCH --nodes=16
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=4GB
#SBATCH --output=ssort-n1000000-p256.out

module purge
module load openmpi/gnu/4.0.2

mpiexec ../../ssort 1000000
