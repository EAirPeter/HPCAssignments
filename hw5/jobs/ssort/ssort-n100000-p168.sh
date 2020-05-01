#!/bin/bash
#
#SBATCH --job-name=ssort-n100000-p168
#SBATCH --nodes=12
#SBATCH --tasks-per-node=14
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=4GB
#SBATCH --output=ssort-n100000-p168.out

module purge
module load openmpi/gnu/4.0.2

mpiexec ../../ssort 100000
