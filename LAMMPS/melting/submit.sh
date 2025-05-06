#!/usr/bin/env bash
#SBATCH --nodes 1
#SBATCH --ntasks 36
#SBATCH --cpus-per-task 1
#SBATCH --time=12:00:00
#SBATCH --mem=100000

# Bugfix for oneapi issue
export FI_PROVIDER=verbs

module purge
module load gcc/11.3.0
module load openmpi/4.1.3
module load lammps/20220107-mpi-openmp-plumed

srun lmp -in melting.in > melted.out
