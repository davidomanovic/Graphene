#!/usr/bin/env bash
#SBATCH --job-name=physicsproject
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 9
#SBATCH --time=12:00:00
#SBATCH --mem=100000

module purge
module load gcc/11.3.0
module load openmpi/4.1.3
module load lammps/20220107-mpi-openmp-plumed

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on $SLURM_JOB_NODELIST : $SLURM_NTASKS MPI ranks × $OMP_NUM_THREADS threads"
srun lmp -in melting.in > melted.out
