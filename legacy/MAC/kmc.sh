#!/bin/bash
#SBATCH --job-name=kmc_mac
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=60G
#SBATCH --partition=standard

module purge
module load gcc/11.3.0 openmpi/4.1.3 python/3.10 lammps/20220107-mpi-openmp-plumed

export FI_PROVIDER=verbs

echo "Starting KMC at $(date)"
python3 kmc_sw.py > kmc.log 2>&1
echo "Finished KMC at $(date)"
