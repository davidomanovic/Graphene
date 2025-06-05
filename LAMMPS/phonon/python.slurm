#!/bin/bash
#SBATCH --job-name=g6_profile
#SBATCH --output=g6_%j.out
#SBATCH --error=g6_%j.err
#SBATCH --nodes 1
#SBATCH --ntasks 36
#SBATCH --cpus-per-task 1
#SBATCH --time=04:00:00
#SBATCH --mem=190000

# load the required toolchain + Python
module load gcc/11.3.0
module load python/3.10.4

# activate your venv (if you have one)
source $HOME/envs/hexatic/bin/activate

python phono.py
