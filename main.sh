#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --account=def-wan

module load python/3.10
source ~/envs/research/bin/activate

python main.py