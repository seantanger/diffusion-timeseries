#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --account=def-wan
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --output=%N-%j.out
cd ~/projects/def-wan/seantang/diffusion-timeseries

module purge
module load python/3.10
source ~/envs/research/bin/activate

python main.py \
    --attention=True \
    --folderdir='./scale' \
    
