#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --account=def-wan
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --output=%N-%j.out
cd ~/projects/def-wan/seantang/diffusion-timeseries

module purge
module load python/3.10
source ~/envs/research/bin/activate

python -u main_nonspiking.py \
    --train --batch_size=128 --sigma=0.5 --mu=0.05 --epochs=400 --n_samples=100000 \
    --folderdir='./results_nonspiking3' \
    --resume_model='parameters_T=0.5/1200unet_mu=0.05_sigma=0.5_t=0.5.pt' --resume
    
