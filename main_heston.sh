#!/bin/bash
#SBATCH --time=144:00:00
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --output=%N-%j_heston1.out

#SBATCH --mail-user=s222tang@uwaterloo.ca
#SBATCH --mail-type=ALL
cd ~/projects/diffusion-timeseries

source ~/envs/research/bin/activate

python -u main.py \
    --spiking=True --train \
    --dataset='Heston' --batch_size=32 --theta=0.1 --mu=0.05 --epochs=400 --n_samples=100000 \
    --folderdir='./results_heston_initial1' \
    --resume_model='parameters_heston/1200spiking_unet_mu=0.05_theta=0.1_t=0.5v3.pt' # --resume
    
