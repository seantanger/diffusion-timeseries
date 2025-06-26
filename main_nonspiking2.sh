#!/bin/bash
#SBATCH --time=144:00:00
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=4  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --output=%N-%j.out

#SBATCH --mail-user=s222tang@uwaterloo.ca
#SBATCH --mail-type=ALL

#SBATCH -o %N-%j_NS2.out # File to which STDOUT will be written

cd ~/projects/diffusion-timeseries

source ~/envs/research/bin/activate

python -u main.py \
    --train --tune \
    --dataset='GBM' --batch_size=128 --sigma=0.3 --mu=0.05 --epochs=400 --n_samples=500000 \
    --folderdir='./results_nonspiking2' \
    --resume_model='parameters_gbm/400unet_mu=0.05_sigma=0.3_t=0.5.pt' --resume