#!/bin/bash
#SBATCH --job-name=train_tiny
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=90G
#SBATCH --time=4:05:00
#SBATCH --output=train_tiny_%j.out
#SBATCH --error=train_tiny_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate transformer_lm

python3 train.py \
  --dataset "tiny" \
  --vocab_size 10000 \
  --ctx_len 256 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 16 \
  --d_ff 2048 \
  --attn_pdrop 0.1 \
  --residual_pdrop 0.1 \
  --lr_max 0.001 \
  --lr_min 0.00001 \
  --t_warmup 500 \
  --t_cos 10000 \
  --epochs 100 \
  --train_batch_size 32 \
  --val_batch_size 32 \
  --num_train_batches 50 \
  --num_val_batches 10

