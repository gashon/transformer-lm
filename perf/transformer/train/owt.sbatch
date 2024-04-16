#!/bin/bash
#SBATCH --job-name=train_owt
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=90G
#SBATCH --time=4:05:00
#SBATCH --output=train_owt_%j.out
#SBATCH --error=train_owt_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate transformer_lm

python3 train.py \
	--dataset "owt" \
	--vocab_size 32000 \
	--ctx_len 512 \
	--d_model 1028 \
	--num_layers 8 \
	--num_heads 32 \
	--d_ff 4096 \
	--attn_pdrop 0.1 \
	--residual_pdrop 0.1 \
	--lr_max 0.001 \
	--lr_min 0.00001 \
	--t_warmup 500 \
	--t_cos 10000 \
	--train_batch_size 64 \
	--val_batch_size 32 \
	--num_steps 6000 \
	--num_val_batches 10