#!/bin/bash
#SBATCH --job-name=encode_tiny
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=1:05:00
#SBATCH --output=encode_tiny_%j.out
#SBATCH --error=encode_tiny_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate transformer_lm

python3 -m models.tokenizer.encode --dataset tiny --split train
python3 -m models.tokenizer.encode --dataset tiny --split valid 

