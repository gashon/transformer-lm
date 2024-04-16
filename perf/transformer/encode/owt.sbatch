#!/bin/bash
#SBATCH --job-name=encode_owt
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=4:05:00
#SBATCH --output=encode_owt_%j.out
#SBATCH --error=encode_owt_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate transformer_lm

python3 -m models.tokenizer.encode --dataset owt --split train
python3 -m models.tokenizer.encode --dataset owt --split valid
