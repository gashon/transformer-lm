#!/bin/bash
#SBATCH --job-name=profile_bpe
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=00:35:00
#SBATCH --output=profile_bpe_%j.out
#SBATCH --error=profile_bpe_%j.err

conda activate transformer_lm
python3 -m memory_profiler profile.py | tee data/log/output-$(date +%s).log
