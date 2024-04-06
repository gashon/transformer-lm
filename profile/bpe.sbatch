#!/bin/bash
#SBATCH --job-name=profile_bpe
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=00:35:00
#SBATCH --output=profile_bpe_%j.out
#SBATCH --error=profile_bpe_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate transformer_lm
python3 -m memory_profiler bpe.py | tee data/log/output-$(date +%s).log
