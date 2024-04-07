#!/bin/bash
#SBATCH --job-name=profile_bpe_tiny
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=1:05:00
#SBATCH --output=profile_bpe_tiny_%j.out
#SBATCH --error=profile_bpe_tiny_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate transformer_lm
python3 -m memory_profiler perf/bpe/tiny.py | tee data/log/output-tiny-$(date +%s).log
